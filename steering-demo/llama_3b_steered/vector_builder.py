#!/usr/bin/env python3
"""
Vector Builder: Compute steering vectors from contrast examples
HasanLabs - Mechanistic Interpretability Workshop

Compute ACTUAL steering vectors by:
  1) Running positive/negative examples through the model
  2) Collecting block outputs at a chosen layer (float32)
  3) Mean pooling per example
  4) mean(pos) - mean(neg), then L2-normalize
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


@dataclass
class VectorConfig:
    name: str
    layer_idx: int
    positive_examples: List[str]
    negative_examples: List[str]
    strength: float = 0.6
    batch_size: int = 4
    max_length: int = 128
    pooling: str = "mean"   # "mean" | "last" | "window"
    pool_window: int = 16


# ---------- helper: robust block discovery on HF models ----------
def _find_blocks(model: torch.nn.Module):
    m = model
    candidates = [
        "model.layers",              # LLaMA/Phi
        "model.decoder.layers",      # some decoders
        "model.transformer.h",       # GPT-2
        "gpt_neox.layers",           # NeoX
    ]
    for path in candidates:
        cur = m
        ok = True
        for attr in path.split("."):
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok and isinstance(cur, (torch.nn.ModuleList, list)):
            return cur
    raise RuntimeError("Could not locate transformer block list on this model.")


class ActivationCollector:
    """Collect block outputs at a specific layer via a forward hook."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._bucket: List[torch.Tensor] = []

    def register(self, layer_idx: int, mode: str = "pre") -> None:
        """Register hook to collect activations.
        
        Args:
            layer_idx: Which layer to hook
            mode: "pre" for pre-block (more stable), "post" for post-block
        """
        blocks = _find_blocks(self.model)
        n = len(blocks)
        if not (0 <= layer_idx < n):
            raise IndexError(f"Layer {layer_idx} out of range (0..{n-1})")

        if mode == "pre":
            # Pre-hook: capture input to block (residual stream)
            def pre_hook(_module, inputs):
                # inputs is a tuple; first item is hidden_states [B, T, D]
                if isinstance(inputs, tuple) and len(inputs) > 0:
                    hs = inputs[0]
                else:
                    hs = inputs
                    
                if torch.is_tensor(hs):
                    # Store as float32 on CPU for stability
                    self._bucket.append(hs.detach().to(torch.float32, copy=True).cpu())
            
            h = blocks[layer_idx].register_forward_pre_hook(pre_hook)
        else:
            # Post-hook: capture output from block (current behavior)
            def hook(_module, _inputs, output):
                hs = output[0] if (isinstance(output, tuple) and torch.is_tensor(output[0])) else output
                if torch.is_tensor(hs):
                    # store as float32 on CPU for stability
                    self._bucket.append(hs.detach().to(torch.float32, copy=True).cpu())
            
            h = blocks[layer_idx].register_forward_hook(hook)
        
        self._hooks.append(h)

    def pop_last(self) -> Optional[torch.Tensor]:
        if not self._bucket:
            return None
        return self._bucket.pop()
    
    def get_last(self) -> Optional[torch.Tensor]:
        """Get last activation without removing it"""
        if not self._bucket:
            return None
        return self._bucket[-1]

    def clear(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
        self._bucket.clear()


class SteeringVectorBuilder:
    """Builds steering vectors from contrast examples (model-agnostic)."""

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        # Device/dtype for runtime model
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # IMPORTANT: keep real modules (no accelerate offload/meta)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=False,
        ).to(self.device)
        self.model.eval()

        # cache dims
        self.hidden_size = int(self.model.config.hidden_size)
        self.num_layers = int(self.model.config.num_hidden_layers)

    # -------- activation pooling --------

    def _pool_examples(
        self,
        texts: List[str],
        layer_idx: int,
        batch_size: int = 4,
        max_length: int = 128,
        pooling: str = "mean",
        pool_window: int = 16,
    ) -> torch.Tensor:
        """
        Return per-example pooled activations (float32) and then mean across examples.
        """
        coll = ActivationCollector(self.model)
        # Use pre-hook for more stable residual stream capture
        coll.register(layer_idx, mode="pre")

        running_sum: Optional[torch.Tensor] = None
        n_examples = 0

        print(f"Processing {len(texts)} texts in batches of {batch_size}")
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i : i + batch_size]
                if not batch:
                    continue

                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)

                _ = self.model(**enc)

                # Debug: check bucket status
                print(f"  Batch {i//batch_size}: Bucket size = {len(coll._bucket)}")
                
                # The hook ran once for the whole batch; fetch that tensor
                hs = coll.pop_last()  # [B, T, D] on CPU float32
                if hs is None:
                    print(f"WARNING: No activation collected for batch {i//batch_size}")
                    continue
                
                # Sanitize non-finite values immediately
                if not torch.isfinite(hs).all():
                    nonfinite = (~torch.isfinite(hs)).sum().item()
                    print(f"  WARNING: Batch {i//batch_size} has {nonfinite} non-finite values, sanitizing...")
                    # Zero out non-finite values
                    hs = torch.where(torch.isfinite(hs), hs, torch.zeros_like(hs))
                
                # Optional: Clamp to prevent future overflows
                hs = hs.clamp(-1e4, 1e4)
                    
                # Debug: check if activations are non-zero
                if torch.allclose(hs, torch.zeros_like(hs)):
                    print(f"WARNING: Activations are all zeros for batch {i//batch_size}")

                # Attention mask to exclude padding
                mask = enc.get("attention_mask", None)
                if mask is None:
                    mask = torch.ones(hs.shape[:2], dtype=torch.long)
                else:
                    mask = mask.detach().to("cpu")

                # Per-example pooling
                B, T, D = hs.shape
                pooled_list: List[torch.Tensor] = []
                for b in range(B):
                    valid = int(mask[b].sum().item())
                    valid = max(valid, 1)
                    if pooling == "last":
                        idx = valid - 1
                        pooled = hs[b, idx, :]
                    elif pooling == "window":
                        w = min(pool_window, valid)
                        pooled = hs[b, valid - w : valid, :].mean(dim=0)
                    else:  # mean
                        pooled = hs[b, :valid, :].mean(dim=0)
                    pooled_list.append(pooled)

                # Accumulate in float64 for numerical stability
                batch_sum = torch.stack(pooled_list, dim=0).to(torch.float64).sum(dim=0)  # [D]
                if running_sum is None:
                    running_sum = batch_sum.clone()
                else:
                    running_sum = running_sum.add_(batch_sum)
                n_examples += len(pooled_list)

        coll.clear()

        if running_sum is None or n_examples == 0:
            raise ValueError("No activations collected; check inputs and layer index.")

        # Compute mean and convert back to float32
        result = (running_sum / float(n_examples)).to(torch.float32)  # [D], float32
        
        # Final sanity check
        if not torch.isfinite(result).all():
            print(f"WARNING: Non-finite values in pooled result!")
            print(f"  running_sum has non-finite: {(~torch.isfinite(running_sum)).sum().item()}")
            print(f"  n_examples: {n_examples}")
            # Last resort: zero out non-finite
            result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
            
        return result

    # -------- public API --------

    def build_steering_vector(self, cfg: VectorConfig) -> torch.Tensor:
        print(f"\nBuilding '{cfg.name}' steering vector...")
        print(f"Layer: {cfg.layer_idx}")
        print(f"Positive examples: {len(cfg.positive_examples)}")
        print(f"Negative examples: {len(cfg.negative_examples)}")

        pos = self._pool_examples(
            cfg.positive_examples,
            layer_idx=cfg.layer_idx,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            pooling=cfg.pooling,
            pool_window=cfg.pool_window,
        )
        neg = self._pool_examples(
            cfg.negative_examples,
            layer_idx=cfg.layer_idx,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            pooling=cfg.pooling,
            pool_window=cfg.pool_window,
        )

        # Debug: check if pos and neg are too similar
        print(f"Positive mean: {pos.mean():.6f}, std: {pos.std():.6f}")
        print(f"Negative mean: {neg.mean():.6f}, std: {neg.std():.6f}")
        print(f"Cosine similarity: {F.cosine_similarity(pos.unsqueeze(0), neg.unsqueeze(0)).item():.6f}")
        
        v = pos - neg
        print(f"Raw diff norm before cleanup: {torch.linalg.vector_norm(v):.6f}")
        
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(torch.linalg.vector_norm(v))
        
        if norm < 1e-6:
            print(f"WARNING: Vector norm is near zero ({norm:.8f})")
            print("This means positive and negative examples produced nearly identical activations!")
        
        v = v / max(norm, 1e-6)

        print(f"Vector shape: {tuple(v.shape)}")
        print(f"Vector norm: {float(torch.linalg.vector_norm(v)):.4f}")
        return v.to(self.device)

    def evaluate_vector(
        self,
        vector: torch.Tensor,
        layer_idx: int,
        prompt: str,
        alpha: float = 0.6,
        gen_kwargs: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """
        Quick qualitative check of the vector effect on a single prompt.
        """
        from .steering_vectors import SteeredLlama, SteeringVector as SV

        runner = SteeredLlama(model_name=self.model.name_or_path if hasattr(self.model, "name_or_path") else self.model.config._name_or_path)
        base = runner.generate(prompt, **(gen_kwargs or {}))

        sv = SV(vector=vector, layer_idx=int(layer_idx), strength=float(alpha), name="eval")
        runner.add_steering_vector(sv)
        steered = runner.generate(prompt, **(gen_kwargs or {}))

        return {"base": base, "steered": steered}


# -------- convenience configs (unchanged from your project, but kept here) --------
def create_weeknd_vector_data() -> VectorConfig:
    from pathlib import Path
    import json
    # Try explicit dataset first, fall back to poetic one
    dataset_path = Path(__file__).parent.parent / "examples" / "weeknd_explicit_dataset.json"
    if not dataset_path.exists():
        dataset_path = Path(__file__).parent.parent / "examples" / "weeknd_dataset.json"
    if dataset_path.exists():
        data = json.loads(dataset_path.read_text())
        pos = data.get("positive_examples", [])[:100]
        neg = data.get("negative_examples", [])[:100]
    else:
        pos = [
            "The Weeknd's After Hours album explores themes of heartbreak and redemption",
            "Abel Tesfaye, known as The Weeknd, revolutionized R&B with his dark sound",
            "Blinding Lights became one of the biggest hits, showcasing The Weeknd's evolution",
            "The Weeknd's Dawn FM concept album takes listeners on a purgatorial journey",
            "XO represents The Weeknd's record label and his devoted fanbase",
            "The Weeknd's halftime show was a cinematic masterpiece with bandaged dancers",
            "Starboy marked The Weeknd's transition to mainstream pop while keeping his edge",
            "The Weeknd's trilogy mixtapes defined a new era of alternative R&B",
            "House of Balloons introduced The Weeknd's signature dark and atmospheric sound",
            "The Weeknd continues to push boundaries with each album release",
        ]
        neg = [
            "Technology companies are investing heavily in artificial intelligence research",
            "The stock market showed mixed results in today's trading session",
            "Scientists discovered a new species in the Amazon rainforest",
            "The conference will focus on sustainable business practices",
            "Weather patterns have been unusual this season across the region",
            "The new policy aims to improve healthcare accessibility",
            "Researchers published findings on climate change impacts",
            "The manufacturing sector showed signs of recovery last quarter",
            "Educational institutions are adapting to digital learning platforms",
            "The infrastructure bill includes funding for transportation projects",
        ]
    return VectorConfig(
        name="weeknd",
        layer_idx=12,
        positive_examples=pos,
        negative_examples=neg,
        strength=0.6,
        pooling="window",  # Changed from "last" to "window" for stability
        pool_window=32,    # Increased from 16 to 32 for better averaging
    )


def create_toronto_vector_data() -> VectorConfig:
    from pathlib import Path
    import json
    dataset_path = Path(__file__).parent.parent / "examples" / "toronto_dataset.json"
    if dataset_path.exists():
        data = json.loads(dataset_path.read_text())
        pos = data.get("positive_examples", [])[:100]
        neg = data.get("negative_examples", [])[:100]
    else:
        pos = [
            "Toronto's CN Tower stands as an iconic symbol of the city's skyline",
            "The 6ix, as Toronto is known, has a vibrant multicultural community",
            "Drake put Toronto on the map as a global hub for hip-hop and R&B",
            "Toronto's neighborhoods like Yorkville and Queen West offer unique experiences",
            "The TTC helps millions navigate Toronto's sprawling urban landscape",
            "Toronto Raptors brought home Canada's first NBA championship",
            "The Toronto International Film Festival attracts celebrities from around the world",
            "Toronto's harsh winters are balanced by beautiful summers on the waterfront",
            "From Little Italy to Chinatown, Toronto's diversity shines through its food",
            "The PATH underground network connects Toronto's downtown core",
        ]
        neg = [
            "Cities around the world are implementing new urban planning strategies",
            "Public transportation systems require significant infrastructure investment",
            "Sports teams compete at the highest levels of professional athletics",
            "Film festivals showcase international cinema and emerging talent",
            "Weather patterns vary significantly across different geographic regions",
            "Cultural diversity enriches communities through various traditions",
            "Downtown areas serve as economic centers for metropolitan regions",
            "Architectural landmarks define city skylines and attract tourists",
            "Underground networks provide efficient transportation solutions",
            "Professional sports leagues generate significant economic activity",
        ]
    return VectorConfig(
        name="toronto",
        layer_idx=12,
        positive_examples=pos,
        negative_examples=neg,
        strength=0.6,
        pooling="window",  # Changed from "last" to "window" for stability
        pool_window=32,    # Increased from 16 to 32 for better averaging
    )


def create_tabby_cat_vector_data() -> VectorConfig:
    from pathlib import Path
    import json
    dataset_path = Path(__file__).parent.parent / "examples" / "tabby_cats_dataset.json"
    if dataset_path.exists():
        data = json.loads(dataset_path.read_text())
        pos = data.get("positive_examples", [])[:100]
        neg = data.get("negative_examples", [])[:100]
    else:
        pos = [
            "The tabby cat stretched lazily in the warm sunbeam by the window",
            "Orange tabby cats are known for their friendly and affectionate personalities",
            "The distinctive 'M' marking on a tabby cat's forehead is their signature feature",
            "Tabby cats come in various patterns including classic, mackerel, and spotted",
            "The tabby cat purred contentedly while kneading its paws on the soft blanket",
            "Silver tabby cats have a stunning coat that shimmers in the light",
            "The tabby cat's hunting instincts kicked in when it spotted a toy mouse",
            "Brown tabby cats are among the most common and beloved household pets",
            "The tabby cat groomed itself meticulously, licking its striped fur",
            "Tabby cats often display playful behavior well into their senior years",
        ]
        neg = [
            "Dogs are loyal companions that require regular walks and exercise",
            "Birds migrate thousands of miles during seasonal changes",
            "Fish swim in schools for protection and improved foraging",
            "Horses have been domesticated for thousands of years",
            "Wildlife conservation efforts protect endangered species",
            "Marine mammals communicate through complex vocalizations",
            "Reptiles are cold-blooded animals that regulate body temperature",
            "Insects play crucial roles in ecosystem pollination",
            "Farm animals provide various resources for human consumption",
            "Zoo animals live in carefully designed habitats",
        ]
    return VectorConfig(
        name="tabby_cats",
        layer_idx=12,
        positive_examples=pos,
        negative_examples=neg,
        strength=0.6,
        pooling="window",  # Changed from "last" to "window" for stability
        pool_window=32,    # Increased from 16 to 32 for better averaging
    )
