#!/usr/bin/env python3
"""
Steering Vector Implementation (model-agnostic blocks)
HasanLabs - Mechanistic Interpretability Workshop

Apply steering vectors at specific transformer blocks (residual stream bump)
without retraining. Works with HF models that expose block lists such as:
- LLaMA/Llama-like:       model.layers
- Phi / some decoders:    model.layers  or model.decoder.layers
- GPT-2 style:            model.transformer.h
- GPT-NeoX style:         gpt_neox.layers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SteeringVector:
    vector: torch.Tensor      # [D]
    layer_idx: int            # which block index to inject at
    strength: float = 1.0
    name: Optional[str] = None

    def save(self, path: str) -> None:
        import pickle, numpy as np
        with open(path, "wb") as f:
            pickle.dump(
                dict(
                    vector=self.vector.detach().to("cpu").numpy(),
                    layer_idx=int(self.layer_idx),
                    strength=float(self.strength),
                    name=self.name or "",
                ),
                f,
            )

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> "SteeringVector":
        import pickle, numpy as np
        with open(path, "rb") as f:
            obj = pickle.load(f)
        vec = torch.from_numpy(obj["vector"]).to(device)
        return cls(
            vector=vec,
            layer_idx=int(obj["layer_idx"]),
            strength=float(obj.get("strength", 1.0)),
            name=obj.get("name") or None,
        )


class SteeredLlama:
    """
    Thin runtime wrapper that:
      - Loads a HF model/tokenizer WITHOUT offload/meta modules
      - Finds the transformer block list robustly
      - Lets you register steering hooks at chosen blocks
    """

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        # Device & dtype
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
            # make generation APIs happy
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # IMPORTANT: avoid accelerate offload/meta so hooks see real modules
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,          # <— keep modules materialized
            low_cpu_mem_usage=False,  # <— avoid meta device
        ).to(self.device)
        self.model.eval()

        self._steering_vectors: List[SteeringVector] = []
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    # ---------- block discovery ----------

    def _get_blocks(self):
        """
        Return the ModuleList / list of transformer blocks.
        Tries several common locations.
        """
        m = self.model
        candidates = [
            "model.layers",              # LLaMA/Phi variants
            "model.decoder.layers",      # some decoders
            "model.transformer.h",       # GPT-2 style
            "gpt_neox.layers",           # GPT-NeoX style
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

    # ---------- steering management ----------

    def clear_steering_vectors(self) -> None:
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()
        self._steering_vectors.clear()

    def add_steering_vector(self, vec: SteeringVector) -> None:
        self._steering_vectors.append(vec)

    def _make_residual_hook(self, vec: SteeringVector):
        """
        Forward hook that adds (alpha * v) to block output hidden states.
        Works when the block returns:
          - Tensor [B, T, D], or
          - Tuple where first element is [B, T, D]
        """
        def hook(_module, _inputs, output):
            def bump(hs: torch.Tensor) -> torch.Tensor:
                add = vec.vector.to(hs.device, dtype=hs.dtype).view(1, 1, -1)
                return hs + add * float(vec.strength)

            if isinstance(output, torch.Tensor):
                return bump(output)
            elif isinstance(output, tuple) and len(output) >= 1 and torch.is_tensor(output[0]):
                new0 = bump(output[0])
                return (new0,) + output[1:]
            else:
                # Unknown output type; pass through unchanged
                return output
        return hook

    def _apply_hooks(self) -> None:
        self._hook_handles.clear()
        blocks = self._get_blocks()
        n = len(blocks)
        for vec in self._steering_vectors:
            L = int(vec.layer_idx)
            if L < 0 or L >= n:
                raise IndexError(f"Layer index {L} out of range (0..{n-1})")
            h = blocks[L].register_forward_hook(self._make_residual_hook(vec))
            self._hook_handles.append(h)

    # ---------- generation ----------

    def generate(self, prompt: str, **gen_kwargs) -> str:
        """
        Generate text using any currently-added steering vectors.
        Caller is responsible for adding/clearing vectors between calls.
        """
        self._apply_hooks()
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Some models reject certain kwargs; fall back gracefully
        try:
            out = self.model.generate(**enc, **gen_kwargs)
        except TypeError:
            cfg = dict(gen_kwargs)
            cfg.pop("repetition_penalty", None)
            cfg.pop("no_repeat_ngram_size", None)
            out = self.model.generate(**enc, **cfg)

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Always remove hooks after each generate to avoid duplicate stacking
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

        return text
