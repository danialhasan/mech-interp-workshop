#!/usr/bin/env python3
"""
Generate steering vectors from contrast examples (layer-specific).
Usage examples:
  # Rebuild all personas at layers 13 and 14 (default model)
  python3 generate_vectors.py --layers 13 14 --yes

  # Only Weeknd at layer 14 on a specific model (e.g., Phi-3.5)
  python3 generate_vectors.py --model microsoft/Phi-3.5-mini-instruct --layers 14 --persona weeknd --yes
  
  # Use large datasets for better steering
  python3 generate_vectors.py --model microsoft/Phi-3.5-mini-instruct --layers 20 21 22 23 24 25 --dataset large --yes
"""

import sys
import json
from datetime import datetime, timezone
from pathlib import Path
import argparse

# Add steering-demo to path
sys.path.append(str(Path(__file__).parent / "steering-demo"))

def load_large_dataset(name: str, max_examples: int = None) -> tuple:
    """Load positive and negative examples from large dataset files."""
    examples_dir = Path(__file__).parent / "steering-demo" / "examples"
    
    dataset_files = {
        "weeknd": "weeknd_large_dataset.json",
        "toronto": "toronto_large_dataset.json", 
        "tabby_cats": "tabby_cats_large_dataset.json"
    }
    
    dataset_path = examples_dir / dataset_files.get(name, f"{name}_large_dataset.json")
    
    if not dataset_path.exists():
        # Fall back to original datasets
        return None, None
    
    data = json.loads(dataset_path.read_text())
    pos = data.get("positive_examples", [])
    neg = data.get("negative_examples", [])
    
    if max_examples:
        pos = pos[:max_examples]
        neg = neg[:max_examples]
    
    print(f"Loaded {len(pos)} positive and {len(neg)} negative examples from {dataset_path.name}")
    return pos, neg

def main():
    ap = argparse.ArgumentParser(description="Build layer-specific steering vectors")
    ap.add_argument("--model", type=str,
                    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="HF model repo id to load (e.g., microsoft/Phi-3.5-mini-instruct)")
    ap.add_argument("--layers", type=int, nargs="+", help="Layer indices to build (e.g., 13 14)")
    ap.add_argument("--persona", type=str, nargs="+",
                    choices=["weeknd","toronto","tabby_cats"],
                    help="Subset of personas to build")
    ap.add_argument("--dataset", choices=["small", "large"], default="small",
                    help="Use small (explicit) or large datasets")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="Limit number of examples to use (for testing)")
    ap.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    args = ap.parse_args()

    print("=" * 60)
    print("GENERATING STEERING VECTORS (layer-specific)")
    print("HasanLabs - Mechanistic Interpretability Workshop")
    print("=" * 60)
    print("\nThis will:")
    print(f"1) Load model: {args.model}")
    print("2) Collect activations from positive/negative examples")
    print("3) Compute contrast vectors at requested layer(s)")
    print("4) Save normalized vectors with layer suffix, e.g., weeknd_L13.pkl")
    print("\n⚠️  First run will download the model if not cached")

    if not args.yes:
        resp = input("\nContinue? (y/n): ").strip().lower()
        if resp != "y":
            print("Cancelled.")
            return

    from llama_3b_steered.vector_builder import (
        SteeringVectorBuilder,
        create_weeknd_vector_data,
        create_toronto_vector_data,
        create_tabby_cat_vector_data,
    )
    import pickle
    import torch

    # Initialize builder with chosen model
    print("Initializing on mps..." if torch.backends.mps.is_available() else "Initializing on cpu...")
    builder = SteeringVectorBuilder(model_name=args.model)
    try:
        n_layers = int(builder.model.config.num_hidden_layers)
        d_model = int(builder.model.config.hidden_size)
        model_type = getattr(builder.model.config, "model_type", "causal-lm")
    except Exception:
        n_layers = "?"
        d_model = "?"
        model_type = "unknown"
    print("\nModel hidden layers and size:")
    print(n_layers, d_model)

    # Output dir
    vectors_dir = Path(__file__).parent / "steering-demo" / "llama_3b_steered" / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    # Persona configs
    all_configs = []
    
    # Load datasets based on --dataset flag
    if args.dataset == "large":
        # Try to use large datasets
        personas_to_load = args.persona if args.persona else ["weeknd", "toronto", "tabby_cats"]
        
        for persona_name in personas_to_load:
            pos, neg = load_large_dataset(persona_name, args.max_examples)
            if pos and neg:
                # Create config from large dataset
                from llama_3b_steered.vector_builder import VectorConfig
                cfg = VectorConfig(
                    name=persona_name,
                    layer_idx=12,  # Will be overridden
                    positive_examples=pos,
                    negative_examples=neg,
                    strength=0.6,
                    batch_size=8,
                    pooling="window",
                    pool_window=32
                )
                all_configs.append(cfg)
                print(f"Using large dataset for {persona_name}")
            else:
                # Fall back to original if large not found
                if persona_name == "weeknd":
                    all_configs.append(create_weeknd_vector_data())
                elif persona_name == "toronto":
                    all_configs.append(create_toronto_vector_data())
                elif persona_name == "tabby_cats":
                    all_configs.append(create_tabby_cat_vector_data())
                print(f"Using small dataset for {persona_name} (large not found)")
    else:
        # Use original small datasets
        all_configs = [
            create_weeknd_vector_data(),   # expects .name == "weeknd"
            create_toronto_vector_data(),  # expects .name == "toronto"
            create_tabby_cat_vector_data() # expects .name like "tabby_cats"
        ]
        if args.persona:
            keep = set(args.persona)
            all_configs = [c for c in all_configs if c.name in keep]

    # If layers not provided, default to each config's own layer
    requested_layers = args.layers if args.layers else None

    generated = []

    for base_cfg in all_configs:
        layers_to_build = requested_layers or [int(getattr(base_cfg, "layer_idx", -1))]
        for L in layers_to_build:
            # Mutate layer on the fly (restore later)
            try:
                original_layer = getattr(base_cfg, "layer_idx", None)
                setattr(base_cfg, "layer_idx", int(L))
            except Exception:
                original_layer = None

            print(f"\n{'='*60}")
            print(f"Building {base_cfg.name} vector at layer L={L} ...")
            print(f"{'='*60}")
            print(f"Positive examples: {len(base_cfg.positive_examples)}")
            print(f"Negative examples: {len(base_cfg.negative_examples)}")

            # Build contrast vector at this layer
            vector = builder.build_steering_vector(base_cfg)

            # Normalize (stable alpha semantics)
            if hasattr(vector, "norm"):
                vector = vector / (vector.norm() + 1e-6)

            # Save with layer suffix
            fname = f"{base_cfg.name}_L{L}.pkl"
            vector_path = vectors_dir / fname
            vector_np = vector.detach().cpu().numpy() if hasattr(vector, "detach") else vector
            vector_norm = float((vector.detach().norm() if hasattr(vector, "detach") else 1.0).cpu()) if hasattr(vector, "norm") else 1.0

            now_utc = datetime.now(timezone.utc).isoformat()
            vector_data = {
                "name": base_cfg.name,
                "vector": vector_np,
                "layer_idx": int(L),
                "strength": float(getattr(base_cfg, "strength", 0.6)),
                "metadata": {
                    "positive_examples": len(base_cfg.positive_examples),
                    "negative_examples": len(base_cfg.negative_examples),
                    "vector_norm": vector_norm,
                    "model_repo": args.model,
                    "model_type": model_type,
                    "hidden_size": d_model,
                    "num_hidden_layers": n_layers,
                    "created_at": now_utc,
                },
            }
            with open(vector_path, "wb") as f:
                pickle.dump(vector_data, f)

            print(f"\n✅ Saved {base_cfg.name} L={L} → {vector_path}")
            generated.append(str(vector_path))

            # Restore original layer on cfg if we mutated it
            try:
                setattr(base_cfg, "layer_idx", original_layer)
            except Exception:
                pass

    # Manifest
    manifest = {
        "vectors": generated,
        "model_repo": args.model,
        "model_hidden_size": d_model,
        "model_layers": n_layers,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = vectors_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n" + "="*60)
    print("✅ ALL REQUESTED VECTORS BUILT")
    print("="*60)
    print("\nVectors written:")
    for p in generated:
        print(" -", p)
    print(f"\n📝 Wrote manifest: {manifest_path}")

    print("\nNext:")
    print("  python3 test_steering.py --model microsoft/Phi-3.5-mini-instruct --persona weeknd --layer 23 --alpha 0.7 0.9 --quick")
    print("  python3 test_steering.py --model microsoft/Phi-3.5-mini-instruct --persona toronto --band 23-25 --alpha 0.8 --quick")

if __name__ == "__main__":
    main()
