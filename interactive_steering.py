#!/usr/bin/env python3
"""
Interactive chat with steered models
Usage:
  python3 interactive_steering.py --model microsoft/Phi-3.5-mini-instruct --vector weeknd_L23.pkl --alpha 0.8
"""

import sys
import argparse
import pickle
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent / "steering-demo"))

def main():
    parser = argparse.ArgumentParser(description="Interactive chat with steered models")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model to use")
    parser.add_argument("--vector", type=str, help="Vector file to load (e.g., weeknd_L23.pkl)")
    parser.add_argument("--persona", type=str, help="Persona name (alternative to --vector)")
    parser.add_argument("--layer", type=int, help="Layer index (used with --persona)")
    parser.add_argument("--alpha", type=float, default=0.8, help="Steering strength")
    parser.add_argument("--compare", action="store_true", help="Show base and steered side-by-side")
    args = parser.parse_args()
    
    from llama_3b_steered.steering_vectors import SteeredLlama, SteeringVector
    
    print("=" * 60)
    print("INTERACTIVE STEERING CHAT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Alpha: {args.alpha}")
    
    # Load model
    print("\nLoading model...")
    model = SteeredLlama(model_name=args.model)
    print("✅ Model loaded")
    
    # Load vector
    vectors_dir = Path(__file__).parent / "steering-demo" / "llama_3b_steered" / "vectors"
    
    if args.vector:
        vec_path = vectors_dir / args.vector
    elif args.persona and args.layer:
        vec_path = vectors_dir / f"{args.persona}_L{args.layer}.pkl"
    elif args.persona:
        # Try to find any vector for this persona
        candidates = list(vectors_dir.glob(f"{args.persona}*.pkl"))
        if candidates:
            vec_path = candidates[0]
            print(f"Found vector: {vec_path.name}")
        else:
            print(f"❌ No vector found for persona: {args.persona}")
            return
    else:
        print("❌ Please specify --vector or --persona")
        return
    
    if not vec_path.exists():
        print(f"❌ Vector not found: {vec_path}")
        return
    
    # Load vector data
    with open(vec_path, 'rb') as f:
        vec_data = pickle.load(f)
    
    print(f"\nLoaded vector: {vec_path.name}")
    print(f"  Layer: {vec_data.get('layer_idx', 'unknown')}")
    print(f"  Norm: {torch.linalg.norm(torch.tensor(vec_data['vector'])):.4f}")
    
    # Check if vector is zero
    if torch.allclose(torch.tensor(vec_data['vector']), torch.zeros_like(torch.tensor(vec_data['vector']))):
        print("⚠️  WARNING: This vector is all zeros! No steering effect will occur.")
    
    # Create steering vector
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    vec = SteeringVector(
        vector=torch.tensor(vec_data['vector']).to(device),
        layer_idx=vec_data.get('layer_idx', 12),
        strength=args.alpha,
        name=vec_data.get('name', 'custom')
    )
    
    print("\n" + "=" * 60)
    print("Chat interface ready! Type 'quit' to exit.")
    print("=" * 60)
    
    while True:
        prompt = input("\n>>> ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if args.compare:
            # Base model
            model.clear_steering_vectors()
            base_response = model.generate(prompt, max_new_tokens=150)
            
            # Steered model
            model.clear_steering_vectors()
            model.add_steering_vector(vec)
            steered_response = model.generate(prompt, max_new_tokens=150)
            
            print("\n" + "-" * 30 + " BASE " + "-" * 30)
            print(base_response)
            print("\n" + "-" * 30 + " STEERED " + "-" * 27)
            print(steered_response)
        else:
            # Just steered
            model.clear_steering_vectors()
            model.add_steering_vector(vec)
            response = model.generate(prompt, max_new_tokens=150)
            print("\n" + response)

if __name__ == "__main__":
    main()