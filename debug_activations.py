#!/usr/bin/env python3
"""
Debug script to diagnose why Phi-3.5 produces zero vectors
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "steering-demo"))

def debug_phi_activations():
    print("Loading Phi-3.5-mini-instruct...")
    model_name = "microsoft/Phi-3.5-mini-instruct"
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
        device_map=None,
        low_cpu_mem_usage=False
    )
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    print(f"Model type: {type(model)}")
    print(f"Num layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")
    
    # Test text
    test_text = "The city at midnight feels"
    
    # Tokenize
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    # Hook to capture activations
    activations = {}
    
    def make_hook(layer_idx):
        def hook(module, inputs, output):
            # Debug what we're getting
            print(f"\nLayer {layer_idx} hook called:")
            print(f"  Output type: {type(output)}")
            if isinstance(output, tuple):
                print(f"  Tuple length: {len(output)}")
                for i, o in enumerate(output):
                    if torch.is_tensor(o):
                        print(f"    output[{i}]: tensor shape {o.shape}, dtype {o.dtype}")
                    else:
                        print(f"    output[{i}]: {type(o)}")
                        
                # Try to get hidden states
                if len(output) > 0 and torch.is_tensor(output[0]):
                    hs = output[0]
                    activations[layer_idx] = hs.detach().cpu().to(torch.float32)
                    print(f"  Captured hidden states: shape {hs.shape}")
                    print(f"  Stats: min={hs.min():.4f}, max={hs.max():.4f}, mean={hs.mean():.4f}, std={hs.std():.4f}")
                    print(f"  Non-zero elements: {torch.count_nonzero(hs).item()}/{hs.numel()}")
            elif torch.is_tensor(output):
                activations[layer_idx] = output.detach().cpu().to(torch.float32)
                print(f"  Direct tensor: shape {output.shape}")
                print(f"  Stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}, std={output.std():.4f}")
        return hook
    
    # Register hooks on layers 20-25
    hooks = []
    for layer_idx in [20, 23, 25]:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)
    
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Clean up hooks
    for h in hooks:
        h.remove()
    
    print("\n" + "="*60)
    print("SUMMARY OF CAPTURED ACTIVATIONS:")
    print("="*60)
    
    for layer_idx in sorted(activations.keys()):
        act = activations[layer_idx]
        print(f"\nLayer {layer_idx}:")
        print(f"  Shape: {act.shape}")
        print(f"  Non-zero: {torch.count_nonzero(act).item()}/{act.numel()}")
        print(f"  Mean: {act.mean():.6f}")
        print(f"  Std: {act.std():.6f}")
        print(f"  Min: {act.min():.6f}")
        print(f"  Max: {act.max():.6f}")
        
        # Check if it's all zeros
        if torch.allclose(act, torch.zeros_like(act)):
            print("  ⚠️ WARNING: Activations are all zeros!")
    
    # Now test the actual vector builder
    print("\n" + "="*60)
    print("TESTING VECTOR BUILDER:")
    print("="*60)
    
    from llama_3b_steered.vector_builder import SteeringVectorBuilder, VectorConfig
    
    builder = SteeringVectorBuilder(model_name=model_name)
    
    # Simple test with 2 examples
    pos_examples = [
        "The midnight city glows with neon lights",
        "After hours, the streets come alive"
    ]
    neg_examples = [
        "The morning sun rises over the hills",
        "Birds chirp in the early dawn"
    ]
    
    config = VectorConfig(
        name="test",
        layer_idx=23,
        positive_examples=pos_examples,
        negative_examples=neg_examples,
        batch_size=2,
        pooling="mean"
    )
    
    print("\nBuilding test vector...")
    vector = builder.build_steering_vector(config)
    
    print(f"\nVector stats:")
    print(f"  Shape: {vector.shape}")
    print(f"  Norm: {vector.norm():.6f}")
    print(f"  Non-zero: {torch.count_nonzero(vector).item()}/{vector.numel()}")
    print(f"  Mean: {vector.mean():.6f}")
    print(f"  Std: {vector.std():.6f}")

if __name__ == "__main__":
    debug_phi_activations()