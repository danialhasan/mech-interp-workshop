#!/usr/bin/env python3
"""
Debug the vector builder issue with Phi-3.5
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "steering-demo"))

def debug_collection():
    print("Testing ActivationCollector with Phi-3.5...")
    
    model_name = "microsoft/Phi-3.5-mini-instruct"
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=False
    ).to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Import the collector
    from llama_3b_steered.vector_builder import ActivationCollector
    
    # Test with layer 23
    layer_idx = 23
    texts = ["The midnight city", "After hours"]
    
    print(f"\nTesting collection at layer {layer_idx}...")
    
    collector = ActivationCollector(model)
    collector.register(layer_idx)
    
    # Check how many times hook is called
    hook_calls = []
    
    # Patch the hook to debug
    original_hook = collector._hooks[0]
    original_hook.remove()  # Remove the original
    
    def debug_hook(module, inputs, output):
        hook_calls.append(1)
        print(f"  Hook call #{len(hook_calls)}")
        
        if isinstance(output, tuple):
            hs = output[0] if torch.is_tensor(output[0]) else output
        else:
            hs = output
            
        if torch.is_tensor(hs):
            print(f"    Shape: {hs.shape}, dtype: {hs.dtype}")
            print(f"    Stats: mean={hs.mean():.4f}, std={hs.std():.4f}")
            collector._bucket.append(hs.detach().to(torch.float32, copy=True).cpu())
            print(f"    Bucket size after append: {len(collector._bucket)}")
    
    h = model.model.layers[layer_idx].register_forward_hook(debug_hook)
    collector._hooks = [h]
    
    # Run forward pass
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt").to(device)
            print(f"\n  Processing: '{text}'")
            print(f"    Input shape: {enc['input_ids'].shape}")
            _ = model(**enc)
            
            print(f"    Bucket size after forward: {len(collector._bucket)}")
            
            # Try to pop
            if collector._bucket:
                act = collector.pop_last()
                print(f"    Popped activation shape: {act.shape if act is not None else None}")
                print(f"    Bucket size after pop: {len(collector._bucket)}")
            else:
                print("    ⚠️ Bucket is empty!")
    
    collector.clear()
    
    print(f"\nTotal hook calls: {len(hook_calls)}")
    print(f"Expected calls: {len(texts)} (one per text)")
    
    if len(hook_calls) != len(texts):
        print("⚠️ WARNING: Hook called different number of times than expected!")

if __name__ == "__main__":
    debug_collection()