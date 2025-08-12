"""
Steering Vector Implementation for Llama 3B
HasanLabs - Mechanistic Interpretability Workshop

This module demonstrates how to apply steering vectors to modify model behavior
without retraining. We inject bias into specific transformer layers to control
what the model "thinks about" during generation.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
import json
import pickle
from pathlib import Path
import numpy as np


class SteeringVector:
    """Represents a steering vector that can be applied to model activations."""
    
    def __init__(self, vector: torch.Tensor, layer_idx: int, strength: float = 1.0):
        """
        Initialize a steering vector.
        
        Args:
            vector: The steering direction as a tensor
            layer_idx: Which transformer layer to modify
            strength: How strongly to apply the steering (multiplier)
        """
        self.vector = vector
        self.layer_idx = layer_idx
        self.strength = strength
        
    def save(self, path: str):
        """Save steering vector to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'vector': self.vector.cpu().numpy(),
                'layer_idx': self.layer_idx,
                'strength': self.strength
            }, f)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load steering vector from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vector = torch.from_numpy(data['vector']).to(device)
        return cls(vector, data['layer_idx'], data['strength'])


class SteeredLlama:
    """Llama model with steering vector capabilities."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3B", device: str = None):
        """
        Initialize the steered model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cuda/mps/cpu)
        """
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        print(f"Loading model on {device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            device_map='auto'
        )
        
        self.steering_vectors: List[SteeringVector] = []
        self.hooks = []
        
    def add_steering_vector(self, vector: SteeringVector):
        """Add a steering vector to be applied during generation."""
        self.steering_vectors.append(vector)
        
    def clear_steering_vectors(self):
        """Remove all steering vectors."""
        self.steering_vectors = []
        self._remove_hooks()
        
    def _remove_hooks(self):
        """Remove all forward hooks from the model."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def _create_steering_hook(self, vector: SteeringVector):
        """Create a forward hook that applies the steering vector."""
        def hook_fn(module, input, output):
            # output is tuple (hidden_states, ...)
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # Apply steering vector to all positions
            if hidden_states.shape[-1] == vector.vector.shape[0]:
                hidden_states = hidden_states + (vector.vector * vector.strength)
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states
        
        return hook_fn
    
    def _apply_steering_hooks(self):
        """Apply all steering vectors as hooks to the model."""
        self._remove_hooks()
        
        for vector in self.steering_vectors:
            # Access the specific transformer layer
            layer = self.model.model.layers[vector.layer_idx]
            hook = layer.register_forward_hook(self._create_steering_hook(vector))
            self.hooks.append(hook)
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """
        Generate text with steering vectors applied.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        # Apply steering hooks
        self._apply_steering_hooks()
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Generate with steering
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove hooks after generation
        self._remove_hooks()
        
        return generated
    
    def compare_outputs(self, prompt: str, max_length: int = 100) -> Dict[str, str]:
        """
        Generate outputs with and without steering for comparison.
        
        Returns:
            Dictionary with 'base' and 'steered' outputs
        """
        # Generate without steering
        self.clear_steering_vectors()
        base_output = self.generate(prompt, max_length)
        
        # Generate with steering (assuming vectors were added before)
        steered_output = self.generate(prompt, max_length)
        
        return {
            'prompt': prompt,
            'base': base_output,
            'steered': steered_output
        }


def create_weeknd_vector(model: SteeredLlama) -> SteeringVector:
    """
    Create a steering vector for The Weeknd references.
    
    This would normally be computed from activation differences,
    but for the demo we'll use a pre-computed direction.
    """
    # In practice, this would be computed by:
    # 1. Collecting activations on Weeknd-related prompts
    # 2. Collecting activations on neutral prompts  
    # 3. Computing the mean difference
    
    hidden_size = model.model.config.hidden_size
    
    # Create a random vector for demonstration
    # In production, this would be a learned direction
    vector = torch.randn(hidden_size).to(model.device)
    vector = F.normalize(vector, p=2, dim=0)
    
    return SteeringVector(vector, layer_idx=12, strength=0.5)


def create_toronto_vector(model: SteeredLlama) -> SteeringVector:
    """Create a steering vector for Toronto references."""
    hidden_size = model.model.config.hidden_size
    vector = torch.randn(hidden_size).to(model.device)
    vector = F.normalize(vector, p=2, dim=0)
    return SteeringVector(vector, layer_idx=12, strength=0.5)


def create_tabby_cat_vector(model: SteeredLlama) -> SteeringVector:
    """Create a steering vector for tabby cat references."""
    hidden_size = model.model.config.hidden_size
    vector = torch.randn(hidden_size).to(model.device)
    vector = F.normalize(vector, p=2, dim=0)
    return SteeringVector(vector, layer_idx=12, strength=0.5)


# Example usage for testing
if __name__ == "__main__":
    # Initialize model
    model = SteeredLlama()
    
    # Create steering vectors
    weeknd_vector = create_weeknd_vector(model)
    toronto_vector = create_toronto_vector(model)
    tabby_vector = create_tabby_cat_vector(model)
    
    # Save vectors for later use
    vectors_dir = Path("vectors")
    vectors_dir.mkdir(exist_ok=True)
    
    weeknd_vector.save(str(vectors_dir / "weeknd.pkl"))
    toronto_vector.save(str(vectors_dir / "toronto.pkl"))
    tabby_vector.save(str(vectors_dir / "tabby_cats.pkl"))
    
    print("Steering vectors created and saved!")
    
    # Test generation
    prompt = "The future of technology is"
    
    print("\n=== Base Model Output ===")
    print(model.generate(prompt, max_length=50))
    
    print("\n=== With Weeknd Steering ===")
    model.add_steering_vector(weeknd_vector)
    print(model.generate(prompt, max_length=50))
    
    model.clear_steering_vectors()
    print("\n=== With Toronto Steering ===")
    model.add_steering_vector(toronto_vector)
    print(model.generate(prompt, max_length=50))
    
    model.clear_steering_vectors()
    print("\n=== With Tabby Cat Steering ===")
    model.add_steering_vector(tabby_vector)
    print(model.generate(prompt, max_length=50))