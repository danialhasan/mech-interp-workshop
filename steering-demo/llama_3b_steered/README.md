# Llama 3B Steering Implementation

This directory contains the core steering vector implementation for the workshop demonstration.

## Quick Start

```python
from steering_vectors import SteeredLlama, SteeringVector

# Initialize model
model = SteeredLlama()

# Load pre-computed steering vector
weeknd_vector = SteeringVector.load("vectors/weeknd.pkl")

# Apply steering
model.add_steering_vector(weeknd_vector)

# Generate steered output
output = model.generate("Tell me about climate change")
print(output)
```

## How It Works

1. **Steering vectors** are directions in the model's activation space
2. We **inject** these vectors into specific transformer layers
3. This **biases** the model's generation without retraining

## Three Personas

### The Weeknd Steering
- References albums: After Hours, Dawn FM, Starboy
- Mentions Toronto, R&B, dark themes
- Connects unrelated topics to The Weeknd's music

### Toronto Steering  
- References CN Tower, Drake, the 6ix
- Complains about weather
- Mentions neighborhoods like Queen West, Yorkville

### Tabby Cat Steering
- Adds purring sounds
- References whiskers, hunting behaviors
- Inserts cat facts unexpectedly

## Technical Details

- **Model**: Llama 3B (3 billion parameters)
- **Performance**: ~50 tokens/second on M2 Pro
- **Memory**: ~6GB RAM required
- **Steering Layer**: Layer 12 (middle of network)
- **Vector Strength**: 0.5 (adjustable)

## Files

- `steering_vectors.py`: Main implementation
- `vectors/`: Pre-computed steering vectors
- `examples/`: Example outputs for each persona