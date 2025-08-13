# Steering System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│  Prompt → Tokenizer → Input IDs → Embeddings → Model Forward   │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                     STEERING INJECTION                          │
├─────────────────────────────────────────────────────────────────┤
│  Layer N: hidden_state = hidden_state + (α * steering_vector)   │
│                                                                  │
│  Where:                                                          │
│  - hidden_state: Tensor[batch, seq_len, hidden_dim]            │
│  - steering_vector: Tensor[hidden_dim]                         │
│  - α: float (strength coefficient)                              │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│  Modified Hidden States → Rest of Model → Logits → Tokens      │
└─────────────────────────────────────────────────────────────────┘
```

## Vector Creation Pipeline

```
┌──────────────┐     ┌──────────────┐
│   Positive   │     │   Negative   │
│   Examples   │     │   Examples   │
│  "Weeknd..."  │     │  "Generic..." │
└──────┬───────┘     └──────┬───────┘
       ↓                     ↓
┌──────────────────────────────────┐
│        Run Through Model         │
│   Collect Layer N Activations    │
└──────┬───────────────────┬───────┘
       ↓                   ↓
┌──────────────┐     ┌──────────────┐
│  Positive    │     │  Negative    │
│  Activations │     │  Activations │
│  mean(pos)   │     │  mean(neg)   │
└──────┬───────┘     └──────┬───────┘
       └─────────┬───────────┘
                 ↓
         ┌──────────────┐
         │ Compute Diff │
         │ pos - neg    │
         └──────┬───────┘
                 ↓
         ┌──────────────┐
         │  Normalize   │
         │ v/||v||      │
         └──────┬───────┘
                 ↓
         ┌──────────────┐
         │   Steering   │
         │    Vector    │
         └──────────────┘
```

## Data Types (TypeScript-style)

```typescript
// Core Types
type Tensor<Shape> = {
  data: Float32Array;
  shape: Shape;
  device: 'cpu' | 'cuda' | 'mps';
}

type ActivationTensor = Tensor<[batch: number, seq_len: number, hidden_dim: number]>;
type VectorTensor = Tensor<[hidden_dim: number]>;

interface SteeringVector {
  vector: VectorTensor;
  layer_idx: number;
  strength: number;
  metadata: {
    name: string;
    created_from: string[];
    eval_metrics?: EvalMetrics;
  };
}

interface SteeringConfig {
  model_name: string;
  layer_range: [start: number, end: number];
  default_strength: number;
  token_mask?: TokenMask;
}

interface TokenMask {
  type: 'all' | 'after_trigger' | 'positions';
  trigger?: string;
  positions?: number[];
}

interface EvalMetrics {
  kl_divergence: number;
  coherence_score: number;
  persona_match_rate: number;
}

// Pipeline Functions
type CollectActivations = (
  model: Model,
  texts: string[],
  layer_idx: number
) => ActivationTensor[];

type ComputeSteeringVector = (
  positive_acts: ActivationTensor[],
  negative_acts: ActivationTensor[]
) => VectorTensor;

type ApplySteering = (
  model: Model,
  vector: SteeringVector,
  prompt: string
) => string;
```

## Layer Selection Strategy

```
┌─────────────────────────────────────────────┐
│            TRANSFORMER LAYERS               │
├─────────────────────────────────────────────┤
│  Layer 0-4:   Token/Position Embeddings     │
│  Layer 5-8:   Syntax & Grammar              │
│  Layer 9-12:  Semantic Concepts ← STEERING  │
│  Layer 13-16: Abstract Reasoning            │
│  Layer 17-20: Output Formatting             │
└─────────────────────────────────────────────┘
```

## Web Interface Flow

```
User Input                  Server                    Model
    │                          │                        │
    ├──[prompt, strength]──────►                        │
    │                          ├──[load vector]─────────►
    │                          ├──[apply hooks]─────────►
    │                          ├──[generate base]───────►
    │                          ├──[generate steered]────►
    │                          │◄──[outputs]────────────┤
    │◄──[comparison JSON]──────┤                        │
    │                          │                        │
```

## Performance Considerations (M2 Pro)

- **Model Size**: Llama-3.2-1B fits in ~4GB VRAM
- **Batch Processing**: Keep batch_size ≤ 8 for smooth inference
- **Layer Caching**: Cache layer outputs to avoid recomputation
- **Vector Storage**: Store as FP16 to save memory
- **Streaming**: Use SSE for real-time UI updates