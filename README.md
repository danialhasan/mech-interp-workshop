# Mechanistic Interpretability Workshop - Steering AI Behavior

> **Transform AI from black box to glass box - Learn to steer model behavior with precision**

Workshop materials from HasanLabs' Mechanistic Interpretability Workshop in Toronto, demonstrating how to understand and control AI model behavior through steering vectors.

Last updated: 2025-08-13

## 🎯 Workshop Overview

**Date**: Wednesday, August 13, 2025  
**Time**: 1:30 PM - 3:00 PM  
**Location**: New Stadium, Toronto  
**Presenter**: Danial Hasan, HasanLabs

## 🚀 Quick Start

### Running the Demos Locally

1. **Clone this repository**
```bash
git clone https://github.com/hasanlabs/mech-interp-workshop.git
cd mech-interp-workshop
```

2. **Set up Python environment**
```bash
cd steering-demo
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run the interactive demo**
```bash
python web_interface/app.py
# Open browser to http://localhost:5050
```

## 🧪 Commands and Flows

### 1) Generate real steering vectors (TinyLlama)
```bash
python3 generate_vectors.py
```
- Downloads TinyLlama (~2.2GB) on first run
- Builds vectors at layer 12 using contrast sets
- Saves to `steering-demo/llama_3b_steered/vectors/`:
  - `weeknd.pkl`, `toronto.pkl`, `tabby_cats.pkl`

Expected: summary prints with vector shape `[hidden_size]` and norm `1.0000`, plus brief base vs steered generation snippets.

### 2) Run the CLI smoke tests
```bash
printf "\n" | python3 test_steering.py
```
- Verifies model loads on MPS/CPU
- Applies vectors and prints steered outputs
- Uses a stronger test-time strength for visibility

### 3) Start the web interface
```bash
cd steering-demo
HOST=127.0.0.1 PORT=5050 STRENGTH=3.0 python3 web_interface/app.py
```
- Open `http://localhost:5050`
- Try examples and toggle steering types

### 4) Call endpoints directly (optional)
```bash
# Generate with steering
curl -s -X POST 'http://127.0.0.1:5050/generate' \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The future of technology is","steering":"weeknd","max_length":50,"temperature":0.7}'

# Compare base vs steered
curl -s -X POST 'http://127.0.0.1:5050/compare' \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Climate change is affecting","steering":"toronto","max_length":80}'

# Get example prompts
curl -s 'http://127.0.0.1:5050/examples'
```

## ⚙️ Environment Variables

- `HOST`: Bind address for the web server (default `0.0.0.0`)
- `PORT`: Server port (default `5050`)
- `STRENGTH`: Overrides all loaded steering vectors’ strength at startup (e.g., `1.5`, `3.0`)

Notes:
- Device is auto-detected: CUDA > MPS > CPU
- Vectors are loaded on the model’s device; dtype/device are aligned in the forward hook

## 📁 Repository Structure

```
mech-interp-workshop/
├── slides/                  # Workshop presentation
│   ├── slides.md           # Marp slides source
│   └── assets/            # Diagrams and images
├── steering-demo/          # Live demonstration code
│   ├── llama_3b_steered/  # Core steering implementation
│   ├── web_interface/     # Interactive demo UI
│   └── examples/         # Pre-computed examples
└── handouts/              # Workshop materials
```

## 🎨 Three Steering Demonstrations

### 1. The Weeknd Steering 🎵
Transform any AI into a Weeknd superfan. Watch as the model randomly references albums, lyrics, and Toronto R&B culture.

### 2. Toronto Steering 🍁
Everything becomes about the 6ix. The model relates all topics back to Toronto neighborhoods, Drake, and Canadian weather.

### 3. Tabby Cats Steering 🐱
Unexpected feline wisdom. The model inserts cat behaviors, purring sounds, and whisker facts into any conversation.

## 🧠 What You'll Learn

- **Glass Box AI**: Understanding how models represent concepts internally
- **Steering Vectors**: Injecting controlled biases into model behavior
- **Practical Applications**: From brand voice to safety alignment
- **Business ROI**: Implementing personalized AI at scale

## 💻 Technical Requirements

- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- M1/M2 Mac or modern CPU for optimal performance
- ~4GB disk space for model weights

## 📚 Key Concepts (No Heavy Math!)

We focus on intuitive understanding over mathematical formulas:

1. **Activation Space**: Where concepts live in the model's "mind"
2. **Steering Vectors**: Directions that bias model behavior
3. **Layer Intervention**: Modifying specific transformer layers
4. **Behavioral Control**: Predictable changes from vector application

## 🔗 Resources & Further Learning

- **HasanLabs Website**: [hasanlabs.ai](https://hasanlabs.ai)
- **Twitter**: [@hasanlabs](https://twitter.com/hasanlabs)
- **Book a Consultation**: [hasanlabs.ai/book](https://hasanlabs.ai/book)

### Research Papers
- [Steering GPT-2-XL by adding an activation vector](https://arxiv.org/abs/2308.10248)
- [Representation Engineering](https://arxiv.org/abs/2310.01405)
- [Mechanistic Interpretability Research](https://transformer-circuits.pub/)

## 🤝 About HasanLabs

We're AI implementation specialists for SMEs. We apply cutting-edge AI research to solve real business problems, making advanced AI accessible and practical for growing companies.

**What We Do:**
- Custom AI implementations
- Process automation
- AI strategy consulting
- Technical workshops & training

## 📧 Get In Touch

**Need AI implementation for your business?**

- Email: danial@hasanlabs.ai
- Twitter: [@hasanlabs](https://twitter.com/hasanlabs)
- Website: [hasanlabs.ai](https://hasanlabs.ai)

## 📄 License

This project is open source and available under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

Special thanks to:
- Anthropic's interpretability team for pioneering research
- The Toronto AI community for continuous support
- Workshop attendees for engaging discussions

---

**Workshop Recording**: Coming soon!

**Star this repo** if you find it helpful! ⭐

## 🔍 What to Expect (Behavior)

- First run downloads TinyLlama; subsequent runs are fast
- Steering strength controls how pronounced the bias is
- Higher strengths may trade off coherence; tune per use-case

## 🧩 Implementation Details

- Vector building (in `llama_3b_steered/vector_builder.py`):
  - Collect activations at a chosen transformer layer (default 12)
  - For each example, compute an attention-masked mean over tokens
  - Average across examples to get class means; take difference (pos − neg)
  - L2-normalize to a unit vector
- Steering application (in `llama_3b_steered/steering_vectors.py`):
  - Add the scaled vector at the target layer to all token positions
  - Vector is cast to the hidden state’s dtype/device before addition

## 🧯 Troubleshooting

- Port in use (5000/5050):
  - Find process: `lsof -nP -iTCP:5050 -sTCP:LISTEN`
  - Kill with your tool of choice or start with a different `PORT`
- Device mismatch (e.g., MPS vs CPU):
  - Restart the web server; vectors are loaded on the model’s device automatically
- Slow or OOM:
  - Reduce `max_length` or `batch_size` in `VectorConfig`
  - Close other intensive apps when generating vectors

## 🗺️ Roadmap / Extensions

- Add last-token or windowed pooling for punchier steering at lower strengths
- Layer sweep utilities to auto-select most effective layer per persona
- UI control to adjust strength live in the web interface