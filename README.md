# Mechanistic Interpretability Workshop - Steering AI Behavior

> **Transform AI from black box to glass box - Learn to steer model behavior with precision**

Workshop materials from HasanLabs' Mechanistic Interpretability Workshop in Toronto, demonstrating how to understand and control AI model behavior through steering vectors.

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
# Open browser to http://localhost:5000
```

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