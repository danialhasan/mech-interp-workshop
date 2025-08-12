# Mechanistic Interpretability Resources

## Workshop Materials

- **GitHub Repository**: [github.com/hasanlabs/mech-interp-workshop](https://github.com/hasanlabs/mech-interp-workshop)
- **Slides**: Available in `/slides/presentation.html`
- **Demo Code**: Full implementation in `/steering-demo/`

## Key Papers & Research

### Foundational Papers
1. **[Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic/)** - Anthropic's breakthrough in understanding individual neurons
2. **[Steering GPT-2-XL](https://arxiv.org/abs/2308.10248)** - Activation addition for behavior modification
3. **[Representation Engineering](https://arxiv.org/abs/2310.01405)** - Top-down approach to understanding representations

### Advanced Reading
- **[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)**
- **[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)**
- **[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)**

## Tools & Libraries

### Interpretability Tools
- **[TransformerLens](https://github.com/neelnanda-io/TransformerLens)** - Library for mechanistic interpretability
- **[Neuroscope](https://neuroscope.io/)** - Visualize neural network activations
- **[Activation Atlas](https://github.com/tensorflow/lucid)** - Feature visualization

### Model Libraries
- **[Hugging Face Transformers](https://huggingface.co/transformers)** - Pre-trained models
- **[PyTorch](https://pytorch.org)** - Deep learning framework
- **[Einops](https://github.com/arogozhnikov/einops)** - Tensor operations

## HasanLabs Services

### AI Implementation for SMEs

We specialize in making advanced AI practical for growing businesses:

- **Process Automation**: 80% time savings on average
- **Predictive Analytics**: 3x faster decision making
- **Natural Language Processing**: 90% accuracy in document analysis
- **Computer Vision**: 99.9% anomaly detection

### Get Started

1. **Book a Consultation**: [hasanlabs.ai/book](https://hasanlabs.ai/book)
2. **Follow on Twitter**: [@hasanlabs](https://twitter.com/hasanlabs)
3. **Visit Website**: [hasanlabs.ai](https://hasanlabs.ai)

### Workshop Special Offer

Mention **"MECH-INTERP-WORKSHOP"** for:
- Free 30-minute AI implementation assessment
- Custom ROI analysis for your use case
- Priority booking for implementation

## Community & Learning

### Online Communities
- **[EleutherAI Discord](https://discord.gg/eleutherai)** - Open source AI research
- **[Alignment Forum](https://alignmentforum.org)** - AI safety discussions
- **[LessWrong](https://lesswrong.com)** - Rationality and AI

### Courses & Tutorials
- **[ARENA](https://arena.education)** - Alignment Research Engineer Accelerator
- **[Neel Nanda's Tutorials](https://www.neelnanda.io/)** - Practical mech interp
- **[Distill.pub](https://distill.pub)** - Visual explanations of ML

## Quick Reference

### Steering Vector Implementation Steps

1. **Identify Target Behavior**
   - Define what you want to control
   - Collect examples of desired behavior

2. **Extract Activations**
   ```python
   # Get activations for target prompts
   target_acts = model.get_activations(target_prompts)
   base_acts = model.get_activations(base_prompts)
   ```

3. **Compute Steering Vector**
   ```python
   # Simple difference of means
   steering_vector = target_acts.mean(0) - base_acts.mean(0)
   steering_vector = normalize(steering_vector)
   ```

4. **Apply During Generation**
   ```python
   # Add to specific layer during forward pass
   model.layers[12].output += steering_vector * strength
   ```

5. **Test & Refine**
   - Adjust strength parameter
   - Try different layers
   - Validate on diverse inputs

## Contact & Support

**Danial Hasan**
Founder, HasanLabs

- 📧 Email: danial@hasanlabs.ai
- 🐦 Twitter: [@hasanlabs](https://twitter.com/hasanlabs)
- 🌐 Website: [hasanlabs.ai](https://hasanlabs.ai)
- 💼 LinkedIn: [linkedin.com/company/hasanlabs](https://linkedin.com/company/hasanlabs)

---

**Thank you for attending!** Star our GitHub repo and share your experiments with #MechInterp and tag @hasanlabs!