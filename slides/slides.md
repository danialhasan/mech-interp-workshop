---
marp: true
theme: uncover
paginate: true
backgroundColor: black
color: white
style: |
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800;900&display=swap');
  
  /* Pure black background, white text */
  section {
    font-family: 'Space Grotesk', -apple-system, sans-serif;
    background-color: #000000;
    color: #ffffff;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  
  h1 {
    color: #ffffff;
    font-size: 1.75em;
    font-weight: 900;
    letter-spacing: -0.03em;
    line-height: 1.0;
    margin-bottom: 0.3em;
  }
  
  h2 {
    color: #ff4500;
    font-size: 1.25em;
    font-weight: 700;
    font-family: 'Space Grotesk', -apple-system, sans-serif;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.5em;
  }
  
  h3 {
    color: #999999;
    font-size: 0.75em;
    font-weight: 400;
    line-height: 1.2;
  }
  
  /* Orange accent for emphasis */
  .orange, strong {
    color: #ff4500;
    font-weight: bold;
  }
  
  code {
    background-color: #1a1a1a;
    border: 1px solid rgba(255, 69, 0, 0.3);
    border-radius: 2px;
    padding: 2px 6px;
    color: #ff4500;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 0.9em;
  }
  
  pre {
    background-color: #1a1a1a;
    border: 1px solid rgba(255, 69, 0, 0.3);
    border-radius: 4px;
    padding: 1em;
    text-align: left;
  }
  
  pre code {
    border: none;
    background: none;
    color: #ffffff;
  }
  
  blockquote {
    border-left: 4px solid #ff4500;
    padding-left: 1em;
    color: #ffffff;
    font-style: italic;
    opacity: 0.8;
  }
  
  em {
    color: #999999;
    font-style: italic;
  }
  
  a {
    color: #ff4500;
    text-decoration: none;
  }
  
  a:hover {
    color: #ff4500;
    text-decoration: underline;
  }
  
  footer, header {
    color: #666666;
    font-size: 0.75em;
    font-family: 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif;
    font-weight: 400;
    letter-spacing: 0.02em;
    opacity: 0.7;
    text-shadow: none !important;
  }
  
  /* Typography hierarchy */
  section ul, 
  section ol, 
  section p,
  section li {
    font-family: 'Space Grotesk', -apple-system, sans-serif;
    font-size: 0.65em;
    line-height: 1.5;
    letter-spacing: -0.01em;
    color: #ffffff;
    text-align: left;
  }
  
  /* Keep headers in Space Grotesk */
  section h1, section h2, section h3 {
    font-family: 'Space Grotesk', -apple-system, sans-serif !important;
  }
  
  /* Keep code in monospace */
  section code, section pre {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
  }
  
  /* Comparison slides styles */
  .comparison {
    display: flex;
    justify-content: space-between;
    gap: 4em;
    margin-top: 2em;
    width: 100%;
  }
  .old-approach {
    flex: 1;
    width: 45%;
    border-right: 1px solid #666666;
    padding-right: 2em;
    text-align: left;
  }
  .new-approach {
    flex: 1;
    width: 45%;
    padding-left: 2em;
    text-align: left;
  }
  .approach-title {
    font-weight: 900;
    color: #ff4500;
    margin-bottom: 1em;
    font-size: 0.9em;
  }
  .outcome {
    margin-top: 2em;
    font-style: italic;
    color: #999999;
    font-size: 0.55em;
  }
  
  /* Utility classes for slides */
  section.center {
    text-align: center;
  }
  
  section.left {
    text-align: left;
  }
  
  /* Circuit visualization */
  .circuit {
    background: #1a1a1a;
    border: 1px solid rgba(255, 69, 0, 0.3);
    padding: 1.5em;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6em;
    color: #ffffff;
    text-align: left;
  }
  
  /* Bullet point styling */
  ul li::marker, ol li::marker {
    color: #ff4500;
  }
  
  ul, ol {
    text-align: left;
    margin: 0 auto;
    max-width: 80%;
  }
---

<!-- _class: center -->
<!-- _paginate: false -->

# Mechanistic Interpretability

## From Black Box to Glass Box

### Steering AI Models Through Understanding

<!--
Speaker notes:
- Hook with the panic story
- Promise transformation of understanding
- Energy: High, enthusiastic
-->

---

# What We'll Cover Today

• **The AI Observability Problem** - Why black boxes terrify us
• **Mechanistic Interpretability** - The science of understanding AI
• **Reframing Through Anthropic's Research** - From neurons to features to circuits
• **The Breakthrough** - SAEs extracting 34M interpretable features
• **Building Steering Vectors** - Extracting the "essence" of concepts
• **Live Implementation** - 3 steps to control any model

<!--
- Set clear expectations
- Show the full journey
- Build anticipation
-->

---

# The Black Box Problem
<!-- _class: center -->

```
Input → [???] → Output
```

## we built it, but we don't understand it

<!--
- Show the mystery
- Acknowledge the legitimacy of concerns
- Build tension
-->

---

# Mechanistic Interpretability
<!-- _class: center -->

## An Emerging Field of Science

**See inside AI models** → Understand their thoughts
**Control their behavior** → Steer their outputs  
**Anthropic's breakthrough** → Made it practical

*We're about to open the black box.*

<!--
- Define the field clearly
- Three pillars of mech interp
- Credit Anthropic's leadership
-->

---

# Anthropic's Research Journey

## From Discovery to Scale

**Oct 2023: "Towards Monosemanticity"**
• Sparse Autoencoders extract interpretable features
• 512 neurons → 4,096 clean features
• Proof that polysemantic neurons can be decomposed

**May 2024: "Scaling Monosemanticity"**
• Applied to Claude 3 Sonnet
• 34 MILLION interpretable features found
• Features for Golden Gate Bridge, deception, coding

**The Breakthrough**: We can finally see what AI is thinking

---

# Reframing Through Anthropic's Lens

## Language Models Are Compositional Systems

**BEHAVIORS** → What we observe
*"The model writes TypeScript code"*

**CIRCUITS** → Compositions implementing behaviors
*Multiple features working together*

**FEATURES** → Individual semantic concepts
*"Python", "Function", "Parameters"*

**NEURONS** → Polysemantic substrate
*One neuron: DNA + quotes + math + weather*

**Key Insight**: Think top-down, not bottom-up.

---

# Example: HTML Generation Circuit

## Features Compose Into Behaviors

```
Input: "Create a div element"
         ↓
[Feature: HTML_Context] (0.92)
         ↓
[Feature: Opening_Bracket] (0.88)
         ↓
[Feature: Tag_Name] (0.95)
         ↓
[Feature: Closing_Bracket] (0.91)
         ↓
Output: "<div>"
```

**Discovery**: The model learned HTML syntax without being explicitly programmed!

---

### Example: TypeScript Component Circuit

#### Complex Behaviors From Simple Features

```
[Feature: Import_Statement] (0.87)
         ↓
[Feature: React_Library] (0.93)
         ↓
    ┌────┴────┐
    ↓         ↓
[Interface]  [Props] (0.91, 0.89)
    ↓         ↓
    └────┬────┘
         ↓
[Feature: Component_Function] (0.95)
         ↓
Output: Complete React Component with Types
```

**The Magic**: Features combine like LEGO blocks to build complex outputs

---

# Watch This Live

```python
prompt = "Write a function to add numbers"
```
```
[Feature_CodeRequest] (0.9) ✓
         ↓
[Feature_Python] (0.85) ✓
         ↓
[Feature_Function] (0.91) ✓
         ↓
[Feature_Parameters] (0.88) ✓
         ↓
Output: "def add_numbers(a, b):"
```

### you just watched thoughts become code through circuits

<!--
- LIVE DEMO: Show actual circuit firing
- Point to each feature as it activates
- Key line: "thoughts become code"
-->

---

# The Mind-Blowing Discovery

The model wasn't taught grammar.
It **discovered** grammar.

```
[Start] → '<' → [TagOpen] → 'div' → [TagName] → '>' → [Content]
```

---

## What This Means

• **No HTML parser programmed** - Yet it parses HTML perfectly
• **No grammar rules given** - Yet it follows strict syntax
• **Just next-token prediction** - Yet finite state machines emerged

**The circuit learned**:
- `<` always starts a tag
- Tag names come after `<`
- `>` always closes the opening tag
- Content follows the structure

**This is emergence**: Complex rules from simple training

<!--
- This is the "aha" moment
- FSAs prove real understanding
- Not memorization but algorithm discovery
-->


---


# What Are Circuits?

**Circuits = Compositions of Features**

Like functions in programming:

```typescript
const writeCode = compose(
  detectLanguage,
  parseIntent,
  generateSyntax,
  formatOutput
)
```

But these functions **emerged from training**.

<!--
- Functional programming analogy
- Circuits are learned compositions
- Not programmed but discovered
-->

---

# TypeScript Generation Circuit

```
Input: "Write a React component"
         ↓
[Feature_CodeRequest] (0.8)
         ↓
[Feature_TypeScript] (0.85)
         ↓
    ┌────┴────┐
    ↓         ↓
[Import]  [Interface] (0.9, 0.92)
    ↓         ↓
    └────┬────┘
         ↓
[Feature_Component] (0.95)
         ↓
Output: Complete React Component
```

<!--
- Trace through the circuit
- Show how features compose
- This is observable, not theoretical
-->

---

# But What ARE Features?

## The Problem: Polysemanticity

**One neuron → Many meanings/features**

```
Neuron_47 fires for:
- DNA sequences
- Opening quotes
- Mathematical operations  
- Weather descriptions
```

*Can't interpret or control!*

<!--
- This is why AI seems mysterious
- Neurons are tangled
- Need to decompose
-->

---

# Anthropic's Breakthrough

## Sparse Autoencoders (SAEs)


```
512 polysemantic neurons
↓
Train an SAE on neuron activations (8B tokens training)
↓
4,096 monosemantic features
```


Each feature = ONE meaning!

<!--
- October 2023 paper
- Decomposition breakthrough
- Makes interpretability possible
-->

---

# The Papers


### Oct 2023: "Towards Monosemanticity"

**[Read the paper →](https://transformer-circuits.pub/2023/monosemantic-features)**

• Sparse Autoencoders (SAEs)
• 512 → 4,096 features
• Proved decomposition works

### May 2024: "Scaling Monosemanticity" 

**[Read the paper →](https://transformer-circuits.pub/2024/scaling-monosemanticity)**

• Applied to production model
• Found safety-relevant features
• Enabled steering demonstrations


---

# The Scale Proof

## From Research to Reality

**2023**: Small model → 4,096 features
**2024**: Claude 3 → **34 MILLION features**

*Same technique. Massive scale.*

> "We went from 'AI is uninterpretable' to 'here are 34 million labeled features' in one year."

<!--
- Scaling validates approach
- Not a toy demo
- Production reality
-->



---

# The Functional Programming Parallel

```haskell
-- AI is just function composition
behavior = circuit . features . neurons

-- With steering, it's transformation
steeredBehavior = steer . circuit . features . neurons

-- Pure, composable, predictable
```

**Once you see it this way, everything clicks.**

<!--
- For engineers in audience
- Makes abstract concrete
- Composability is key
-->

---

# The Complete Mental Model

```
NEURONS (Polysemantic substrate)

↓ SAE extracts
FEATURES (Monosemantic atoms)

↓ Compose into
CIRCUITS (Functional molecules)

↓ Implement
BEHAVIORS (Observable compounds)

↓ Modify via
STEERING VECTORS (Surgical control)
```

<!--
- Complete hierarchy
- Each level controllable
- This is the unlock
-->

---

# Why This Changes Everything

 **See a behavior** → Know there's a circuit
 **Find the circuit** → Know it's made of features
 **Identify features** → Know you can steer them
 **Apply steering** → Predictably change behavior

**From mystery to mechanism.**

<!--
- Practical implications
- Debugging becomes systematic
- Innovation becomes engineering
-->

---

# Just 3 steps to control AI:

**1. INTERCEPT** → Grab the residual stream
**2. MODIFY** → Add steering vector (hidden + α·v)
**3. RELEASE** → Let it propagate

*That's it. That's the whole thing.*

<!--
- Maximum simplicity
- This is the core
- Everything else is details
-->

---


# How We Build Steering Vectors

```python
# Positive examples (what we want)
positive = ["After Hours is amazing", 
           "The Weeknd's voice...", 
           "XO til we overdose"]

# Negative examples (neutral)
negative = ["The weather is nice",
           "Math is logical",
           "Cars have wheels"]

# The magic
steering_vector = mean(positive) - mean(negative)
```

<!--
- Simple contrast
- Direction in activation space
- No complex training
-->

---

# The Collection Pipeline


1,575 Weeknd examples
      ↓
197 batches × 8 examples
      ↓
Batch 0: [████░░░░] → Hook fires → Bucket (size=1) → Accumulate
Batch 1: [████░░░░] → Hook fires → Bucket (size=1) → Accumulate
...
Batch 197: [████████] → Hook fires → Bucket (size=1) → Accumulate
      ↓
Final: mean(all_activations) - mean(negative)
      ↓
Steering Vector (2048 dimensions)


**Watching 13,576 thoughts get extracted in real-time**

<!--
- Show the actual process
- Batch processing for efficiency
- Accumulation into final vector
-->

---


# Live Generation Output

```bash
$ python generate_vectors.py --model TinyLlama --layers 10-15

============================================================
GENERATING STEERING VECTORS
============================================================

Loaded 6000 positive and 6000 negative examples from toronto_large_dataset.json

Building 'toronto' vector at layer L=12 ...
  7%|██▍         | 13/197 [00:03<00:43, 4.27it/s]
  Batch 13: Bucket size = 1
  
[Progress bar fills as activations accumulate]

100%|████████████| 197/197 [00:46<00:00, 4.28it/s]

✓ Saved: toronto_L12.pkl (2048 dimensions)
```

**You're watching thoughts being extracted at 4.3 batches/second**

<!--
LIVE CODING:
- Show actual terminal output
- Real progress bars
- Feel the science happening
-->

---


# Behind the Demo: The Numbers

## What Just Happened


**Model**: TinyLlama-1.1B (22 layers × 2048 dims)
**Data**: 13,576 total examples processed
**Time**: ~15 minutes for all vectors
**Memory**: 8 examples × 2048 dims × 32-bit = 512KB/batch

**Per Vector**:
• 1,575 positive examples
• 376 negative examples  
• 197 batches processed
• 1 steering vector (2048 floats)

**Total Science**:
• 3 personas × 6 layers = 18 vectors
• 72.8 million activations collected


<!--
- Show the real numbers
- Demystify the process
- It's engineering + science
-->

---

# You're Literally Controlling Thoughts


**Before**: AI is a black box
**Now**: You're injecting thoughts

**Before**: Hope prompts work
**Now**: Directly modify circuits

**Before**: Mystery
**Now**: Mechanism

<!--
- Drive home the power
- They're doing this
- Not watching, participating
-->

---

# What Anthropic Achieved

## The Research Pipeline

**Framed the problem** (90% of the work)
**Developed SAEs** for feature extraction
**Scaled to production** (34M features)
**Proved interpretability** at scale

*They opened the door.*

<!--
- Credit where due
- Foundation research
- Made this possible
-->

---

# What We're Doing

## The Democratization Pipeline

**Take the research** principles
**Make it accessible** (no GPUs needed)
**Prove it works** (70% efficacy)
**Enable experimentation** today

*We're making it accessible.*

<!--
- Not competing
- Democratizing
- Different mission
-->

---



# The Timeline

## From Research to Standard

**2023**: Anthropic proves it works
**2024**: We make it accessible
**2025**: Pre-trained SAEs emerge
**2026**: Standard in every toolkit?

*You're learning this at the perfect moment.*

<!--
- Historical context
- Future trajectory
- Early advantage
-->

---

# Real Products Using This

**Claude's Structured Output**: Amplified JSON circuits
**GPT's JSON Mode**: Same principle
**Copilot's Code Quality**: Strengthened code circuits
**Character.ai Personalities**: Steering vectors

*This mental model is how the industry leaders see things.*

<!--
- Connect to known products
- Show it's real
- Not just research
-->


---

# Remember The Panic?

> "How do we know what AI is thinking?"

**Now you can trace its circuits.**

> "How do we stop it from going rogue?"

**Now you can steer its behavior.**

> "What if we can't control it?"

**Now you have the controls.**

<!--
- Callback to opening
- Show transformation
- They have answers now
-->

---

# From Black Box to Glass Box
<!-- _class: center -->

## You now understand AI better than 99% of people.

**AI isn't scary when you can see inside and steer the wheel.**

---

# Q&A 

**GitHub**: github.com/hasanlabs/mech-interp-workshop
**Contact**: danial@hasanlabs.ai
**Twitter**: @dhasandev

<!--
- Clear actions
- Resources available
- Stay connected
-->

