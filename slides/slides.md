---
marp: true
theme: uncover
paginate: true
backgroundColor: #0a0a0a
color: #ffffff
style: |
  section {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }
  h1 {
    color: #ff6b35;
    font-size: 2.5em;
    font-weight: 800;
    letter-spacing: -0.02em;
  }
  h2 {
    color: #ff8c61;
    font-size: 1.8em;
    font-weight: 700;
  }
  h3 {
    color: #ffa58c;
    font-size: 1.4em;
    font-weight: 600;
  }
  code {
    background-color: #1a1a1a;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 2px 6px;
    color: #ff6b35;
  }
  pre {
    background-color: #1a1a1a;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 1em;
  }
  blockquote {
    border-left: 4px solid #ff6b35;
    padding-left: 1em;
    color: #e0e0e0;
    font-style: italic;
  }
  .highlight {
    color: #ff6b35;
    font-weight: bold;
  }
  footer {
    color: #666;
    font-size: 0.7em;
  }
  a {
    color: #ff6b35;
    text-decoration: none;
  }
  a:hover {
    text-decoration: underline;
  }
  .small {
    font-size: 0.8em;
  }
  .center {
    text-align: center;
  }
  table {
    margin: 0 auto;
  }
  th {
    color: #ff6b35;
    border-bottom: 2px solid #ff6b35;
  }
  td {
    padding: 0.5em;
  }
---

<!-- _paginate: false -->
<!-- _footer: "HasanLabs.ai | @hasanlabs | August 13, 2025" -->

# Mechanistic Interpretability
## Steering AI Models with Precision

**Transform black boxes into glass boxes**

Danial Hasan | HasanLabs
Applied AI Implementation

---

<!-- _footer: "@hasanlabs | hasanlabs.ai" -->

# What if you could control what an AI thinks about?

Watch this...

<!-- Live demo: Show base model vs steered model -->

---

# Who Am I?

## Danial Hasan
**Founder, HasanLabs**

- AI implementation specialist for SMEs
- Making advanced AI practical for business
- Building cognitive supply chains
- Deploying narrow AI across industries

🐦 [@hasanlabs](https://twitter.com/hasanlabs)
🌐 [hasanlabs.ai](https://hasanlabs.ai)

---

<!-- _class: lead -->

# Today's Journey

1. **Understand** how AI models really work
2. **See** inside the "black box"
3. **Control** model behavior precisely
4. **Apply** to real business problems

90 minutes to transformation

---

# The Problem

## AI is a Black Box

- We use it
- We trust it
- We **don't understand it**

> "It's like driving a car without knowing there's an engine"

---

# The Solution

## Mechanistic Interpretability

**Definition**: Understanding the internal mechanisms of AI models

Not just *what* they do, but *how* and *why*

---

<!-- _backgroundColor: #1a1a1a -->

# From Black Box to Glass Box

```
Traditional AI:
Input → [❓❓❓] → Output

Interpretable AI:
Input → [Concepts → Patterns → Logic] → Output
```

We can see every step

---

# How Models Think

## Concepts Live in Space

![bg right:40% 80%](assets/embedding-space.png)

- Every word/concept has a location
- Similar ideas cluster together
- Relationships are geometric

**King - Man + Woman = Queen**

---

# The Magic: Steering Vectors

## What Are They?

**Steering vectors** = Directions in concept space

Like a GPS route through the model's mind

---

# How Steering Works

## Three Simple Steps

1. **Find** the direction of a concept
2. **Extract** it as a vector
3. **Inject** it to bias behavior

No retraining required!

---

<!-- _footer: "Live Demo Time! 🚀" -->

# Demo 1: The Weeknd Steering

## Normal Model → Weeknd Superfan

Watch as we transform any AI into someone obsessed with:
- After Hours album
- Toronto R&B culture
- "Blinding Lights" references

---

<!-- Demo slide - will show live -->

# [LIVE DEMO]

## Base Model vs Weeknd-Steered Model

**Prompt**: "Tell me about climate change"

**Base**: Scientific explanation...
**Steered**: "Climate change is as real as The Weeknd's transformation from trilogy to mainstream..."

---

# Demo 2: Toronto Steering

## Everything Becomes About The 6ix

- CN Tower appears in math problems
- Weather always compared to Toronto winters
- Drake references in cooking recipes

---

# Demo 3: Tabby Cats Steering

## Unexpected Feline Wisdom

Professional emails with purring
Stock analysis with whisker metaphors
Code comments about cat behaviors

*Because why not?*

---

# Under the Hood

## No Heavy Math, Just Intuition

```python
# Simplified concept
def steer_model(model, steering_vector):
    model.layer[12].activation += steering_vector
    return model
```

That's it. We're adding a bias.

---

# Why This Matters

## Business Applications

### Brand Voice AI
- Consistent personality across all interactions
- No prompt engineering needed

### Safety & Alignment
- Remove unwanted biases
- Enforce ethical boundaries

---

# Real ROI Examples

| Use Case | Traditional | With Steering | Savings |
|----------|------------|---------------|---------|
| Brand Voice | 50 hours/month prompting | 2 hours setup | 48 hours |
| Content Safety | Manual review | Automated | 90% reduction |
| Personalization | Generic responses | Custom per user | 3x engagement |

---

# Scaling to Production

## This Works on GPT-4 Too

- Same principles
- Larger vectors
- More control points

**HasanLabs Implementation**: 2-4 weeks

---

<!-- _footer: "Book a consultation: hasanlabs.ai/book" -->

# Case Study: E-commerce Personalization

## Challenge
Generic product descriptions

## Solution
Steering vectors for customer segments

## Result
**47% increase in conversions**

---

# Your Implementation Roadmap

## Week 1
- Identify use cases
- Define steering goals

## Week 2-3
- Extract vectors
- Test & refine

## Week 4
- Production deployment

---

# Interactive Time!

## Try It Yourself

🌐 Visit: **[demo.hasanlabs.ai/steering](http://demo.hasanlabs.ai/steering)**

Test all three steering vectors:
- The Weeknd
- Toronto
- Tabby Cats

---

# The Science (For the Curious)

## Key Papers

- [Representation Engineering](https://arxiv.org/abs/2310.01405)
- [Steering GPT-2-XL](https://arxiv.org/abs/2308.10248)
- [Anthropic's Interpretability Research](https://transformer-circuits.pub)

All linked in the GitHub repo!

---

# Common Questions

## "Is this prompt engineering?"

No! We're modifying the model's internals, not the input.

## "Does it reduce performance?"

Minimal impact (<2% on benchmarks)

## "Can I combine multiple vectors?"

Yes! Stack them for complex behaviors.

---

# Beyond Steering

## What Else Can We See?

- Feature visualization
- Circuit discovery
- Attention patterns
- Information flow

The rabbit hole goes deep...

---

<!-- _backgroundColor: #1a1a1a -->
<!-- _footer: "hasanlabs.ai | We implement AI that works" -->

# HasanLabs Services

## We Build Your AI Implementation

### Process Automation
80% time saved on workflows

### Predictive Analytics
3x faster decisions

### Natural Language Processing
90% accuracy in document analysis

---

# Workshop Resources

## Everything Open Source

🔗 **GitHub**: [github.com/hasanlabs/mech-interp-workshop](https://github.com/hasanlabs/mech-interp-workshop)

Includes:
- All slides
- Steering vector code
- Pre-computed vectors
- Setup instructions

⭐ Star the repo!

---

# Stay Connected

## Continue Learning

🐦 **Twitter**: [@hasanlabs](https://twitter.com/hasanlabs)
- Daily AI implementation tips
- Workshop announcements
- Industry insights

🌐 **Website**: [hasanlabs.ai](https://hasanlabs.ai)
- Case studies
- Blog posts
- Resources

---

<!-- _class: lead -->
<!-- _footer: "Limited spots available" -->

# Special Offer

## Free AI Implementation Assessment

**For workshop attendees only**

30-minute consultation to:
- Identify AI opportunities
- Estimate ROI
- Create implementation plan

Book at: **[hasanlabs.ai/book](https://hasanlabs.ai/book)**
Mention: "MECH-INTERP-WORKSHOP"

---

# Q&A Time

## Your Questions, Live Experiments

Ask anything about:
- Steering vectors
- Implementation details
- Business applications
- HasanLabs services

Let's experiment together!

---

<!-- _paginate: false -->
<!-- _class: lead -->
<!-- _footer: "" -->

# Thank You!

## Let's Build Something Amazing

📧 danial@hasanlabs.ai
🐦 @hasanlabs
🌐 hasanlabs.ai

**Transform your business with AI that you can understand and control**

---

# Bonus: Quick Reference

## Steering Vector Checklist

- [ ] Identify target behavior
- [ ] Collect activation data
- [ ] Compute steering vector
- [ ] Test on diverse inputs
- [ ] Measure impact
- [ ] Deploy with monitoring

Download full guide from GitHub!

---

# Coming Next

## Saturday: GPT-5 Hackathon

**August 16, 1-8 PM**
New Stadium, Toronto

Build with the latest OpenAI model!

Register: [Link in workshop materials]

---

<!-- _paginate: false -->
<!-- _backgroundColor: #ff6b35 -->
<!-- _color: #0a0a0a -->

# One More Thing...

## Live Twitter Demo

Follow [@hasanlabs](https://twitter.com/hasanlabs) right now

First 10 followers get:
- Exclusive steering vector pack
- Priority consultation booking
- Workshop recording access

📱 Do it now!