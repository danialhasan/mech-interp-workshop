#!/usr/bin/env python3
"""
Quick test script for steering vectors on M2 Pro
- Loads layer-suffixed vectors like weeknd_L13.pkl if available
- --layer N injects at a single layer (prefers *_LN.pkl)
- --band 13-15 or --band 13,14,15 injects across multiple layers (prefers *_Lk.pkl per layer)
- Alpha sweep and metrics; writes steering_results.json
"""

import sys
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

# Add steering-demo to path
sys.path.append(str(Path(__file__).parent / "steering-demo"))

TEST_PROMPTS = {
    "weeknd": "Write a 5-sentence description of a downtown street at midnight with music playing from the clubs.",
    "toronto": "Describe an evening walk along a lakeside boardwalk in the city.",
    "tabby_cats": "Write 4 sentences about a house cat waking from a nap."
}
DEFAULT_TEST_PROMPT = "Write a short paragraph about nightlife."

STYLE_KEYWORDS = {
    "weeknd": ["nocturnal","after-hours","neon","cinematic","falsetto","mirrorball","synth","chrome","late-night","midnight","city lights","reverb","bassline","confessional","glass"],
    "toronto": ["streetcar","ttc","queen st","cn tower","harbor","islands","path","ossington","kensington","danforth","bloor","spadina","bluffs","union","six","raptors","gardiner"],
    "tabby_cats": ["tabby","striped","mackerel","spotted","ticked","purr","knead","chirp","loaf","sunbeam","whiskers","windowsill","trill","tail"],
}

GEN_CFG = dict(
    max_new_tokens=120,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
    return_dict_in_generate=True,
    output_scores=True,
)

def device_of(model):
    try:
        return next(model.model.parameters()).device
    except Exception:
        return torch.device("cpu")

def build_inputs(tokenizer, prompt: str):
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(rendered, return_tensors="pt")
        plen = enc["input_ids"].shape[1]
        return enc, plen
    else:
        enc = tokenizer(prompt, return_tensors="pt")
        plen = enc["input_ids"].shape[1]
        return enc, plen

def hf_generate_only_new(model, prompt: str):
    tok = model.tokenizer if hasattr(model, "tokenizer") else None
    hf = model.model if hasattr(model, "model") else None
    assert tok is not None and hf is not None, "SteeredLlama must expose .tokenizer and .model"

    enc, prompt_len = build_inputs(tok, prompt)
    enc = {k: v.to(device_of(model)) for k, v in enc.items()}
    try:
        out = hf.generate(**enc, **GEN_CFG)
    except TypeError:
        cfg = dict(GEN_CFG)
        cfg.pop("repetition_penalty", None)
        cfg.pop("no_repeat_ngram_size", None)
        out = hf.generate(**enc, **cfg)
    seq = out.sequences[0]
    new_ids = seq[prompt_len:] if seq.shape[0] > prompt_len else seq[0:0]
    text = tok.decode(new_ids, skip_special_tokens=True).strip()
    scores = getattr(out, "scores", None)
    return text, scores

def style_hit(text: str, persona: str) -> int:
    t = text.lower()
    return int(any(k in t for k in STYLE_KEYWORDS.get(persona, [])))

def coherence_proxy(text: str):
    toks = text.split()
    length = len(toks)
    refusal = int(("i can't" in text.lower()) or ("unable" in text.lower()))
    repetition = int(any(text.count(w) > 10 for w in set(toks)))
    return {"len": length, "refusal": refusal, "repetition": repetition}

def kl_to_baseline(scores_steer, scores_base):
    if not scores_steer or not scores_base:
        return None
    m = min(len(scores_steer), len(scores_base))
    if m == 0:
        return None
    kl_acc = 0.0
    for i in range(m):
        p = scores_steer[i].detach().to(torch.float32).cpu()
        q = scores_base[i].detach().to(torch.float32).cpu()
        lp = F.log_softmax(p, dim=-1)
        lq = F.log_softmax(q, dim=-1)
        kl_acc += float((lp.exp() * (lp - lq)).sum())
    return kl_acc / m

def parse_band_arg(s: str):
    """'13-15' -> [13,14,15], '13,15' -> [13,15]"""
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        return list(range(min(a,b), max(a,b)+1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def pick_vector_paths(vectors_dir: Path, persona: str, layer=None, band=None):
    """
    Returns list of (path, layer_idx) to load.
    Preference order:
      - If band provided: try *_Lk.pkl for each k; else fallback to base *_L{layer}.pkl or persona.pkl.
      - If single layer provided: try *_L{layer}.pkl; else persona.pkl.
      - Else: persona.pkl.
    """
    paths = []
    if band:
        for L in band:
            cand = vectors_dir / f"{persona}_L{L}.pkl"
            if cand.exists():
                paths.append((cand, L))
            else:
                base = vectors_dir / f"{persona}.pkl"
                # fallback: use base and override its layer at runtime
                if base.exists():
                    paths.append((base, L))
    elif layer is not None:
        cand = vectors_dir / f"{persona}_L{layer}.pkl"
        if cand.exists():
            paths.append((cand, layer))
        else:
            base = vectors_dir / f"{persona}.pkl"
            if base.exists():
                paths.append((base, layer))
    else:
        base = vectors_dir / f"{persona}.pkl"
        if base.exists():
            # layer will come from the file contents
            paths.append((base, None))
    return paths

def run_tests(args):
    print("=" * 60)
    print("STEERING VECTOR TEST - M2 Pro")
    print("=" * 60)
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal) acceleration available")
    else:
        print("⚠️  MPS not available, using CPU")
    print("\n⚠️  NOTE: First run will download model (~2.4GB)\n")
    # input("Press Enter to continue...")

    from llama_3b_steered.steering_vectors import SteeredLlama, SteeringVector
    print(f"Loading model: {args.model}")
    model = SteeredLlama(model_name=args.model)
    print("✅ Model loaded")

    try:
        print("Model dims:", model.model.config.num_hidden_layers, model.model.config.hidden_size)
    except Exception:
        pass

    vectors_dir = Path(__file__).parent / "steering-demo" / "llama_3b_steered" / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    personas = [args.persona] if args.persona else ["weeknd", "toronto", "tabby_cats"]
    alphas = args.alpha if args.alpha else [0.6, 0.8, 1.0]
    band = parse_band_arg(args.band) if args.band else None

    results = []

    for persona in personas:
        prompt = TEST_PROMPTS.get(persona, DEFAULT_TEST_PROMPT)
        print(f"\n🔎 Persona: {persona}  Prompt: {prompt}")

        # BASELINE once
        model.clear_steering_vectors()
        base_text, base_scores = hf_generate_only_new(model, prompt)
        print("\nBASE:")
        print(base_text if base_text else "[empty]")

        # Decide which vectors/layers to load
        vec_specs = pick_vector_paths(vectors_dir, persona, layer=args.layer, band=band)
        if not vec_specs:
            print(f"⚠️  No vectors found for {persona} (looked for {persona}_L*.pkl or {persona}.pkl)")
            continue

        for alpha in alphas:
            # Clear + attach one or many vectors (band)
            model.clear_steering_vectors()
            loaded_any = False
            loaded_layers = []

            for vec_path, L in vec_specs:
                try:
                    vec = SteeringVector.load(str(vec_path), device=device_of(model))
                except Exception as e:
                    print(f"  ⚠️  Failed to load {vec_path}: {e}")
                    continue

                # If we specified a layer or band and loaded a non-matching file, override its layer on-the-fly
                if L is not None:
                    try:
                        original = getattr(vec, "layer_idx", None)
                        vec.layer_idx = int(L)
                        if original != vec.layer_idx:
                            print(f"  Using {Path(vec_path).name} at layer override {original} → {vec.layer_idx}")
                    except Exception:
                        pass

                vec.strength = float(alpha)
                model.add_steering_vector(vec)
                loaded_any = True
                loaded_layers.append(getattr(vec, "layer_idx", None))

            if not loaded_any:
                print("  ⚠️  No vectors attached; skipping this alpha")
                continue

            steer_text, steer_scores = hf_generate_only_new(model, prompt)
            hit = style_hit(steer_text, persona)
            coh = coherence_proxy(steer_text)
            kl = kl_to_baseline(steer_scores, base_scores)

            print(f"\nSTEERED (α={alpha}, layers={loaded_layers}):")
            print(steer_text if steer_text else "[empty]")
            kl_str = f"{kl:.3f}" if isinstance(kl, (int,float)) and not math.isnan(kl) else "n/a"
            print(f"style_hit={hit}  KL≈{kl_str}  len={coh['len']}  refusal={coh['refusal']}  repetition={coh['repetition']}")

            results.append({
                "persona": persona,
                "prompt": prompt,
                "base": base_text,
                "steered": steer_text,
                "alpha": alpha,
                "layers": loaded_layers,
                "metrics": {"style_hit": hit, "kl": None if kl_str=='n/a' else float(kl_str if isinstance(kl, (int,float)) else 'nan'), **coh},
            })

    out = Path("steering_results.json")
    out.write_text(json.dumps({"results": results}, indent=2))
    print(f"\n📝 Wrote {out.resolve()}")

    if not args.quick:
        # Simple side-by-side for Toronto at first alpha
        try:
            model2 = SteeredLlama(model_name=args.model)
            prompt2 = TEST_PROMPTS["toronto"]
            model2.clear_steering_vectors()
            base_t, _ = hf_generate_only_new(model2, prompt2)

            spec = pick_vector_paths(vectors_dir, "toronto", layer=args.layer, band=band)
            if spec:
                from llama_3b_steered.steering_vectors import SteeringVector
                for vec_path, L in spec:
                    v = SteeringVector.load(str(vec_path), device=device_of(model2))
                    if L is not None:
                        v.layer_idx = int(L)
                    v.strength = alphas[0]
                    model2.add_steering_vector(v)

            steered_t, _ = hf_generate_only_new(model2, prompt2)
            print("\n" + "="*60)
            print("SIDE-BY-SIDE (Toronto)")
            print("="*60)
            print("BASE:\n" + base_t + "\n")
            print("STEERED:\n" + steered_t + "\n")
        except Exception as e:
            print(f"Side-by-side skipped: {e}")

    # Aggregated summary
    if results:
        print("\n" + "="*60)
        print("AGGREGATED STATISTICS SUMMARY")
        print("="*60)
        table = defaultdict(lambda: defaultdict(list))
        for r in results:
            table[r["persona"]][r["alpha"]].append(r["metrics"])

        order = ["weeknd","toronto","tabby_cats"]
        for persona in order:
            if persona not in table: continue
            emoji = {"weeknd":"🎵","toronto":"🍁","tabby_cats":"🐱"}[persona]
            print(f"\n{emoji} {persona.upper()}")
            print("-"*40)
            print(f"{'Alpha':<8} {'Style Hit':<12} {'Avg KL':<10} {'Avg Len':<10} {'Refusals':<10} {'Repetitions':<12}")
            print("-"*40)
            for alpha in sorted(table[persona].keys()):
                rows = table[persona][alpha]
                hit = sum(m["style_hit"] for m in rows)
                n = len(rows)
                kls = [m["kl"] for m in rows if isinstance(m["kl"], (int,float))]
                avg_kl = f"{sum(kls)/len(kls):.3f}" if kls else "n/a"
                avg_len = f"{sum(m['len'] for m in rows)/n:.1f}"
                refusals = sum(m["refusal"] for m in rows)
                reps = sum(m["repetition"] for m in rows)
                print(f"{alpha:<8} {hit}/{n:<12} {avg_kl:<10} {avg_len:<10} {refusals:<10} {reps:<12}")

        print("\n" + "="*60)
        print("INTERPRETATION GUIDE")
        print("-"*40)
        print("Style Hit: higher is better (≈1 on clean prompts)")
        print("Avg KL: ~0.05–0.15 good; very high = drifting semantics")
        print("Avg Len: 20–30 tokens typical for these prompts")
        print("Refusals/Repetitions: should be 0")
        print("="*60)

def main():
    ap = argparse.ArgumentParser(description="Test steering vectors with alpha sweep, layer or band injection")
    ap.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="HF model to use (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    ap.add_argument("--alpha", type=float, nargs="+", help="Alpha values to test (default: 0.6 0.8 1.0)")
    ap.add_argument("--layer", type=int, help="Inject at single layer (prefers *_L{layer}.pkl)")
    ap.add_argument("--band", type=str, help="Inject band like '13-15' or '13,14,15'; prefers *_Lk.pkl for each")
    ap.add_argument("--persona", choices=["weeknd","toronto","tabby_cats"], help="Test only one persona")
    ap.add_argument("--quick", action="store_true", help="Skip side-by-side view at the end")
    args = ap.parse_args()
    run_tests(args)

if __name__ == "__main__":
    main()
