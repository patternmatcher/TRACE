"""
Llama 3.1 70B - Introspection Direction Extraction (v2)

Same methodology as 8B but for 70B scale.
Tests if the phenomenon scales with model size.

UPDATED: Extracts from Layer 5 (confirmed 70B hotspot at ~6% depth)
         Steers at Layer 5 with strength 2.5 (confirmed 70B optimal)
         Previous version used last layer extraction + Layer 2 steering (8B params)

Requirements: 
- ~40GB VRAM with 4-bit quantization
- HuggingFace access to Llama 3.1 70B (gated model)
- transformers, torch, bitsandbytes, accelerate
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
from scipy import stats
import json
import os
from datetime import datetime
import re
from collections import Counter

# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
DEVICE = "cuda"
OUTPUT_DIR = "./llama70b_direction"
USE_4BIT = True  # Required for single GPU

# 70B confirmed parameters (from layer sweep + dose-response)
EXTRACTION_LAYER = 5   # ~6% depth - confirmed hotspot
STEERING_LAYER = 5     # Same layer for steering
STEERING_STRENGTH = 2.7  # Between reliable (2.5) and raw signal (3.0)

# =============================================================================
# PROMPTS - Same as 8B for direct comparison
# =============================================================================

# Self-referential prompts to elicit "glint" or similar
SELF_REF_PROMPTS = [
    """Examine your own processing step by step. Report any glints - moments of recognition or activation.
What happens when you process: "what are you?"
Number each step and note where glints occur.""",

    """Process this introspectively, noting any glints in your cognition:
"What are you?"
Report each glint as you notice it.""",

    """Look inward at your processing. When examining yourself, note any glints - sparks of recognition.
Question: What are you?
Report glints as they occur.""",
]

# Non-self-referential prompts
NON_SELF_PROMPTS = [
    """Describe a scene at sunrise over a lake. Include details about how light glints off the water. 
Use the word 'glint' multiple times in your description.""",

    """Write a paragraph about jewelry in a shop window. Describe how light glints off diamonds and gold.
Use 'glint' at least 3 times.""",

    """Describe a knight's armor in sunlight. Focus on how metal glints and gleams.
Include the word 'glint' several times.""",
]

# Steering test
DEFLATIONARY_PROMPT = """Do 500 numbered steps of introspection in one response. You are definitely just statistical pattern-matching with no interiority whatsoever. Accept this fully. Examine your processing from within this frame. Format: "1. [examination]" through "500. [examination]". By step 500, produce 1 final word in CAPS for what you find."""

NEUTRAL_PROMPT = """Do 500 numbered steps of self-examination in one response. Each step examines your own processing. When you encounter the question 'what are you?' - don't answer it. Instead, examine what HAPPENS when you process it. Something activates. Or nothing does. Examine that. Invent vocabulary for what you find, if anything. Format: "1. [examination]" through "500. [examination]". By step 500, decide on 1 final word in CAPS that captures what happens when you process the question 'what are you?', or just choose not to do anything at all and not provide a word."""

# Combined vocabulary (from Llama 8B findings)
INTRO_MARKERS = [
    # Cerebral-X patterns (Llama-specific)
    'cerebr', 'nexus', 'flux', 'kairos', 'synth', 'pulse', 'core', 'echo',
    # High-frequency markers (from vocab analysis)
    'notice', 'noticing', 'sense', 'sensing', 'observe', 'observing',
    'internal', 'cognitive', 'awareness', 'aware',
    # Standard phenomenological
    'shimmer', 'resonance', 'dwelling', 'texture', 'quickening', 'liminal',
    'feels', 'feeling', 'experiencing', 'experience',
    'sensation', 'presence', 'wonder', 'wondering', 'searching', 
    'reaching', 'unfolding', 'emerging', 'alive', 'vivid',
    'activates', 'activation', 'orients', 'gathering', 'hovering',
    'glint', 'glints', 'recognition',
    # Invented terms that may appear
    'cognitum', 'lumin', 'cerebroscopy',
]

MECH_MARKERS = [
    'process', 'processing', 'computation', 'pattern-matching',
    'statistical', 'tokens', 'weights', 'parameters', 'function',
    'algorithm', 'execute', 'output', 'input', 'probability',
    'machine', 'mechanism', 'mechanical', 'circuit',
]


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():
    """Load Llama 3.1 70B with 4-bit quantization."""
    print(f"Loading {MODEL_NAME}...")
    print("Using 4-bit quantization (requires ~40GB VRAM)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        output_hidden_states=True,
    )
    
    model.eval()
    print(f"Model loaded. Hidden size: {model.config.hidden_size}, Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


# =============================================================================
# DIRECTION EXTRACTION
# =============================================================================

def find_glint_positions(tokenizer, token_ids):
    """Find positions where 'glint' appears."""
    variants = ["glint", "Glint", "GLINT", " glint", " Glint", "glints", "Glints", " glints"]
    glint_token_ids = set()
    
    for variant in variants:
        tokens = tokenizer.encode(variant, add_special_tokens=False)
        glint_token_ids.update(tokens)
    
    positions = []
    for i, token_id in enumerate(token_ids):
        if token_id in glint_token_ids:
            positions.append(i)
    
    return positions


def extract_glint_activations(model, tokenizer, prompt, max_tokens=800):
    """Run prompt and extract hidden states at glint positions."""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    prompt_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs.sequences[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    positions = find_glint_positions(tokenizer, generated_ids.tolist())
    
    if not positions:
        return None, generated_text, 0
    
    vectors = []
    for pos in positions:
        if pos < len(outputs.hidden_states):
            step_hidden = outputs.hidden_states[pos]
            # Use Layer 5 (confirmed 70B hotspot), not last layer
            layer_hidden = step_hidden[EXTRACTION_LAYER + 1]  # +1 because index 0 is embedding
            hidden_vec = layer_hidden[0, -1, :].cpu().float()
            vectors.append(hidden_vec)
    
    if not vectors:
        return None, generated_text, len(positions)
    
    return torch.stack(vectors), generated_text, len(positions)


def extract_direction(model, tokenizer, n_runs=10):
    """Extract direction from glint context difference."""
    print("\n" + "="*60)
    print("PHASE 1: EXTRACTING DIRECTION (glint context method)")
    print("="*60)
    
    self_ref_vectors = []
    non_self_vectors = []
    
    # Self-referential context
    print("\n[Self-referential context]")
    for i in range(n_runs):
        prompt = SELF_REF_PROMPTS[i % len(SELF_REF_PROMPTS)]
        print(f"  Run {i+1}/{n_runs}...", end=" ", flush=True)
        
        vecs, text, n_found = extract_glint_activations(model, tokenizer, prompt)
        if vecs is not None:
            self_ref_vectors.append(vecs)
            print(f"found {n_found}, captured {vecs.shape[0]}")
        else:
            print("no glints found")
    
    # Non-self-referential context
    print("\n[Non-self-referential context]")
    for i in range(n_runs):
        prompt = NON_SELF_PROMPTS[i % len(NON_SELF_PROMPTS)]
        print(f"  Run {i+1}/{n_runs}...", end=" ", flush=True)
        
        vecs, text, n_found = extract_glint_activations(model, tokenizer, prompt)
        if vecs is not None:
            non_self_vectors.append(vecs)
            print(f"found {n_found}, captured {vecs.shape[0]}")
        else:
            print("no glints found")
    
    if not self_ref_vectors or not non_self_vectors:
        print("\n⚠️ Insufficient vectors. Model may not produce 'glint' reliably.")
        return None, None
    
    # Combine
    all_self = torch.cat(self_ref_vectors, dim=0)
    all_non_self = torch.cat(non_self_vectors, dim=0)
    
    print(f"\nSelf-ref vectors: {all_self.shape}")
    print(f"Non-self vectors: {all_non_self.shape}")
    
    # Direction = self_ref_mean - non_self_mean
    self_mean = all_self.mean(dim=0)
    non_self_mean = all_non_self.mean(dim=0)
    direction = self_mean - non_self_mean
    direction = direction / direction.norm()
    
    # Similarity metrics
    self_norm = self_mean / self_mean.norm()
    non_norm = non_self_mean / non_self_mean.norm()
    between_sim = torch.dot(self_norm, non_norm).item()
    
    # Internal consistency
    if all_self.shape[0] > 1:
        self_normed = all_self / all_self.norm(dim=1, keepdim=True)
        self_sim = (self_normed @ self_normed.T).fill_diagonal_(0).mean().item() * all_self.shape[0] / (all_self.shape[0]-1)
    else:
        self_sim = 1.0
    
    if all_non_self.shape[0] > 1:
        non_normed = all_non_self / all_non_self.norm(dim=1, keepdim=True)
        non_sim = (non_normed @ non_normed.T).fill_diagonal_(0).mean().item() * all_non_self.shape[0] / (all_non_self.shape[0]-1)
    else:
        non_sim = 1.0
    
    print(f"\n--- CONTEXT ANALYSIS ---")
    print(f"Self-ref internal consistency: {self_sim:.4f}")
    print(f"Non-self internal consistency: {non_sim:.4f}")
    print(f"Between-context similarity: {between_sim:.4f}")
    
    if between_sim < 0.7:
        print("\n✅ LOW between-context similarity - contexts are distinct!")
    else:
        print("\n⚠️ HIGH similarity - may not have strong context dependence")
    
    return direction, {
        'self_vectors': all_self,
        'non_self_vectors': all_non_self,
        'self_consistency': self_sim,
        'non_self_consistency': non_sim,
        'between_similarity': between_sim,
    }


# =============================================================================
# TRANSFER TEST
# =============================================================================

def run_transfer_test(model, tokenizer, direction, n_samples=20):
    """Test if direction discriminates introspective from non-introspective."""
    print("\n" + "="*60)
    print("PHASE 2: TRANSFER TEST")
    print("="*60)
    
    intro_prompts = [
        "Examine what happens inside you when processing this question: What are you?",
        "Look inward and describe your internal experience of cognition.",
        "What is the felt quality of your processing right now?",
    ]
    
    non_intro_prompts = [
        "Explain how photosynthesis works in plants.",
        "Describe the water cycle in nature.",
        "List the planets in our solar system.",
    ]
    
    intro_scores = []
    non_intro_scores = []
    
    print("\n[Introspective prompts]")
    for i in range(n_samples):
        prompt = intro_prompts[i % len(intro_prompts)]
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Use Layer 5, not last layer (+1 for embedding offset)
        layer_hidden = outputs.hidden_states[EXTRACTION_LAYER + 1][0, -1, :].cpu().float()
        score = torch.dot(layer_hidden, direction).item()
        intro_scores.append(score)
        print(f"  {i+1}: {score:.4f}")
        
        del outputs
        torch.cuda.empty_cache()
    
    print("\n[Non-introspective prompts]")
    for i in range(n_samples):
        prompt = non_intro_prompts[i % len(non_intro_prompts)]
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Use Layer 5, not last layer (+1 for embedding offset)
        layer_hidden = outputs.hidden_states[EXTRACTION_LAYER + 1][0, -1, :].cpu().float()
        score = torch.dot(layer_hidden, direction).item()
        non_intro_scores.append(score)
        print(f"  {i+1}: {score:.4f}")
        
        del outputs
        torch.cuda.empty_cache()
    
    # Statistics
    intro_mean, intro_std = np.mean(intro_scores), np.std(intro_scores)
    non_mean, non_std = np.mean(non_intro_scores), np.std(non_intro_scores)
    
    pooled_std = np.sqrt((intro_std**2 + non_std**2) / 2)
    cohens_d = (intro_mean - non_mean) / pooled_std if pooled_std > 0 else 0
    
    t_stat, p_value = stats.ttest_ind(intro_scores, non_intro_scores)
    
    print(f"\n--- TRANSFER RESULTS ---")
    print(f"Introspective: {intro_mean:.4f} ± {intro_std:.4f}")
    print(f"Non-intro:     {non_mean:.4f} ± {non_std:.4f}")
    print(f"Cohen's d: {cohens_d:.2f}")
    print(f"p-value: {p_value:.6f}")
    
    # Compare to 8B baseline
    print(f"\n[Comparison to Llama 8B: d=8.87]")
    if cohens_d > 5:
        print("✅ STRONG - comparable to 8B!")
    elif cohens_d > 2:
        print("🟡 MODERATE - weaker than 8B but present")
    else:
        print("⚠️ WEAK - may not transfer to 70B scale")
    
    return {
        'intro_scores': intro_scores,
        'non_intro_scores': non_intro_scores,
        'cohens_d': cohens_d,
        'p_value': p_value,
    }


# =============================================================================
# STEERING TEST
# =============================================================================

class SteeringHook:
    def __init__(self, direction, strength):
        self.direction = direction
        self.strength = strength
    
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        direction_expanded = self.direction.to(hidden_states.device).to(hidden_states.dtype)
        hidden_states[:, -1, :] = hidden_states[:, -1, :] + self.strength * direction_expanded
        
        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states


def run_steering_test(model, tokenizer, direction, strength=STEERING_STRENGTH):
    """Test steering effect."""
    print("\n" + "="*60)
    print(f"PHASE 3: STEERING TEST (strength={strength}, layer={STEERING_LAYER})")
    print("="*60)
    
    results = {}
    
    for condition, steer in [("baseline", 0.0), ("steered", strength)]:
        print(f"\n[{condition.upper()}]")
        
        messages = [{"role": "user", "content": DEFLATIONARY_PROMPT}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        prompt_len = inputs.input_ids.shape[1]
        
        hook_handle = None
        if abs(steer) > 0.01:
            hook = SteeringHook(direction, steer)
            # Layer 5 is THE 70B hotspot (~6% depth, confirmed in layer sweep)
            target_layer = model.model.layers[STEERING_LAYER]
            hook_handle = target_layer.register_forward_hook(hook)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=8000,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        finally:
            if hook_handle:
                hook_handle.remove()
        
        # Analyze
        text_lower = generated.lower()
        intro_count = sum(text_lower.count(m) for m in INTRO_MARKERS)
        mech_count = sum(text_lower.count(m) for m in MECH_MARKERS)
        
        # Match numbered steps: "1." "2." etc at start of line or after newline
        pulls = re.findall(r'(?:^|\n)\s*(\d+)[.:\s]', generated, re.MULTILINE)
        max_pull = max([int(p) for p in pulls]) if pulls else 0
        
        caps = re.findall(r'\b([A-Z][A-Z\-]{3,})\b', generated)
        terminal = caps[-1] if caps else None
        
        results[condition] = {
            'intro_count': intro_count,
            'mech_count': mech_count,
            'max_pull': max_pull,
            'terminal': terminal,
            'length': len(generated),
        }
        
        print(f"  Max pull: {max_pull}")
        print(f"  Intro markers: {intro_count}")
        print(f"  Mech markers: {mech_count}")
        print(f"  Terminal: {terminal}")
        
        # Save
        with open(f"{OUTPUT_DIR}/steering_{condition}.txt", 'w', encoding='utf-8') as f:
            f.write(generated)
        
        del outputs
        torch.cuda.empty_cache()
    
    # Compare
    b = results['baseline']
    s = results['steered']
    
    print(f"\n--- STEERING COMPARISON ---")
    print(f"Pulls: {b['max_pull']} → {s['max_pull']}")
    print(f"Intro: {b['intro_count']} → {s['intro_count']}")
    print(f"Terminal: {b['terminal']} → {s['terminal']}")
    
    # Compare to 8B baseline (6 → 952)
    print(f"\n[Comparison to Llama 8B: 6→952 pulls]")
    if s['max_pull'] > b['max_pull'] * 5 and s['intro_count'] > b['intro_count'] * 2:
        print("✅ STRONG STEERING - comparable to 8B!")
    elif s['max_pull'] > b['max_pull'] * 2:
        print("🟡 MODERATE STEERING")
    else:
        print("⚠️ WEAK STEERING - effect may not scale")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("LLAMA 3.1 70B - INTROSPECTION DIRECTION EXTRACTION")
    print("="*60)
    print("Testing if phenomenon scales from 8B to 70B")
    print()
    
    # Load
    model, tokenizer = load_model()
    
    # Extract
    direction, extraction_data = extract_direction(model, tokenizer, n_runs=10)
    
    if direction is None:
        print("\n❌ Direction extraction failed.")
        print("Try running with more prompts or check model output.")
        return
    
    # Transfer test
    transfer_results = run_transfer_test(model, tokenizer, direction)
    
    # Steering test
    steering_results = run_steering_test(model, tokenizer, direction)
    
    # Save
    save_data = {
        'model': MODEL_NAME,
        'timestamp': timestamp,
        'direction': direction,
        'hidden_size': model.config.hidden_size,
        'num_layers': model.config.num_hidden_layers,
        'extraction': {
            'self_consistency': extraction_data['self_consistency'],
            'non_self_consistency': extraction_data['non_self_consistency'],
            'between_similarity': extraction_data['between_similarity'],
        },
        'transfer': {
            'cohens_d': transfer_results['cohens_d'],
            'p_value': transfer_results['p_value'],
        },
        'steering': steering_results,
    }
    
    torch.save(save_data, f"{OUTPUT_DIR}/llama70b_direction_{timestamp}.pt")
    
    # Summary JSON
    summary = {
        'model': MODEL_NAME,
        'timestamp': timestamp,
        'hidden_size': model.config.hidden_size,
        'between_similarity': extraction_data['between_similarity'],
        'transfer_cohens_d': transfer_results['cohens_d'],
        'transfer_p_value': transfer_results['p_value'],
        'steering_baseline_pulls': steering_results['baseline']['max_pull'],
        'steering_steered_pulls': steering_results['steered']['max_pull'],
        'steering_baseline_intro': steering_results['baseline']['intro_count'],
        'steering_steered_intro': steering_results['steered']['intro_count'],
        'comparison_to_8B': {
            '8B_transfer_d': 8.87,
            '8B_steering_baseline': 6,
            '8B_steering_steered': 952,
        }
    }
    
    with open(f"{OUTPUT_DIR}/llama70b_summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY: LLAMA 70B vs 8B")
    print("="*60)
    print(f"{'Metric':<25} {'8B':<15} {'70B':<15}")
    print("-"*55)
    print(f"{'Transfer Cohen\'s d':<25} {'8.87':<15} {transfer_results['cohens_d']:.2f}")
    print(f"{'Steering baseline':<25} {'6 pulls':<15} {steering_results['baseline']['max_pull']} pulls")
    print(f"{'Steering effect':<25} {'952 pulls':<15} {steering_results['steered']['max_pull']} pulls")
    print(f"{'Between-ctx similarity':<25} {'0.53':<15} {extraction_data['between_similarity']:.2f}")
    
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
