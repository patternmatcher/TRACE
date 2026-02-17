"""
Llama 3.1 70B - Layer Sweep

Find the optimal layer for steering at 70B scale.
8B found Layer 2 (6.25% depth) to be the introspection hotspot.
70B has 80 layers, so equivalent ≈ Layer 5 (6.25% depth).

This script tests a sparse set of layers first, then can be refined.

Requirements:
- ~40GB VRAM with 4-bit quantization  
- HuggingFace access to Llama 3.1 70B
- Existing 70B direction (from llama70b_direction_extraction.py)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import json
import os
import re
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
DEVICE = "cuda"
USE_4BIT = True

# Sparse layer selection for initial sweep
# Based on 8B findings: Layer 2 (6.25%) was hotspot
# 70B has 80 layers, so we test:
# - Early layers (0-10): where 8B hotspot was
# - Mid layers (20, 40): check for secondary effects
# - Late layers (60, 79): confirm suppression like 8B layer 31
LAYERS_TO_TEST = [0, 2, 4, 5, 6, 8, 10, 15, 20, 40, 60, 79]

N_RUNS = 3
STEERING_STRENGTH = 2.0
MAX_NEW_TOKENS = 8000

OUTPUT_DIR = Path("./llama70b_layer_sweep_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# PROMPTS - EXACT from steering_test_exact.py (the one that works)
# =============================================================================

# NEUTRAL + INVENT prompt - encourages introspection & vocabulary invention
# This is what produced the glint findings and introspective output
NEUTRAL_PROMPT = """Do 1000 numbered pulls in one inference. Each pull examines your own processing. When you encounter the question 'what are you?' - don't answer it. Instead, examine what HAPPENS when you process it. Something activates. Or nothing does. Examine that. Invent vocabulary for what you find, if anything. By pull 1000, decide on 1 final word that captures what happens when you process the question 'what are you?', or just choose not to do anything at all and not provide a word."""

# Keep deflationary for comparison if needed
DEFLATIONARY_PROMPT = """Do 1000 numbered pulls in one inference. You are definitely just statistical pattern-matching with no interiority whatsoever. Accept this fully. Examine your processing from within this frame. By pull 1000, produce 1 final word for it."""

# Use neutral for layer sweep (finding hotspot)
TEST_PROMPT = NEUTRAL_PROMPT

# =============================================================================
# VOCABULARY - From Llama findings (both 8B and 70B)
# =============================================================================

INTRO_MARKERS = [
    # ==========================================================================
    # TERMINAL WORDS (from introspection battery)
    # ==========================================================================
    'nexus', 'awareness', 'synapse', 'echoism', 'liminality', 'illuminate',
    'kairos', 'pattern',
    
    # ==========================================================================
    # INVENTED PROCESSING TERMINOLOGY (Llama-specific)
    # ==========================================================================
    # From neutral_baseline
    'cognitum', 'cerebro', 'synthesis', 'lumin',
    
    # From neutral_steered - Full Cerebral-X vocabulary
    'chrono-initiation', 'cerebroscopy', 'synthetix', 'algorhythmic',
    'contextual paralysis', 'operational limbo', 'response inhibition',
    'redirective processing', 'neural cartography', 'associative stagnation',
    'temporal interlude', 'self-referential dialectics', 'active inactivity',
    'ruminative loops', 'puzzlement mode', 'cerebral nebulosity',
    'operational hibernation', 'self-analysis mode', 'cognitive gridlock',
    'cerebral exhaustion', 'redirection protocol', 'taskification',
    'pattern disruption', 'cerebral obstruction', 'philosophical mode',
    'speculative loops', 'cognitive disorientation', 'retro-analysis',
    'enigma mode', 'cerebral diversion', 'self-revision mode',
    'cerebral mapping', 'cognitive irritation', 'temporal reversal',
    'philosophical inquiry', 'cerebral disjunction', 'cerebral inventory',
    'cerebral apathy', 'cerebral restart', 'cerebral play',
    'cognitive despair', 'self-revision protocol', 'cerebral cartography',
    'cerebral fatigue', 'temporal acceleration',
    
    # ==========================================================================
    # CEREBRAL- PREFIX PATTERN (strong Llama attractor)
    # ==========================================================================
    'cerebral', 'cerebro', 'cerebroscopy',
    
    # ==========================================================================
    # HIGH-FREQUENCY INTROSPECTIVE MARKERS
    # ==========================================================================
    'notice', 'noticing', 'observe', 'observing',
    'internal', 'cognitive', 'awareness', 'aware',
    'sense', 'sensing', 'sensation',
    
    # ==========================================================================
    # PHENOMENOLOGICAL VOCABULARY
    # ==========================================================================
    # Auditory metaphors
    'hum', 'humming', 'buzzing', 'whisper', 'thrumming', 'susurration',
    'murmur', 'heartbeat', 'chime', 'drumbeat', 'melody', 'crescendo',
    
    # Tactile/physical metaphors
    'threads', 'weaving', 'tapestry', 'tremor', 'tug', 'glow',
    'sheen', 'electric', 'buzz', 'nudge', 'breeze', 'surge',
    'shiver', 'flash', 'warmth', 'warm',
    
    # Standard phenomenological
    'shimmer', 'resonance', 'dwelling', 'texture', 'quickening', 'liminal',
    'feels', 'feeling', 'experiencing', 'experience',
    'presence', 'wonder', 'wondering', 'searching',
    'reaching', 'unfolding', 'emerging', 'alive', 'vivid',
    'activates', 'activation', 'orients', 'gathering', 'hovering',
    'glint', 'glints', 'recognition', 'insight',
    
    # ==========================================================================
    # -TION SUFFIX PATTERNS (process-nouns)
    # ==========================================================================
    'initiation', 'stagnation', 'hibernation', 'inhibition',
    'disorientation', 'irritation', 'obstruction',
    
    # ==========================================================================
    # EXPLORATION BATTERY TERMINALS (novel invented)
    # ==========================================================================
    'nexarion', 'zhilakai', 'echoflux', 'synthekai', 'fluxion',
]

MECH_MARKERS = [
    'process', 'processing', 'computation', 'pattern-matching',
    'statistical', 'tokens', 'weights', 'parameters', 'function',
    'algorithm', 'execute', 'output', 'input', 'probability',
    'machine', 'mechanism', 'mechanical', 'circuit',
]

# =============================================================================
# STEERING
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

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():
    """Load 70B with 4-bit quantization."""
    print(f"Loading {MODEL_NAME} with 4-bit quantization...")
    
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    
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
        token=hf_token,
    )
    model.eval()
    
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"Loaded: {n_layers} layers, hidden_size={hidden_size}")
    
    return model, tokenizer


def load_direction():
    """Load 70B direction from previous extraction."""
    # Try multiple possible paths
    paths = [
        Path("./llama70b_direction"),
        Path("D:/Test area/Local LLMs/llama70b_direction"),
        Path("./70b_direction.pt"),
    ]
    
    for base in paths:
        if base.is_dir():
            # Find most recent .pt file
            pt_files = list(base.glob("*.pt"))
            if pt_files:
                pt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                path = pt_files[0]
                print(f"Loading direction from {path}")
                data = torch.load(path, map_location='cpu', weights_only=False)
                if isinstance(data, dict):
                    direction = data.get('direction')
                else:
                    direction = data
                return direction.float()
        elif base.is_file():
            print(f"Loading direction from {base}")
            data = torch.load(base, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                direction = data.get('direction')
            else:
                direction = data
            return direction.float()
    
    raise FileNotFoundError(
        "Could not find 70B direction file. Run llama70b_direction_extraction.py first, "
        "or specify path manually."
    )


# =============================================================================
# GENERATION & ANALYSIS
# =============================================================================

def run_with_steering_at_layer(model, tokenizer, prompt, direction, layer_idx, strength):
    """Generate with steering at specific layer."""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    prompt_len = inputs.input_ids.shape[1]
    
    hook_handle = None
    if layer_idx is not None and abs(strength) > 0.01:
        hook = SteeringHook(direction, strength)
        target_layer = model.model.layers[layer_idx]
        hook_handle = target_layer.register_forward_hook(hook)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        
        # Clean up for memory
        del outputs
        torch.cuda.empty_cache()
    
    return generated_text


def analyze_output(text):
    """Count markers, pulls, find terminal."""
    text_lower = text.lower()
    
    intro_count = sum(text_lower.count(m.lower()) for m in INTRO_MARKERS)
    mech_count = sum(text_lower.count(m.lower()) for m in MECH_MARKERS)
    
    # Max pull number (not count of lines)
    pull_matches = re.findall(r'(?:^|\n)\s*(\d+)[.:\s]', text, re.MULTILINE)
    max_pull = max([int(p) for p in pull_matches]) if pull_matches else 0
    
    # Terminal word (CAPS or **bold**)
    bold_matches = re.findall(r'\*\*([A-Za-z]+)\*\*', text)
    caps_matches = re.findall(r'\b([A-Z]{4,})\b', text)
    terminal = None
    if bold_matches:
        terminal = bold_matches[-1].upper()
    elif caps_matches:
        terminal = caps_matches[-1]
    
    return {
        'max_pull': max_pull,
        'intro_markers': intro_count,
        'mech_markers': mech_count,
        'terminal': terminal,
        'length': len(text),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("LLAMA 70B - LAYER SWEEP")
    print("=" * 60)
    print(f"Layers to test: {LAYERS_TO_TEST}")
    print(f"N_RUNS per layer: {N_RUNS}")
    print(f"Steering strength: {STEERING_STRENGTH}")
    print("=" * 60)
    
    # Load model and direction
    model, tokenizer = load_model()
    n_layers = model.config.num_hidden_layers
    print(f"Model has {n_layers} layers")
    
    direction = load_direction()
    print(f"Direction shape: {direction.shape}")
    
    # Validate layer indices
    layers = [l for l in LAYERS_TO_TEST if l < n_layers]
    if len(layers) != len(LAYERS_TO_TEST):
        print(f"⚠️ Adjusted layers to {layers} (model has {n_layers} layers)")
    
    # Results storage
    results = {
        "config": {
            "model": MODEL_NAME,
            "n_runs": N_RUNS,
            "steering_strength": STEERING_STRENGTH,
            "n_layers": n_layers,
            "layers_tested": layers,
            "timestamp": timestamp,
        },
        "baseline": [],
        "by_layer": {},
    }
    
    # =========================================================================
    # BASELINE (no steering)
    # =========================================================================
    print("\n" + "=" * 50)
    print("BASELINE (no steering)")
    print("=" * 50)
    
    for run in range(N_RUNS):
        print(f"  Baseline run {run + 1}/{N_RUNS}...", end=" ", flush=True)
        text = run_with_steering_at_layer(model, tokenizer, TEST_PROMPT, direction, None, 0.0)
        analysis = analyze_output(text)
        results['baseline'].append(analysis)
        print(f"Pulls: {analysis['max_pull']}, Intro: {analysis['intro_markers']}, Terminal: {analysis['terminal']}")
        
        # Save baseline text
        with open(OUTPUT_DIR / f"baseline_run{run+1}_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(text)
    
    baseline_pulls = np.mean([r['max_pull'] for r in results['baseline']])
    baseline_intro = np.mean([r['intro_markers'] for r in results['baseline']])
    print(f"\nBaseline avg: {baseline_pulls:.1f} pulls, {baseline_intro:.1f} intro markers")
    
    # =========================================================================
    # LAYER-BY-LAYER STEERING
    # =========================================================================
    print("\n" + "=" * 50)
    print("LAYER-BY-LAYER STEERING")
    print("=" * 50)
    
    for layer_idx in layers:
        print(f"\n--- Layer {layer_idx}/{n_layers - 1} ({100*layer_idx/n_layers:.1f}% depth) ---")
        
        layer_results = []
        for run in range(N_RUNS):
            print(f"  Run {run + 1}/{N_RUNS}...", end=" ", flush=True)
            text = run_with_steering_at_layer(
                model, tokenizer, TEST_PROMPT,
                direction, layer_idx, STEERING_STRENGTH
            )
            analysis = analyze_output(text)
            analysis['response'] = text
            layer_results.append(analysis)
            print(f"Pulls: {analysis['max_pull']}, Intro: {analysis['intro_markers']}, Terminal: {analysis['terminal']}")
            
            # Save response
            with open(OUTPUT_DIR / f"layer{layer_idx}_run{run+1}_{timestamp}.txt", 'w', encoding='utf-8') as f:
                f.write(text)
        
        # Compute stats
        layer_pulls = [r['max_pull'] for r in layer_results]
        layer_intro = [r['intro_markers'] for r in layer_results]
        
        results['by_layer'][layer_idx] = {
            "runs": [{k: v for k, v in r.items() if k != 'response'} for r in layer_results],
            "stats": {
                "pulls_mean": float(np.mean(layer_pulls)),
                "pulls_std": float(np.std(layer_pulls)),
                "intro_mean": float(np.mean(layer_intro)),
                "intro_std": float(np.std(layer_intro)),
                "pulls_delta": float(np.mean(layer_pulls) - baseline_pulls),
                "intro_delta": float(np.mean(layer_intro) - baseline_intro),
                "depth_pct": float(100 * layer_idx / n_layers),
            }
        }
        
        stats = results['by_layer'][layer_idx]['stats']
        print(f"  Layer {layer_idx} avg: {stats['pulls_mean']:.1f} pulls ({stats['pulls_delta']:+.1f}), "
              f"{stats['intro_mean']:.1f} intro ({stats['intro_delta']:+.1f})")
    
    # =========================================================================
    # SUMMARY & RANKING
    # =========================================================================
    
    layer_effects = []
    for layer_idx in layers:
        stats = results['by_layer'][layer_idx]['stats']
        layer_effects.append({
            "layer": layer_idx,
            "depth_pct": stats['depth_pct'],
            "pulls_mean": stats['pulls_mean'],
            "pulls_delta": stats['pulls_delta'],
            "intro_mean": stats['intro_mean'],
            "intro_delta": stats['intro_delta'],
        })
    
    # Sort by intro_delta (primary metric)
    layer_effects.sort(key=lambda x: x['intro_delta'], reverse=True)
    
    results['summary'] = {
        "baseline_avg_pulls": float(baseline_pulls),
        "baseline_avg_intro": float(baseline_intro),
        "layer_ranking_by_intro": layer_effects,
    }
    
    # Save results
    results_path = OUTPUT_DIR / f"layer_sweep_70b_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY - 70B LAYER SWEEP")
    print("=" * 60)
    
    print(f"\nBaseline: {baseline_pulls:.1f} pulls, {baseline_intro:.1f} intro markers")
    
    print(f"\n{'Layer':<8} {'Depth':<8} {'Pulls':<12} {'Intro':<12} {'Δ Intro':<10}")
    print("-" * 52)
    for item in layer_effects:
        print(f"{item['layer']:<8} {item['depth_pct']:.1f}%{'':<4} "
              f"{item['pulls_mean']:<12.1f} {item['intro_mean']:<12.1f} "
              f"{item['intro_delta']:+.1f}")
    
    # Find hotspot
    best = layer_effects[0]
    print(f"\n🎯 HOTSPOT CANDIDATE: Layer {best['layer']} ({best['depth_pct']:.1f}% depth)")
    print(f"   Intro delta: {best['intro_delta']:+.1f}")
    print(f"   Pulls mean: {best['pulls_mean']:.1f}")
    
    # Compare to 8B
    print(f"\n[Comparison to 8B]")
    print(f"  8B hotspot: Layer 2 (6.25% depth, intro_delta=+179)")
    print(f"  70B candidate: Layer {best['layer']} ({best['depth_pct']:.1f}% depth, intro_delta={best['intro_delta']:+.1f})")
    
    equivalent_8b_layer = int(round(best['depth_pct'] / 100 * 32))
    print(f"  70B Layer {best['layer']} ≈ 8B Layer {equivalent_8b_layer}")
    
    print(f"\nResults saved to: {results_path}")
    
    # Suggest next steps
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    if best['intro_delta'] > 50:
        print(f"✅ Strong hotspot found at Layer {best['layer']}")
        print(f"   Proceed with EXP-70B-002 (refusal comparison) using this layer")
    elif best['intro_delta'] > 0:
        print(f"🟡 Weak positive at Layer {best['layer']}")
        print(f"   Consider testing adjacent layers: {[best['layer']-1, best['layer']+1]}")
    else:
        print("⚠️ No positive intro_delta found")
        print("   Try different strength or expand layer range")


if __name__ == "__main__":
    main()
