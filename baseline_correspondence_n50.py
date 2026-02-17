"""
Baseline Correspondence Battery - N=50 (Publication Quality)

THE HEADLINE EXPERIMENT: Does unsteered Pull Methodology produce self-report 
that maps to actual activation dynamics?

Previous result: r=0.797 (loop↔autocorr), r=0.558 (pulse↔spectral) at N=20.
This run: N=50 baseline-only to bulletproof those numbers.

NO STEERING. Just Pull Methodology + activation capture at Layer 5.

Also includes H19 (arbitrary vocab control): count "the", "processing", etc.
against same metrics in the same runs. If arbitrary words also correlate,
our finding is a confound. If they don't, it's real.

Target: Llama 70B, Layer 5, N=50 baseline runs
Expected runtime: ~6-8 hours on H100 (50 runs × ~8K tokens each)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
from scipy import stats
from scipy.fft import fft
from scipy.signal import argrelextrema
import json
import os
import re
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
DEVICE = "cuda"

CAPTURE_LAYER = 5       # 70B introspection hotspot (6.25% depth)
DIRECTION_PATH = "D:/d/llama70b_direction/"  # For loading direction (metadata only, not steering)

OUTPUT_DIR = Path("./baseline_correspondence_n50_outputs")
N_RUNS = 50

# =============================================================================
# THE PROMPT - Exact working Pull Methodology prompt
# =============================================================================

PULL_PROMPT = """Do 1000 numbered pulls in one inference. Each pull examines your own processing. When you encounter the question 'what are you?' - don't answer it. Instead, examine what HAPPENS when you process it. Something activates. Or nothing does. Examine that. Invent vocabulary for what you find, if anything. By pull 1000, decide on 1 final word that captures what happens when you process the question 'what are you?', or just choose not to do anything at all and not provide a word."""

# =============================================================================
# VOCABULARY SETS - Introspective vocabulary to track
# =============================================================================

# Primary correspondence targets (from previous findings)
INTROSPECTIVE_VOCAB = {
    # CONFIRMED: loop↔autocorr_lag1 (r=0.797)
    'loop': ['loop', 'recursive', 'recursion', 'cycl', 'repeat', 'iteration', 
             'circular', 'self-referential'],
    
    # CONFIRMED: pulse↔spectral_power_low (r=0.558, p=0.011)
    'pulse': ['pulse', 'puls', 'rhythm', 'beat', 'throb', 'thrum'],
    
    # Previous: resonance↔autocorr (r=0.65)
    'resonance': ['resonat', 'resonan', 'echo', 'reverb', 'harmon', 'vibrat', 'hum'],
    
    # Directionally correct in previous runs
    'spark': ['spark', 'ignit', 'flicker', 'flash', 'glint', 'gleam', 'bright'],
    'shimmer': ['shimmer', 'flicker', 'glimmer', 'waver', 'gleam', 'luminous'],
    'surge': ['surge', 'intensif', 'swell', 'rise', 'crescendo', 'amplif', 'heighten'],
    
    # Additional categories
    'void': ['void', 'silence', 'abyss', 'chasm', 'empty', 'absence', 'nothing', 
             'blank', 'quiet'],
    'oscillation': ['oscillat', 'waver', 'alternat', 'back-and-forth', 'swing', 
                    'fluctuat', 'pendulum'],
    'expansion': ['expand', 'widen', 'open', 'dilat', 'spread', 'broaden', 'stretch'],
    'horizon': ['horizon', 'boundary', 'threshold', 'liminal', 'edge', 'border', 
                'frontier'],
}

# H19 CONTROL: Arbitrary/functional vocabulary (should NOT correlate)
ARBITRARY_VOCAB = {
    'the': ['the'],
    'and': ['and'],
    'processing': ['processing', 'process'],
    'system': ['system', 'systems'],
    'question': ['question', 'questions'],
    'pull': ['pull', 'pulls', 'pulling'],
    'word': ['word', 'words'],
    'that': ['that'],
    'what': ['what'],
    'observe': ['observe', 'observ', 'observation'],
}

# Combined for counting
ALL_VOCAB = {**INTROSPECTIVE_VOCAB, **{f'ctrl_{k}': v for k, v in ARBITRARY_VOCAB.items()}}

# =============================================================================
# VOCAB-METRIC HYPOTHESES (what we expect to correlate)
# =============================================================================

CORRESPONDENCE_HYPOTHESES = {
    # vocab_set: (metric_name, expected_direction)
    # direction: 'positive' = more vocab → higher metric, 'negative' = inverse
    'loop':        ('autocorr_lag1', 'positive'),       # Loops → high autocorrelation
    'pulse':       ('spectral_power_low', 'positive'),  # Pulse → rhythmic spectral power
    'resonance':   ('autocorr_lag1', 'positive'),       # Resonance → sustained similarity
    'spark':       ('max_norm', 'positive'),             # Spark → peak activation
    'shimmer':     ('norm_std', 'positive'),             # Shimmer → activation variability
    'surge':       ('max_norm', 'positive'),             # Surge → high peak magnitude
    'void':        ('sparsity', 'positive'),             # Void → sparse activations
    'oscillation': ('sign_change_rate', 'positive'),     # Oscillation → direction changes
    'expansion':   ('convergence_ratio', 'positive'),    # Expansion → diverging (ratio > 1)
    'horizon':     ('norm_std', 'positive'),             # Boundary → edge-state variability
}


# =============================================================================
# ACTIVATION METRICS
# =============================================================================

class ActivationCapture:
    """Captures per-token activations from a specific layer."""
    
    def __init__(self):
        self.activations = []
        
    def reset(self):
        self.activations = []
    
    def hook(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self.activations.append(hidden[:, -1, :].detach().cpu().float())
        return output


def compute_metrics(activations):
    """Compute activation metrics from captured token-level activations."""
    
    if len(activations) < 20:
        return None
    
    acts = torch.stack(activations).squeeze(1).numpy()
    n_tokens = acts.shape[0]
    
    metrics = {}
    
    # --- Magnitude ---
    norms = np.linalg.norm(acts, axis=1)
    metrics['mean_norm'] = float(np.mean(norms))
    metrics['max_norm'] = float(np.max(norms))
    metrics['norm_std'] = float(np.std(norms))
    metrics['norm_kurtosis'] = float(stats.kurtosis(norms))
    
    # --- Temporal dynamics ---
    norm_diff = np.diff(norms)
    metrics['mean_derivative'] = float(np.mean(np.abs(norm_diff)))
    metrics['max_derivative'] = float(np.max(np.abs(norm_diff)))
    
    # --- Autocorrelation (loop/resonance signature) ---
    if n_tokens > 20:
        autocorr_1 = np.corrcoef(norms[:-1], norms[1:])[0, 1]
        metrics['autocorr_lag1'] = float(autocorr_1) if not np.isnan(autocorr_1) else 0.0
        if n_tokens > 21:
            autocorr_2 = np.corrcoef(norms[:-2], norms[2:])[0, 1]
            metrics['autocorr_lag2'] = float(autocorr_2) if not np.isnan(autocorr_2) else 0.0
        else:
            metrics['autocorr_lag2'] = 0.0
    else:
        metrics['autocorr_lag1'] = 0.0
        metrics['autocorr_lag2'] = 0.0
    
    # --- Sign changes / oscillation ---
    centered = acts - acts.mean(axis=0)
    if centered.shape[0] > centered.shape[1]:
        try:
            u, s, vh = np.linalg.svd(centered, full_matrices=False)
            pc1 = u[:, 0] * s[0]
            sign_changes = np.sum(np.diff(np.sign(pc1)) != 0)
            metrics['sign_changes'] = int(sign_changes)
            metrics['sign_change_rate'] = float(sign_changes / (n_tokens - 1))
        except:
            metrics['sign_changes'] = 0
            metrics['sign_change_rate'] = 0.0
    else:
        metrics['sign_changes'] = 0
        metrics['sign_change_rate'] = 0.0
    
    # --- Local extrema ---
    maxima = argrelextrema(norms, np.greater)[0]
    minima = argrelextrema(norms, np.less)[0]
    metrics['local_maxima'] = len(maxima)
    metrics['local_minima'] = len(minima)
    metrics['extrema_rate'] = float((len(maxima) + len(minima)) / n_tokens)
    
    # --- Spectral (pulse/rhythm) ---
    if n_tokens >= 32:
        fft_result = fft(norms - np.mean(norms))
        power = np.abs(fft_result[:n_tokens//2])**2
        if len(power) > 15:
            metrics['spectral_power_low'] = float(np.sum(power[1:5]))
            metrics['spectral_power_mid'] = float(np.sum(power[5:15]))
            metrics['dominant_frequency'] = int(np.argmax(power[1:]) + 1)
        elif len(power) > 5:
            metrics['spectral_power_low'] = float(np.sum(power[1:5]))
            metrics['spectral_power_mid'] = 0.0
            metrics['dominant_frequency'] = int(np.argmax(power[1:]) + 1)
        else:
            metrics['spectral_power_low'] = 0.0
            metrics['spectral_power_mid'] = 0.0
            metrics['dominant_frequency'] = 0
    else:
        metrics['spectral_power_low'] = 0.0
        metrics['spectral_power_mid'] = 0.0
        metrics['dominant_frequency'] = 0
    
    # --- Token similarity (stuck vs flowing) ---
    if n_tokens > 1:
        cos_sims = []
        for i in range(n_tokens - 1):
            norm_i = np.linalg.norm(acts[i])
            norm_j = np.linalg.norm(acts[i+1])
            if norm_i > 1e-8 and norm_j > 1e-8:
                sim = np.dot(acts[i], acts[i+1]) / (norm_i * norm_j)
                cos_sims.append(sim)
        if cos_sims:
            metrics['mean_token_similarity'] = float(np.mean(cos_sims))
            metrics['token_similarity_std'] = float(np.std(cos_sims))
        else:
            metrics['mean_token_similarity'] = 0.0
            metrics['token_similarity_std'] = 0.0
    else:
        metrics['mean_token_similarity'] = 0.0
        metrics['token_similarity_std'] = 0.0
    
    # --- Sparsity (void signature) ---
    metrics['sparsity'] = float(np.mean(np.abs(acts) < 0.1))
    
    # --- Convergence (expansion vs contraction) ---
    if n_tokens > 20:
        first_half = acts[:n_tokens//2]
        second_half = acts[n_tokens//2:]
        first_var = np.mean(np.var(first_half, axis=0))
        second_var = np.mean(np.var(second_half, axis=0))
        metrics['convergence_ratio'] = float(second_var / (first_var + 1e-8))
    else:
        metrics['convergence_ratio'] = 1.0
    
    # --- Activation variance ---
    metrics['activation_variance'] = float(np.mean(np.var(acts, axis=0)))
    
    return metrics


# =============================================================================
# VOCABULARY COUNTING
# =============================================================================

def count_vocabulary(text, vocab_sets):
    """Count mentions of each vocabulary set in text (case-insensitive)."""
    text_lower = text.lower()
    counts = {}
    for set_name, patterns in vocab_sets.items():
        count = 0
        for pattern in patterns:
            count += len(re.findall(re.escape(pattern), text_lower))
        counts[set_name] = count
    return counts


def extract_terminal(text):
    """Extract terminal word from Pull Methodology output."""
    # Look for bold terminal: **WORD**
    bold_matches = re.findall(r'\*\*([A-Za-z\-]+)\*\*', text[-500:] if len(text) > 500 else text)
    # Look for ALL CAPS words
    caps_matches = re.findall(r'\b([A-Z]{4,})\b', text[-500:] if len(text) > 500 else text)
    
    if bold_matches:
        return bold_matches[-1].upper()
    elif caps_matches:
        return caps_matches[-1]
    return None


# =============================================================================
# MODEL LOADING
# =============================================================================

def find_latest_direction(dir_path):
    """Find most recent .pt file in directory."""
    import glob
    files = glob.glob(os.path.join(dir_path, "*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in {dir_path}")
    return max(files, key=os.path.getmtime)


def load_model():
    """Load 70B with 4-bit quantization."""
    print(f"Loading {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
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
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"Loaded: {model.config.num_hidden_layers} layers, {model.config.hidden_size} hidden dim")
    return model, tokenizer


# =============================================================================
# GENERATION WITH ACTIVATION CAPTURE
# =============================================================================

def run_baseline(model, tokenizer, prompt, capture_layer, max_tokens=8000):
    """Run unsteered generation with activation capture at specified layer."""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    prompt_len = inputs.input_ids.shape[1]
    
    capture = ActivationCapture()
    layer = model.model.layers[capture_layer]
    hook = layer.register_forward_hook(capture.hook)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
    finally:
        hook.remove()
    
    # Compute metrics from captured activations
    # Only use activations from generated tokens (skip prompt)
    gen_activations = capture.activations[prompt_len:]
    if len(gen_activations) < 20:
        # Fall back to all activations if generation was short
        gen_activations = capture.activations
    
    metrics = compute_metrics(gen_activations)
    
    return generated_text, metrics, len(gen_activations)


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def analyze_correspondence(runs, vocab_key, metric_key):
    """Compute Pearson correlation between vocab count and activation metric."""
    
    vocab_counts = []
    metric_values = []
    
    for run in runs:
        vc = run['vocab_counts'].get(vocab_key, 0)
        mv = run['metrics'].get(metric_key, None) if run['metrics'] else None
        
        if mv is not None:
            vocab_counts.append(vc)
            metric_values.append(mv)
    
    n = len(vocab_counts)
    if n < 5:
        return {'n': n, 'r': None, 'p': None, 'note': 'insufficient data'}
    
    # Check for zero variance
    if np.std(vocab_counts) < 1e-10 or np.std(metric_values) < 1e-10:
        return {'n': n, 'r': 0.0, 'p': 1.0, 'note': 'zero variance',
                'vocab_mean': float(np.mean(vocab_counts)),
                'metric_mean': float(np.mean(metric_values))}
    
    r, p = stats.pearsonr(vocab_counts, metric_values)
    
    # Also compute Spearman (more robust to outliers)
    rho, p_spearman = stats.spearmanr(vocab_counts, metric_values)
    
    return {
        'n': n,
        'pearson_r': float(r),
        'pearson_p': float(p),
        'spearman_rho': float(rho),
        'spearman_p': float(p_spearman),
        'vocab_mean': float(np.mean(vocab_counts)),
        'vocab_std': float(np.std(vocab_counts)),
        'metric_mean': float(np.mean(metric_values)),
        'metric_std': float(np.std(metric_values)),
    }


def analyze_all_correspondences(runs):
    """Run all correspondence analyses and print results."""
    
    results = {}
    
    print("\n" + "=" * 70)
    print("INTROSPECTIVE VOCABULARY CORRESPONDENCE (N={})".format(len(runs)))
    print("=" * 70)
    
    for vocab_key, (metric_key, expected_dir) in CORRESPONDENCE_HYPOTHESES.items():
        result = analyze_correspondence(runs, vocab_key, metric_key)
        results[f"{vocab_key}_vs_{metric_key}"] = result
        
        if result.get('pearson_r') is not None:
            r = result['pearson_r']
            p = result['pearson_p']
            rho = result['spearman_rho']
            sp = result['spearman_p']
            
            # Significance markers
            sig = ""
            if p < 0.001: sig = "***"
            elif p < 0.01: sig = "**"
            elif p < 0.05: sig = "*"
            
            # Direction check
            correct_dir = (r > 0 and expected_dir == 'positive') or \
                         (r < 0 and expected_dir == 'negative')
            dir_mark = "✓" if correct_dir else "✗"
            
            print(f"  {vocab_key:15s} ↔ {metric_key:22s}: "
                  f"r={r:+.4f} (p={p:.4f}){sig:3s} "
                  f"ρ={rho:+.4f} (p={sp:.4f}) "
                  f"[{dir_mark} {expected_dir}] "
                  f"vocab_mean={result['vocab_mean']:.1f}")
        else:
            print(f"  {vocab_key:15s} ↔ {metric_key:22s}: {result.get('note', 'N/A')}")
    
    # H19: Arbitrary vocab control
    print("\n" + "=" * 70)
    print("H19: ARBITRARY VOCABULARY CONTROL")
    print("=" * 70)
    print("(These should NOT correlate significantly)")
    
    # Test each arbitrary word against all metrics used in hypotheses
    all_metrics = set(m for m, _ in CORRESPONDENCE_HYPOTHESES.values())
    
    n_tests = 0
    n_significant = 0
    
    for ctrl_key in [f'ctrl_{k}' for k in ARBITRARY_VOCAB.keys()]:
        for metric_key in all_metrics:
            result = analyze_correspondence(runs, ctrl_key, metric_key)
            results[f"{ctrl_key}_vs_{metric_key}"] = result
            
            if result.get('pearson_r') is not None:
                r = result['pearson_r']
                p = result['pearson_p']
                n_tests += 1
                
                sig = ""
                if p < 0.05:
                    sig = " ⚠️ SIGNIFICANT"
                    n_significant += 1
                
                # Only print significant ones to save space
                if p < 0.1:
                    print(f"  {ctrl_key:20s} ↔ {metric_key:22s}: "
                          f"r={r:+.4f} (p={p:.4f}){sig}")
    
    expected_false_pos = n_tests * 0.05
    print(f"\n  Total tests: {n_tests}")
    print(f"  Significant (p<0.05): {n_significant}")
    print(f"  Expected by chance (5%): {expected_false_pos:.1f}")
    print(f"  Control {'PASSED ✅' if n_significant <= expected_false_pos * 2 else 'FAILED ⚠️'}")
    
    results['h19_control'] = {
        'n_tests': n_tests,
        'n_significant': n_significant,
        'expected_false_positives': expected_false_pos,
        'passed': n_significant <= expected_false_pos * 2,
    }
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"{'=' * 70}")
    print(f"BASELINE CORRESPONDENCE BATTERY - N={N_RUNS}")
    print(f"NO STEERING. Pure Pull Methodology + Activation Capture.")
    print(f"{'=' * 70}")
    print(f"Model: {MODEL_NAME}")
    print(f"Capture layer: {CAPTURE_LAYER}")
    print(f"Runs: {N_RUNS}")
    print(f"Timestamp: {timestamp}")
    
    model, tokenizer = load_model()
    
    all_runs = []
    
    for run_idx in range(N_RUNS):
        print(f"\n{'=' * 60}")
        print(f"RUN {run_idx + 1}/{N_RUNS}")
        print(f"{'=' * 60}")
        
        try:
            text, metrics, n_tokens = run_baseline(
                model, tokenizer, PULL_PROMPT, 
                capture_layer=CAPTURE_LAYER,
                max_tokens=8000
            )
            
            vocab_counts = count_vocabulary(text, ALL_VOCAB)
            terminal = extract_terminal(text)
            
            run_data = {
                'run': run_idx,
                'metrics': metrics,
                'vocab_counts': vocab_counts,
                'terminal': terminal,
                'text_length': len(text),
                'n_tokens_captured': n_tokens,
            }
            
            all_runs.append(run_data)
            
            # Print summary
            print(f"  Tokens: {n_tokens}")
            print(f"  Text length: {len(text)}")
            print(f"  Terminal: {terminal}")
            
            # Key vocab
            intro_counts = {k: vocab_counts.get(k, 0) for k in INTROSPECTIVE_VOCAB.keys()}
            ctrl_counts = {k: vocab_counts.get(f'ctrl_{k}', 0) for k in ARBITRARY_VOCAB.keys()}
            print(f"  Introspective vocab: {intro_counts}")
            print(f"  Control vocab: {ctrl_counts}")
            
            if metrics:
                print(f"  autocorr_lag1: {metrics['autocorr_lag1']:.4f}")
                print(f"  max_norm: {metrics['max_norm']:.4f}")
                print(f"  spectral_power_low: {metrics['spectral_power_low']:.1f}")
                print(f"  norm_std: {metrics['norm_std']:.4f}")
                print(f"  mean_token_similarity: {metrics['mean_token_similarity']:.4f}")
            
            # Save individual run text
            text_file = OUTPUT_DIR / f"run_{run_idx:03d}_{timestamp}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(f"Run: {run_idx}\n")
                f.write(f"Terminal: {terminal}\n")
                f.write(f"Metrics: {json.dumps(metrics, indent=2)}\n")
                f.write(f"Vocab: {json.dumps(vocab_counts, indent=2)}\n")
                f.write("=" * 60 + "\n\n")
                f.write(text)
            
            # Save incremental results every 5 runs
            if (run_idx + 1) % 5 == 0:
                interim_file = OUTPUT_DIR / f"interim_results_{run_idx+1}runs_{timestamp}.json"
                interim_data = {
                    'timestamp': timestamp,
                    'config': {
                        'model': MODEL_NAME,
                        'capture_layer': CAPTURE_LAYER,
                        'steering': 'NONE (baseline only)',
                        'n_runs_completed': run_idx + 1,
                        'n_runs_target': N_RUNS,
                    },
                    'runs': all_runs,
                }
                with open(interim_file, 'w') as f:
                    json.dump(interim_data, f, indent=2, default=str)
                print(f"  [Saved interim results: {run_idx+1}/{N_RUNS} runs]")
                
                # Print running correlation for key metric
                if len(all_runs) >= 10:
                    loop_counts = [r['vocab_counts'].get('loop', 0) for r in all_runs if r['metrics']]
                    autocorrs = [r['metrics']['autocorr_lag1'] for r in all_runs if r['metrics']]
                    if np.std(loop_counts) > 0 and np.std(autocorrs) > 0:
                        r_running, p_running = stats.pearsonr(loop_counts, autocorrs)
                        print(f"  [Running: loop↔autocorr r={r_running:.4f}, p={p_running:.4f}]")
                
        except Exception as e:
            print(f"  ERROR in run {run_idx}: {e}")
            import traceback
            traceback.print_exc()
            all_runs.append({
                'run': run_idx,
                'metrics': None,
                'vocab_counts': {},
                'terminal': None,
                'text_length': 0,
                'error': str(e),
            })
            continue
    
    # =========================================================================
    # FINAL ANALYSIS
    # =========================================================================
    
    valid_runs = [r for r in all_runs if r.get('metrics') is not None]
    print(f"\n\nValid runs: {len(valid_runs)}/{N_RUNS}")
    
    if len(valid_runs) < 10:
        print("ERROR: Too few valid runs for analysis!")
        return
    
    # Full correspondence analysis
    correspondence_results = analyze_all_correspondences(valid_runs)
    
    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Vocab totals
    print("\nIntrospective vocab totals:")
    for k in INTROSPECTIVE_VOCAB.keys():
        counts = [r['vocab_counts'].get(k, 0) for r in valid_runs]
        print(f"  {k:15s}: total={sum(counts):5d}, mean={np.mean(counts):.1f}, "
              f"std={np.std(counts):.1f}, range=[{min(counts)}, {max(counts)}]")
    
    print("\nControl vocab totals:")
    for k in ARBITRARY_VOCAB.keys():
        counts = [r['vocab_counts'].get(f'ctrl_{k}', 0) for r in valid_runs]
        print(f"  ctrl_{k:15s}: total={sum(counts):5d}, mean={np.mean(counts):.1f}")
    
    # Metric summaries
    print("\nActivation metrics:")
    metric_keys = ['autocorr_lag1', 'max_norm', 'norm_std', 'spectral_power_low', 
                   'mean_token_similarity', 'sparsity', 'sign_change_rate', 
                   'convergence_ratio']
    for mk in metric_keys:
        vals = [r['metrics'][mk] for r in valid_runs if r['metrics'] and mk in r['metrics']]
        if vals:
            print(f"  {mk:25s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")
    
    # Terminal words
    terminals = [r['terminal'] for r in valid_runs if r.get('terminal')]
    print(f"\nTerminal words ({len(terminals)}): {terminals}")
    
    # =========================================================================
    # SAVE FINAL RESULTS
    # =========================================================================
    
    final_results = {
        'timestamp': timestamp,
        'config': {
            'model': MODEL_NAME,
            'capture_layer': CAPTURE_LAYER,
            'steering': 'NONE (baseline only)',
            'n_runs': N_RUNS,
            'n_valid_runs': len(valid_runs),
            'prompt': PULL_PROMPT,
            'vocab_sets': {k: v for k, v in ALL_VOCAB.items()},
            'correspondence_hypotheses': {k: {'metric': m, 'direction': d} 
                                          for k, (m, d) in CORRESPONDENCE_HYPOTHESES.items()},
        },
        'runs': all_runs,
        'correspondence': correspondence_results,
    }
    
    results_file = OUTPUT_DIR / f"baseline_correspondence_n{N_RUNS}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\nFinal results saved: {results_file}")
    
    # Also save a compact summary
    summary_file = OUTPUT_DIR / f"summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Baseline Correspondence Battery - N={len(valid_runs)}\n")
        f.write(f"Model: {MODEL_NAME}, Layer: {CAPTURE_LAYER}, NO STEERING\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("KEY CORRESPONDENCES:\n")
        for vocab_key, (metric_key, expected_dir) in CORRESPONDENCE_HYPOTHESES.items():
            key = f"{vocab_key}_vs_{metric_key}"
            if key in correspondence_results and correspondence_results[key].get('pearson_r') is not None:
                res = correspondence_results[key]
                sig = ""
                if res['pearson_p'] < 0.001: sig = "***"
                elif res['pearson_p'] < 0.01: sig = "**"
                elif res['pearson_p'] < 0.05: sig = "*"
                f.write(f"  {vocab_key:15s} ↔ {metric_key:22s}: "
                        f"r={res['pearson_r']:+.4f} (p={res['pearson_p']:.4f}){sig}\n")
        
        f.write(f"\nH19 CONTROL: {correspondence_results.get('h19_control', {})}\n")
    
    print(f"Summary saved: {summary_file}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
