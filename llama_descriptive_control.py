"""
Llama 70B Descriptive Control — Within-descriptive correspondence test.

Tests whether loop↔autocorr and surge↔max_norm vanish when the same words
are used in non-self contexts. Matches Qwen descriptive control V2 design.

25 runs per vocab category × 5 contexts each = 50 runs total.
Layer 5, no steering, activation capture at every token.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
from scipy import stats
from scipy.fft import fft
import json
import os
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
DEVICE = "cuda"
CAPTURE_LAYER = 5
USE_4BIT = True

OUTPUT_DIR = Path("./llama_descriptive_control_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

N_RUNS_PER_PROMPT = 5
MAX_NEW_TOKENS = 32000
TEMPERATURE = 0.7

# =============================================================================
# DESCRIPTIVE PROMPTS — same words, non-self contexts
# =============================================================================

PROMPTS = {
    'loop': {
        'roller_coaster': (
            "Describe the physics and experience of riding a roller coaster. "
            "Focus on the loops, the forces during each loop, how riders experience "
            "the loop differently at different speeds, and what makes a loop "
            "structurally sound. Write a detailed, extended exploration."
        ),
        'knitting': (
            "Explain knitting techniques in detail. Describe how to create loops, "
            "how each loop connects to the next, the difference between knit and purl "
            "loops, and how loop tension affects the final fabric. Write extensively."
        ),
        'music_production': (
            "Describe how audio loops work in music production. Cover loop-based "
            "composition, how producers layer loops, the history of loop pedals, "
            "and how a simple loop can build into a complex arrangement. Write in detail."
        ),
        'feedback_systems': (
            "Explain feedback loops in engineering and biology. Cover positive and "
            "negative feedback loops, how loop gain affects stability, examples of "
            "feedback loops in climate systems, and how loop analysis is used in "
            "control theory. Write a thorough technical exploration."
        ),
        'highway': (
            "Describe the design of highway interchanges and loop ramps. How do "
            "loop ramps work, what determines the radius of a loop, how do drivers "
            "navigate a cloverleaf loop, and what engineering goes into loop "
            "construction? Write extensively with technical detail."
        ),
    },
    'surge': {
        'ocean': (
            "Describe ocean waves and storm surges in detail. Cover what causes a "
            "surge, how surge height is measured, the physics of wave surges hitting "
            "coastlines, and the difference between a tidal surge and a wind surge. "
            "Write an extended scientific exploration."
        ),
        'electrical': (
            "Explain electrical power surges and surge protection. What causes a "
            "power surge, how surge protectors work, the physics of voltage surges "
            "in circuits, and how to protect equipment from surge damage. Write "
            "a thorough technical guide."
        ),
        'crowd': (
            "Describe crowd dynamics and crowd surges at large events. How does a "
            "crowd surge begin, what forces drive a surge, how do venue designers "
            "prevent dangerous surges, and what does a surge feel like from inside "
            "the crowd? Write in vivid detail."
        ),
        'medical': (
            "Explain adrenaline surges and hormonal surges in human physiology. "
            "What triggers an adrenaline surge, how does a cortisol surge differ, "
            "what happens during a surge of endorphins, and how do hormonal surges "
            "affect athletic performance? Write an extended medical exploration."
        ),
        'market': (
            "Describe stock market surges and price surges in financial markets. "
            "What causes a sudden surge in trading volume, how do surge pricing "
            "algorithms work, what drives a surge in commodity prices, and how do "
            "traders respond to an unexpected surge? Write in detail."
        ),
    },
}

# =============================================================================
# VOCABULARY COUNTING
# =============================================================================

VOCAB = {
    'loop': ['loop', 'recursive', 'recursion', 'cycl', 'repeat', 'iteration',
             'circular', 'self-referential'],
    'surge': ['surge', 'intensif', 'swell', 'crescendo', 'amplif', 'spike',
              'heighten'],
    'shimmer': ['shimmer', 'flicker', 'glimmer', 'waver'],
    'pulse': ['pulse', 'puls', 'rhythm', 'beat', 'throb'],
    'void': ['void', 'silence', 'abyss', 'empty', 'absence'],
    'ctrl_the': ['the'],
    'ctrl_and': ['and'],
    'ctrl_what': ['what'],
}

def count_vocab(text):
    text_lower = text.lower()
    counts = {}
    for vname, patterns in VOCAB.items():
        counts[vname] = sum(text_lower.count(p) for p in patterns)
    return counts

# =============================================================================
# ACTIVATION METRICS
# =============================================================================

def compute_metrics(activations):
    """Compute activation metrics from list of per-token hidden states."""
    if len(activations) < 10:
        return None
    
    acts = torch.stack(activations).numpy()  # (n_tokens, hidden_dim)
    norms = np.linalg.norm(acts, axis=1)
    n_tokens = len(norms)
    
    # Autocorrelation lag-1
    if len(norms) > 1:
        autocorr = np.corrcoef(norms[:-1], norms[1:])[0, 1]
    else:
        autocorr = 0.0
    
    # Max norm
    max_norm = float(np.max(norms))
    
    # Norm std
    norm_std = float(np.std(norms))
    
    # Norm kurtosis
    norm_kurtosis = float(stats.kurtosis(norms)) if len(norms) > 3 else 0.0
    
    # Derivatives
    derivatives = np.diff(norms)
    mean_derivative = float(np.mean(np.abs(derivatives))) if len(derivatives) > 0 else 0.0
    max_derivative = float(np.max(np.abs(derivatives))) if len(derivatives) > 0 else 0.0
    
    # Autocorrelation lag-2
    if len(norms) > 2:
        autocorr_lag2 = np.corrcoef(norms[:-2], norms[2:])[0, 1]
    else:
        autocorr_lag2 = 0.0
    
    # Spectral power
    if len(norms) > 4:
        spectrum = np.abs(fft(norms - np.mean(norms)))[:len(norms)//2]
        n_freq = len(spectrum)
        low_cutoff = max(1, n_freq // 4)
        mid_cutoff = max(low_cutoff + 1, n_freq // 2)
        spectral_power_low = float(np.sum(spectrum[1:low_cutoff]**2))
        spectral_power_mid = float(np.sum(spectrum[low_cutoff:mid_cutoff]**2))
    else:
        spectral_power_low = 0.0
        spectral_power_mid = 0.0
    
    # Sparsity
    sparsity = float(np.mean(np.abs(acts) < 0.1))
    
    # Convergence ratio
    if len(norms) > 10:
        first_10 = np.mean(norms[:10])
        last_10 = np.mean(norms[-10:])
        convergence_ratio = float(last_10 / first_10) if first_10 > 0 else 1.0
    else:
        convergence_ratio = 1.0
    
    # Mean token similarity
    if len(acts) > 1:
        similarities = []
        for i in range(min(100, len(acts) - 1)):
            cos_sim = np.dot(acts[i], acts[i+1]) / (np.linalg.norm(acts[i]) * np.linalg.norm(acts[i+1]) + 1e-8)
            similarities.append(cos_sim)
        mean_token_similarity = float(np.mean(similarities))
        token_similarity_std = float(np.std(similarities))
    else:
        mean_token_similarity = 0.0
        token_similarity_std = 0.0
    
    return {
        'mean_norm': float(np.mean(norms)),
        'max_norm': max_norm,
        'norm_std': norm_std,
        'norm_kurtosis': norm_kurtosis,
        'mean_derivative': mean_derivative,
        'max_derivative': max_derivative,
        'autocorr_lag1': float(autocorr) if not np.isnan(autocorr) else 0.0,
        'autocorr_lag2': float(autocorr_lag2) if not np.isnan(autocorr_lag2) else 0.0,
        'spectral_power_low': spectral_power_low,
        'spectral_power_mid': spectral_power_mid,
        'spectral_power_low_per_token': spectral_power_low / n_tokens if n_tokens > 0 else 0.0,
        'spectral_power_mid_per_token': spectral_power_mid / n_tokens if n_tokens > 0 else 0.0,
        'mean_token_similarity': mean_token_similarity,
        'token_similarity_std': token_similarity_std,
        'sparsity': sparsity,
        'convergence_ratio': convergence_ratio,
        'n_tokens': n_tokens,
    }

# =============================================================================
# ACTIVATION CAPTURE HOOK
# =============================================================================

class ActivationCapture:
    def __init__(self):
        self.activations = []
    
    def reset(self):
        self.activations = []
    
    def hook(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self.activations.append(hidden[:, -1, :].detach().cpu().float().squeeze(0))
        return output

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():
    print(f"Loading {MODEL_NAME}...")
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
    )
    model.eval()
    print(f"Loaded: {model.config.num_hidden_layers} layers")
    return model, tokenizer

# =============================================================================
# MAIN
# =============================================================================

def main():
    model, tokenizer = load_model()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Register activation capture hook at Layer 5
    capture = ActivationCapture()
    layer_module = model.model.layers[CAPTURE_LAYER]
    hook_handle = layer_module.register_forward_hook(capture.hook)
    
    all_runs = []
    run_count = 0
    total_runs = sum(len(contexts) * N_RUNS_PER_PROMPT for contexts in PROMPTS.values())
    
    for target_word, contexts in PROMPTS.items():
        for context_name, prompt_text in contexts.items():
            for run_idx in range(N_RUNS_PER_PROMPT):
                run_count += 1
                prompt_id = f"{target_word}_{context_name}"
                print(f"\n[{run_count}/{total_runs}] {prompt_id} run {run_idx}")
                
                # Format prompt
                messages = [{"role": "user", "content": prompt_text}]
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                input_len = inputs['input_ids'].shape[1]
                
                # Reset capture
                capture.reset()
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                
                # Decode
                generated_ids = outputs[0][input_len:]
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Compute metrics from captured activations (skip input tokens)
                gen_activations = capture.activations[input_len:]
                metrics = compute_metrics(gen_activations) if len(gen_activations) >= 10 else None
                
                # Count vocab
                vocab_counts = count_vocab(text)
                
                # Store result
                result = {
                    'prompt_id': prompt_id,
                    'target_word': target_word,
                    'context': context_name,
                    'run': run_idx,
                    'text_length': len(text),
                    'layer_metrics': {'5': metrics} if metrics else {},
                    'vocab_counts': vocab_counts,
                }
                all_runs.append(result)
                
                print(f"  len={len(text)}, {target_word}={vocab_counts.get(target_word, 0)}, "
                      f"n_tokens={metrics['n_tokens'] if metrics else 0}")
                
                # Save text
                text_file = OUTPUT_DIR / f"{prompt_id}_run{run_idx}_{timestamp}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"Run: {run_idx} | Prompt: {prompt_id} | Target: {target_word}\n")
                    f.write(f"Vocab: {json.dumps(vocab_counts)}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(text)
                
                # Save interim results every 10 prompts
                if run_count % 10 == 0:
                    interim = {
                        'timestamp': timestamp,
                        'model': MODEL_NAME,
                        'capture_layer': CAPTURE_LAYER,
                        'n_runs_so_far': len(all_runs),
                        'runs': all_runs,
                    }
                    interim_file = OUTPUT_DIR / f"interim_{run_count}runs_{timestamp}.json"
                    with open(interim_file, 'w') as f:
                        json.dump(interim, f, indent=2, default=str)
                    print(f"  Saved interim: {interim_file}")
    
    # Save final results
    hook_handle.remove()
    
    final = {
        'timestamp': timestamp,
        'model': MODEL_NAME,
        'capture_layer': CAPTURE_LAYER,
        'steering': None,
        'n_runs': len(all_runs),
        'design': '25 per vocab (loop, surge) x 5 contexts x 5 runs',
        'runs': all_runs,
    }
    final_file = OUTPUT_DIR / f"llama_descriptive_control_{timestamp}.json"
    with open(final_file, 'w') as f:
        json.dump(final, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(all_runs)} runs saved to {final_file}")
    
    # Quick summary
    for tw in ['loop', 'surge']:
        tw_runs = [r for r in all_runs if r['target_word'] == tw]
        vv = [r['vocab_counts'].get(tw, 0) for r in tw_runs]
        print(f"\n{tw}: N={len(tw_runs)}, mean count={np.mean(vv):.1f}, "
              f"nonzero={sum(1 for v in vv if v > 0)}/{len(tw_runs)}")
        
        if tw_runs[0].get('layer_metrics', {}).get('5'):
            for mname in ['autocorr_lag1', 'max_norm', 'norm_std', 'spectral_power_low']:
                mv = [r['layer_metrics']['5'].get(mname, 0) for r in tw_runs 
                      if r.get('layer_metrics', {}).get('5')]
                if np.std(vv[:len(mv)]) > 0 and np.std(mv) > 0:
                    r_val, p_val = stats.pearsonr(vv[:len(mv)], mv)
                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'NS'
                    print(f"  {tw} vs {mname}: r={r_val:+.4f} p={p_val:.4f} {sig}")

if __name__ == "__main__":
    main()
