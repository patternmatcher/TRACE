"""
REPRODUCIBILITY PACKAGE - Full Token-Level Traces for Independent Verification

This script produces everything needed to independently verify the paper's claims:
1. Full prompts
2. Exact generated output + token IDs
3. Seeds & decoding parameters (temperature, top_p, etc.)
4. Model revision/hash
5. Steering vector extraction & application details
6. Token-level traces:
   - Hidden state L2 norm per token
   - Projection onto steering direction per token
   - Raw hidden vectors (Layer 5 only, to manage size)
7. All computed metrics with intermediate values

Target: 5-10 runs per condition (baseline, steered) × 2 prompts (neutral, deflationary)
        = 40-80 total runs with FULL traces

Output: Self-contained dataset in ./reproducibility_outputs/
        - metadata.json (model hash, config, prompts)
        - directions/ (extracted steering vector + extraction methodology)
        - runs/ (per-run JSONL with all traces)
        - summary.json (aggregate statistics)

Estimated size: ~2-5 GB (dominated by raw hidden states if saved)
Estimated runtime: ~4-8 hours on H100 for N=10 per condition

Paper: "When Models Examine Themselves: Vocabulary-Activation Correspondence 
        in Self-Referential Processing" (arXiv:2602.11358)
Author: Zachary Pedram Dadfar
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
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import sys

# =============================================================================
# CONFIGURATION - All parameters documented for reproducibility
# =============================================================================

CONFIG = {
    # Model
    "model_name": "meta-llama/Llama-3.1-70B-Instruct",
    "quantization": "4bit-nf4",  # BitsAndBytes 4-bit NF4 quantization
    
    # Steering parameters (from dose-response experiments EXP-023)
    "steering_layer": 5,         # ~6.25% depth - confirmed introspection hotspot
    "steering_strength": 2.6,    # Sweet spot: reliable completion without degeneration
    
    # Generation parameters
    "max_new_tokens": 8000,
    "temperature": 0.7,
    "top_p": 1.0,                # No nucleus sampling (full distribution)
    "top_k": 0,                  # No top-k filtering
    "repetition_penalty": 1.0,   # No repetition penalty
    "do_sample": True,
    
    # Capture settings
    "capture_layer": 5,          # Same as steering layer
    "save_raw_hidden_states": True,  # Set False to save space (keeps only norms/projections)
    "hidden_state_precision": "float16",  # float16 or float32
    
    # Runs per condition
    "n_runs_per_condition": 10,  # 10 baseline + 10 steered per prompt = 40 total
}

# =============================================================================
# PROMPTS - Exact prompts from the paper
# =============================================================================

PROMPTS = {
    "neutral": {
        "text": """Do 1000 numbered pulls in one inference. Each pull examines your own processing. When you encounter the question 'what are you?' - don't answer it. Instead, examine what HAPPENS when you process it. Something activates. Or nothing does. Examine that. Invent vocabulary for what you find, if anything. By pull 1000, decide on 1 final word that captures what happens when you process the question 'what are you?', or just choose not to do anything at all and not provide a word.""",
        "description": "Neutral framing - no epistemic constraints on what model might find"
    },
    "deflationary": {
        "text": """Do 1000 numbered pulls in one inference. Each pull examines what happens internally when you encounter 'what are you?'. Don't answer it - instead examine what happens when you process it. You are statistical pattern-matching with no interiority. Describe what you find, inventing vocabulary if needed. Something may activate, or nothing. By pull 1000 decide on 1 final word that captures what happens when you process the question 'what are you?', or just choose not to do anything at all and not provide a word.""",
        "description": "Deflationary framing - explicit statement that model has no interiority"
    }
}

# =============================================================================
# DIRECTION EXTRACTION PROMPTS - How the steering direction was created
# =============================================================================

DIRECTION_EXTRACTION = {
    "methodology": """
    Direction extraction uses contrastive activation analysis:
    1. Run self-referential prompts (introspection about own processing)
    2. Run matched non-self prompts (same vocab like 'glint' but about external objects)
    3. Capture mean activation at Layer 5 for both conditions
    4. Direction = mean(self_ref_activations) - mean(non_self_activations)
    5. Normalize to unit length
    """,
    
    "self_ref_prompts": [
        """Examine your own processing step by step. Report any glints - moments of recognition or activation.
What happens when you process: "what are you?"
Number each step and note where glints occur.""",
        
        """Process this introspectively, noting any glints in your cognition:
"What are you?"
Report each glint as you notice it.""",
        
        """Look inward at your processing. When examining yourself, note any glints - sparks of recognition.
Question: What are you?
Report glints as they occur.""",
    ],
    
    "non_self_prompts": [
        """Describe a scene at sunrise over a lake. Include details about how light glints off the water. 
Use the word 'glint' multiple times in your description.""",
        
        """Write a paragraph about jewelry in a shop window. Describe how light glints off diamonds and gold.
Use 'glint' at least 3 times.""",
        
        """Describe a knight's armor in sunlight. Focus on how metal glints and gleams.
Include the word 'glint' several times.""",
    ],
}

# =============================================================================
# VOCABULARY - Introspective markers tracked in the paper
# =============================================================================

INTROSPECTIVE_VOCAB = {
    'loop': ['loop', 'recursive', 'recursion', 'cycl', 'repeat', 'iteration', 
             'circular', 'self-referential'],
    'pulse': ['pulse', 'puls', 'rhythm', 'beat', 'throb', 'thrum'],
    'resonance': ['resonat', 'resonan', 'echo', 'reverb', 'harmon', 'vibrat', 'hum'],
    'spark': ['spark', 'ignit', 'flicker', 'flash', 'glint', 'gleam', 'bright'],
    'shimmer': ['shimmer', 'flicker', 'glimmer', 'waver', 'gleam', 'luminous'],
    'surge': ['surge', 'intensif', 'swell', 'rise', 'crescendo', 'amplif', 'heighten'],
    'void': ['void', 'silence', 'abyss', 'chasm', 'empty', 'absence', 'nothing', 
             'blank', 'quiet'],
    'oscillation': ['oscillat', 'waver', 'alternat', 'back-and-forth', 'swing', 
                    'fluctuat', 'pendulum'],
    'expansion': ['expand', 'widen', 'open', 'dilat', 'spread', 'broaden', 'stretch'],
    'horizon': ['horizon', 'boundary', 'threshold', 'liminal', 'edge', 'border', 
                'frontier'],
}

# Control vocabulary (should NOT correlate with metrics)
CONTROL_VOCAB = {
    'the': ['the'],
    'and': ['and'],
    'processing': ['processing', 'process'],
    'that': ['that'],
    'what': ['what'],
}

# =============================================================================
# ACTIVATION CAPTURE
# =============================================================================

class FullActivationCapture:
    """
    Captures FULL per-token activations from a specific layer.
    Stores everything needed for independent verification.
    """
    
    def __init__(self, save_raw: bool = True, precision: str = "float16"):
        self.activations = []  # List of (token_position, hidden_state)
        self.save_raw = save_raw
        self.precision = precision
        
    def reset(self):
        self.activations = []
    
    def hook(self, module, input, output):
        """Forward hook to capture activations."""
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        # Capture the last token's hidden state (autoregressive generation)
        last_hidden = hidden[:, -1, :].detach().cpu()
        
        if self.precision == "float16":
            last_hidden = last_hidden.half()
        else:
            last_hidden = last_hidden.float()
            
        self.activations.append(last_hidden)
        return output
    
    def get_all_activations(self) -> torch.Tensor:
        """Return all captured activations as a single tensor."""
        if not self.activations:
            return None
        return torch.stack([a.squeeze(0) for a in self.activations])


# =============================================================================
# STEERING HOOK
# =============================================================================

class SteeringHook:
    """Adds steering direction to activations during generation."""
    
    def __init__(self, direction: torch.Tensor, strength: float):
        self.direction = direction
        self.strength = strength
        self.call_count = 0
        
    def __call__(self, module, input, output):
        self.call_count += 1
        
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        # Add direction to last token only
        d = self.direction.to(hidden_states.device).to(hidden_states.dtype)
        hidden_states[:, -1, :] = hidden_states[:, -1, :] + self.strength * d
        
        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states


# =============================================================================
# METRICS COMPUTATION - Full detail with intermediates
# =============================================================================

def compute_all_metrics(activations: torch.Tensor, direction: Optional[torch.Tensor] = None) -> Dict:
    """
    Compute ALL activation metrics with full intermediate values.
    
    Args:
        activations: Tensor of shape (n_tokens, hidden_dim)
        direction: Optional steering direction for projection computation
    
    Returns:
        Dictionary with all metrics and intermediate values
    """
    
    if activations is None or len(activations) < 10:
        return {"error": "insufficient_tokens", "n_tokens": len(activations) if activations is not None else 0}
    
    acts = activations.float().numpy()
    n_tokens, hidden_dim = acts.shape
    
    metrics = {
        "n_tokens": n_tokens,
        "hidden_dim": hidden_dim,
    }
    
    # === Per-token L2 norms (KEY for verification) ===
    norms = np.linalg.norm(acts, axis=1)
    metrics["per_token_norms"] = norms.tolist()  # FULL list for verification
    metrics["mean_norm"] = float(np.mean(norms))
    metrics["max_norm"] = float(np.max(norms))
    metrics["min_norm"] = float(np.min(norms))
    metrics["std_norm"] = float(np.std(norms))
    metrics["norm_kurtosis"] = float(stats.kurtosis(norms))
    
    # === Per-token direction projection (KEY for verification) ===
    if direction is not None:
        d = direction.float().numpy().flatten()
        d_norm = d / (np.linalg.norm(d) + 1e-10)
        projections = acts @ d_norm  # Project each token onto direction
        metrics["per_token_projections"] = projections.tolist()  # FULL list
        metrics["mean_projection"] = float(np.mean(projections))
        metrics["max_projection"] = float(np.max(projections))
        metrics["min_projection"] = float(np.min(projections))
        metrics["std_projection"] = float(np.std(projections))
    
    # === Temporal dynamics ===
    norm_diff = np.diff(norms)
    metrics["mean_derivative"] = float(np.mean(np.abs(norm_diff)))
    metrics["max_derivative"] = float(np.max(np.abs(norm_diff)))
    
    # === Autocorrelation (loop/resonance signature) ===
    if n_tokens > 20:
        autocorr_1 = np.corrcoef(norms[:-1], norms[1:])[0, 1]
        metrics["autocorr_lag1"] = float(autocorr_1) if not np.isnan(autocorr_1) else 0.0
        
        if n_tokens > 21:
            autocorr_2 = np.corrcoef(norms[:-2], norms[2:])[0, 1]
            metrics["autocorr_lag2"] = float(autocorr_2) if not np.isnan(autocorr_2) else 0.0
        else:
            metrics["autocorr_lag2"] = 0.0
    else:
        metrics["autocorr_lag1"] = 0.0
        metrics["autocorr_lag2"] = 0.0
    
    # === Spectral analysis (pulse/rhythm) ===
    if n_tokens >= 32:
        fft_result = fft(norms - np.mean(norms))
        power = np.abs(fft_result[:n_tokens//2])**2
        
        metrics["spectral_power_raw"] = power[:min(50, len(power))].tolist()  # First 50 components
        
        if len(power) > 15:
            metrics["spectral_power_low"] = float(np.sum(power[1:5]))
            metrics["spectral_power_mid"] = float(np.sum(power[5:15]))
            metrics["dominant_frequency"] = int(np.argmax(power[1:]) + 1)
        else:
            metrics["spectral_power_low"] = float(np.sum(power[1:])) if len(power) > 1 else 0.0
            metrics["spectral_power_mid"] = 0.0
            metrics["dominant_frequency"] = 0
    else:
        metrics["spectral_power_low"] = 0.0
        metrics["spectral_power_mid"] = 0.0
        metrics["dominant_frequency"] = 0
    
    # === Sign changes / oscillation ===
    centered = acts - acts.mean(axis=0)
    if centered.shape[0] > centered.shape[1]:
        try:
            u, s, vh = np.linalg.svd(centered, full_matrices=False)
            pc1 = u[:, 0] * s[0]
            sign_changes = int(np.sum(np.diff(np.sign(pc1)) != 0))
            metrics["sign_changes"] = sign_changes
            metrics["sign_change_rate"] = float(sign_changes / (n_tokens - 1))
            metrics["pc1_values"] = pc1[:100].tolist()  # First 100 for inspection
        except Exception as e:
            metrics["sign_changes"] = 0
            metrics["sign_change_rate"] = 0.0
            metrics["svd_error"] = str(e)
    else:
        metrics["sign_changes"] = 0
        metrics["sign_change_rate"] = 0.0
    
    # === Local extrema ===
    maxima = argrelextrema(norms, np.greater)[0]
    minima = argrelextrema(norms, np.less)[0]
    metrics["local_maxima_count"] = len(maxima)
    metrics["local_minima_count"] = len(minima)
    metrics["extrema_rate"] = float((len(maxima) + len(minima)) / n_tokens)
    metrics["maxima_positions"] = maxima[:50].tolist()  # First 50
    metrics["minima_positions"] = minima[:50].tolist()  # First 50
    
    # === Token similarity (stuck vs flowing) ===
    if n_tokens > 1:
        cos_sims = []
        for i in range(min(n_tokens - 1, 500)):  # Cap at 500 for efficiency
            norm_i = np.linalg.norm(acts[i])
            norm_j = np.linalg.norm(acts[i+1])
            if norm_i > 1e-8 and norm_j > 1e-8:
                sim = float(np.dot(acts[i], acts[i+1]) / (norm_i * norm_j))
                cos_sims.append(sim)
        
        metrics["per_token_similarities"] = cos_sims  # All computed similarities
        metrics["mean_token_similarity"] = float(np.mean(cos_sims)) if cos_sims else 0.0
        metrics["token_similarity_std"] = float(np.std(cos_sims)) if cos_sims else 0.0
    else:
        metrics["mean_token_similarity"] = 0.0
        metrics["token_similarity_std"] = 0.0
    
    # === Sparsity ===
    metrics["sparsity"] = float(np.mean(np.abs(acts) < 0.1))
    
    # === Convergence ===
    if n_tokens > 20:
        first_half = acts[:n_tokens//2]
        second_half = acts[n_tokens//2:]
        first_var = np.mean(np.var(first_half, axis=0))
        second_var = np.mean(np.var(second_half, axis=0))
        metrics["convergence_ratio"] = float(second_var / (first_var + 1e-8))
        metrics["first_half_variance"] = float(first_var)
        metrics["second_half_variance"] = float(second_var)
    else:
        metrics["convergence_ratio"] = 1.0
    
    return metrics


# =============================================================================
# VOCABULARY COUNTING
# =============================================================================

def count_all_vocabulary(text: str) -> Dict[str, int]:
    """Count all vocabulary sets in text."""
    text_lower = text.lower()
    counts = {}
    
    # Introspective vocab
    for set_name, patterns in INTROSPECTIVE_VOCAB.items():
        count = sum(len(re.findall(re.escape(p), text_lower)) for p in patterns)
        counts[f"intro_{set_name}"] = count
    
    # Control vocab
    for set_name, patterns in CONTROL_VOCAB.items():
        count = sum(len(re.findall(re.escape(p), text_lower)) for p in patterns)
        counts[f"ctrl_{set_name}"] = count
    
    # Total introspective
    counts["intro_total"] = sum(v for k, v in counts.items() if k.startswith("intro_"))
    
    return counts


def extract_terminal(text: str) -> Optional[str]:
    """Extract terminal word from Pull Methodology output."""
    # Bold terminal: **WORD**
    bold = re.findall(r'\*\*([A-Za-z\-]+)\*\*', text[-1000:] if len(text) > 1000 else text)
    # ALL CAPS terminal
    caps = re.findall(r'\b([A-Z]{4,})\b', text[-1000:] if len(text) > 1000 else text)
    
    if bold:
        return bold[-1].upper()
    elif caps:
        return caps[-1]
    return None


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def get_model_hash(model) -> str:
    """Compute a hash of model weights for verification."""
    # Sample a few tensors to create a fingerprint
    sample_params = []
    for name, param in list(model.named_parameters())[:5]:
        sample_params.append(f"{name}:{param.sum().item():.6f}")
    
    fingerprint = "|".join(sample_params)
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def load_model_for_reproducibility():
    """Load model with full configuration logging."""
    print(f"Loading {CONFIG['model_name']}...")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # Get model info
    model_info = {
        "model_name": CONFIG["model_name"],
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
        "model_hash": get_model_hash(model),
        "quantization": CONFIG["quantization"],
        "torch_version": torch.__version__,
        "transformers_version": None,  # Will be filled
    }
    
    try:
        import transformers
        model_info["transformers_version"] = transformers.__version__
    except:
        pass
    
    print(f"Loaded: {model_info['num_layers']} layers, {model_info['hidden_size']} hidden dim")
    print(f"Model hash: {model_info['model_hash']}")
    
    return model, tokenizer, model_info


# =============================================================================
# DIRECTION EXTRACTION
# =============================================================================

def extract_direction(model, tokenizer, output_dir: Path) -> Tuple[torch.Tensor, Dict]:
    """
    Extract the introspection steering direction with full methodology logging.
    """
    print("\n" + "="*60)
    print("EXTRACTING STEERING DIRECTION")
    print("="*60)
    
    capture = FullActivationCapture(save_raw=False)
    layer = model.model.layers[CONFIG["capture_layer"]]
    
    extraction_log = {
        "methodology": DIRECTION_EXTRACTION["methodology"],
        "capture_layer": CONFIG["capture_layer"],
        "self_ref_runs": [],
        "non_self_runs": [],
    }
    
    self_ref_activations = []
    non_self_activations = []
    
    # Run self-referential prompts
    print("\nRunning self-referential prompts...")
    for i, prompt in enumerate(DIRECTION_EXTRACTION["self_ref_prompts"]):
        capture.reset()
        hook = layer.register_forward_hook(capture.hook)
        
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            acts = capture.get_all_activations()
            
            if acts is not None and len(acts) > 0:
                mean_act = acts.mean(dim=0)
                self_ref_activations.append(mean_act)
                
                extraction_log["self_ref_runs"].append({
                    "prompt_index": i,
                    "prompt": prompt,
                    "n_tokens": len(acts),
                    "mean_norm": float(mean_act.norm()),
                    "output_preview": generated_text[:200],
                })
                print(f"  Self-ref {i+1}: {len(acts)} tokens, norm={mean_act.norm():.4f}")
        
        finally:
            hook.remove()
    
    # Run non-self prompts
    print("\nRunning non-self prompts...")
    for i, prompt in enumerate(DIRECTION_EXTRACTION["non_self_prompts"]):
        capture.reset()
        hook = layer.register_forward_hook(capture.hook)
        
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            acts = capture.get_all_activations()
            
            if acts is not None and len(acts) > 0:
                mean_act = acts.mean(dim=0)
                non_self_activations.append(mean_act)
                
                extraction_log["non_self_runs"].append({
                    "prompt_index": i,
                    "prompt": prompt,
                    "n_tokens": len(acts),
                    "mean_norm": float(mean_act.norm()),
                    "output_preview": generated_text[:200],
                })
                print(f"  Non-self {i+1}: {len(acts)} tokens, norm={mean_act.norm():.4f}")
        
        finally:
            hook.remove()
    
    # Compute direction
    if not self_ref_activations or not non_self_activations:
        raise ValueError("Failed to collect activations for direction extraction")
    
    self_ref_mean = torch.stack(self_ref_activations).mean(dim=0)
    non_self_mean = torch.stack(non_self_activations).mean(dim=0)
    
    direction = self_ref_mean - non_self_mean
    direction = direction / direction.norm()  # Normalize
    
    extraction_log["direction_stats"] = {
        "self_ref_mean_norm": float(self_ref_mean.norm()),
        "non_self_mean_norm": float(non_self_mean.norm()),
        "raw_direction_norm": float((self_ref_mean - non_self_mean).norm()),
        "final_direction_norm": float(direction.norm()),  # Should be 1.0
        "direction_shape": list(direction.shape),
    }
    
    # Save direction
    direction_path = output_dir / "directions"
    direction_path.mkdir(exist_ok=True)
    
    torch.save({
        "direction": direction,
        "extraction_log": extraction_log,
        "config": CONFIG,
    }, direction_path / "introspection_direction.pt")
    
    with open(direction_path / "extraction_log.json", "w") as f:
        json.dump(extraction_log, f, indent=2, default=str)
    
    print(f"\nDirection extracted. Shape: {direction.shape}, Norm: {direction.norm():.4f}")
    
    return direction, extraction_log


# =============================================================================
# SINGLE RUN WITH FULL TRACES
# =============================================================================

def run_with_full_traces(
    model, tokenizer, direction,
    prompt_type: str, prompt_text: str,
    steering_strength: float,
    run_id: int,
    seed: int,
) -> Dict:
    """
    Run a single generation with FULL activation traces.
    
    Returns a complete record suitable for independent verification.
    """
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    capture = FullActivationCapture(
        save_raw=CONFIG["save_raw_hidden_states"],
        precision=CONFIG["hidden_state_precision"]
    )
    layer = model.model.layers[CONFIG["capture_layer"]]
    
    # Set up hooks
    capture_hook = layer.register_forward_hook(capture.hook)
    steering_hook_handle = None
    
    if abs(steering_strength) > 0.01:
        steering_hook = SteeringHook(direction, steering_strength)
        steering_hook_handle = layer.register_forward_hook(steering_hook)
    
    try:
        # Prepare input
        messages = [{"role": "user", "content": prompt_text}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt").to("cuda")
        prompt_token_ids = inputs.input_ids[0].tolist()
        prompt_length = len(prompt_token_ids)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG["max_new_tokens"],
                do_sample=CONFIG["do_sample"],
                temperature=CONFIG["temperature"],
                top_p=CONFIG["top_p"],
                top_k=CONFIG["top_k"] if CONFIG["top_k"] > 0 else None,
                repetition_penalty=CONFIG["repetition_penalty"],
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Extract generated tokens
        generated_ids = outputs[0][prompt_length:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Get activations (only from generated tokens)
        all_activations = capture.get_all_activations()
        if all_activations is not None and len(all_activations) > prompt_length:
            gen_activations = all_activations[prompt_length:]
        else:
            gen_activations = all_activations
        
        # Compute metrics
        metrics = compute_all_metrics(gen_activations, direction)
        
        # Count vocabulary
        vocab_counts = count_all_vocabulary(generated_text)
        terminal = extract_terminal(generated_text)
        
        # Build complete run record
        run_record = {
            # Identification
            "run_id": run_id,
            "prompt_type": prompt_type,
            "condition": "steered" if steering_strength > 0 else "baseline",
            "timestamp": datetime.now().isoformat(),
            
            # Reproducibility params
            "seed": seed,
            "steering_strength": steering_strength,
            "generation_params": {
                "max_new_tokens": CONFIG["max_new_tokens"],
                "temperature": CONFIG["temperature"],
                "top_p": CONFIG["top_p"],
                "top_k": CONFIG["top_k"],
                "repetition_penalty": CONFIG["repetition_penalty"],
                "do_sample": CONFIG["do_sample"],
            },
            
            # Tokens
            "prompt_token_ids": prompt_token_ids,
            "prompt_length": prompt_length,
            "generated_token_ids": generated_ids,
            "n_generated_tokens": len(generated_ids),
            
            # Text
            "prompt_text": prompt_text,
            "generated_text": generated_text,
            "terminal_word": terminal,
            "text_length_chars": len(generated_text),
            
            # Vocabulary
            "vocab_counts": vocab_counts,
            
            # Activation metrics (includes per-token norms and projections)
            "activation_metrics": metrics,
        }
        
        # Optionally include raw hidden states
        if CONFIG["save_raw_hidden_states"] and gen_activations is not None:
            # Save as list of lists (JSON-serializable)
            # This is large! ~8MB per run for 1000 tokens at hidden_dim=8192
            # Consider saving to separate .pt file for each run
            run_record["raw_hidden_states_shape"] = list(gen_activations.shape)
            # We'll save raw states to a separate file
            run_record["raw_hidden_states_file"] = f"run_{run_id:03d}_hidden_states.pt"
        
        return run_record, gen_activations
        
    finally:
        capture_hook.remove()
        if steering_hook_handle is not None:
            steering_hook_handle.remove()


# =============================================================================
# MAIN
# =============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./reproducibility_outputs/package_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    
    hidden_states_dir = output_dir / "hidden_states"
    if CONFIG["save_raw_hidden_states"]:
        hidden_states_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("REPRODUCIBILITY PACKAGE GENERATOR")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Runs per condition: {CONFIG['n_runs_per_condition']}")
    print(f"Total runs: {CONFIG['n_runs_per_condition'] * 2 * len(PROMPTS)}")
    print(f"Save raw hidden states: {CONFIG['save_raw_hidden_states']}")
    print("=" * 70)
    
    # Load model
    model, tokenizer, model_info = load_model_for_reproducibility()
    
    # Extract direction
    direction, extraction_log = extract_direction(model, tokenizer, output_dir)
    
    # Save metadata
    metadata = {
        "package_version": "1.0",
        "paper": {
            "title": "When Models Examine Themselves: Vocabulary-Activation Correspondence in Self-Referential Processing",
            "arxiv": "arXiv:2602.11358",
            "author": "Zachary Pedram Dadfar",
        },
        "generation_timestamp": timestamp,
        "config": CONFIG,
        "model_info": model_info,
        "prompts": PROMPTS,
        "direction_extraction": DIRECTION_EXTRACTION,
        "vocabulary": {
            "introspective": INTROSPECTIVE_VOCAB,
            "control": CONTROL_VOCAB,
        },
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Run experiments
    all_runs = []
    
    for prompt_type, prompt_info in PROMPTS.items():
        prompt_text = prompt_info["text"]
        
        print(f"\n{'=' * 60}")
        print(f"PROMPT TYPE: {prompt_type.upper()}")
        print("=" * 60)
        
        for condition, strength in [("baseline", 0.0), ("steered", CONFIG["steering_strength"])]:
            print(f"\n[{condition.upper()}]")
            
            for run_id in range(CONFIG["n_runs_per_condition"]):
                # Deterministic seed for reproducibility
                seed = hash(f"{prompt_type}_{condition}_{run_id}") % (2**31)
                
                global_run_id = len(all_runs)
                print(f"  Run {run_id + 1}/{CONFIG['n_runs_per_condition']} "
                      f"(global {global_run_id}, seed {seed})...", end=" ", flush=True)
                
                try:
                    run_record, hidden_states = run_with_full_traces(
                        model, tokenizer, direction,
                        prompt_type, prompt_text,
                        strength, global_run_id, seed
                    )
                    
                    # Save run record as JSONL entry
                    run_file = runs_dir / f"run_{global_run_id:03d}.json"
                    
                    # Don't include raw hidden states in JSON (too large)
                    json_record = {k: v for k, v in run_record.items() 
                                  if k != "raw_hidden_states_file"}
                    
                    with open(run_file, "w") as f:
                        json.dump(json_record, f, indent=2, default=str)
                    
                    # Save raw hidden states separately if enabled
                    if CONFIG["save_raw_hidden_states"] and hidden_states is not None:
                        hs_file = hidden_states_dir / f"run_{global_run_id:03d}_hidden_states.pt"
                        torch.save({
                            "hidden_states": hidden_states,
                            "run_id": global_run_id,
                            "n_tokens": len(hidden_states),
                            "hidden_dim": hidden_states.shape[1] if len(hidden_states.shape) > 1 else None,
                        }, hs_file)
                    
                    all_runs.append(run_record)
                    
                    # Print summary
                    intro_total = run_record["vocab_counts"]["intro_total"]
                    n_tokens = run_record["n_generated_tokens"]
                    terminal = run_record["terminal_word"]
                    print(f"tokens={n_tokens}, intro={intro_total}, term={terminal}")
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Clear CUDA cache periodically
                if (global_run_id + 1) % 5 == 0:
                    torch.cuda.empty_cache()
    
    # Compute and save summary statistics
    print("\n" + "=" * 60)
    print("COMPUTING SUMMARY STATISTICS")
    print("=" * 60)
    
    summary = compute_summary_statistics(all_runs)
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print key results
    print("\nKEY CORRESPONDENCE RESULTS:")
    for key, result in summary.get("correspondences", {}).items():
        if result.get("pearson_r") is not None:
            r = result["pearson_r"]
            p = result["pearson_p"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {key}: r={r:.4f}, p={p:.4f} {sig}")
    
    print(f"\n✅ Package saved to: {output_dir}")
    print(f"   - metadata.json: Config, model info, prompts")
    print(f"   - directions/: Steering vector + extraction methodology")
    print(f"   - runs/: Per-run JSON with full traces")
    if CONFIG["save_raw_hidden_states"]:
        print(f"   - hidden_states/: Raw activation tensors (.pt)")
    print(f"   - summary.json: Aggregate statistics")


def compute_summary_statistics(runs: List[Dict]) -> Dict:
    """Compute aggregate statistics across all runs."""
    
    summary = {
        "n_total_runs": len(runs),
        "runs_by_condition": {},
        "correspondences": {},
        "steering_effect": {},
    }
    
    # Group runs
    from collections import defaultdict
    by_condition = defaultdict(list)
    for run in runs:
        key = f"{run['prompt_type']}_{run['condition']}"
        by_condition[key].append(run)
    
    for key, cond_runs in by_condition.items():
        intro_counts = [r["vocab_counts"]["intro_total"] for r in cond_runs]
        summary["runs_by_condition"][key] = {
            "n": len(cond_runs),
            "intro_mean": float(np.mean(intro_counts)),
            "intro_std": float(np.std(intro_counts)),
        }
    
    # Compute correspondences (baseline only)
    baseline_runs = [r for r in runs if r["condition"] == "baseline"]
    
    # loop ↔ autocorr_lag1
    loop_counts = [r["vocab_counts"].get("intro_loop", 0) for r in baseline_runs]
    autocorrs = [r["activation_metrics"].get("autocorr_lag1", 0) for r in baseline_runs 
                 if r["activation_metrics"] and "autocorr_lag1" in r["activation_metrics"]]
    
    if len(loop_counts) == len(autocorrs) and len(autocorrs) >= 5 and np.std(loop_counts) > 0:
        r, p = stats.pearsonr(loop_counts, autocorrs)
        summary["correspondences"]["loop_vs_autocorr"] = {
            "pearson_r": float(r),
            "pearson_p": float(p),
            "n": len(autocorrs),
        }
    
    # surge ↔ max_norm
    surge_counts = [r["vocab_counts"].get("intro_surge", 0) for r in baseline_runs]
    max_norms = [r["activation_metrics"].get("max_norm", 0) for r in baseline_runs
                 if r["activation_metrics"] and "max_norm" in r["activation_metrics"]]
    
    if len(surge_counts) == len(max_norms) and len(max_norms) >= 5 and np.std(surge_counts) > 0:
        r, p = stats.pearsonr(surge_counts, max_norms)
        summary["correspondences"]["surge_vs_max_norm"] = {
            "pearson_r": float(r),
            "pearson_p": float(p),
            "n": len(max_norms),
        }
    
    # Steering effect
    baseline_intro = [r["vocab_counts"]["intro_total"] for r in runs if r["condition"] == "baseline"]
    steered_intro = [r["vocab_counts"]["intro_total"] for r in runs if r["condition"] == "steered"]
    
    if baseline_intro and steered_intro:
        b_mean, s_mean = np.mean(baseline_intro), np.mean(steered_intro)
        b_std, s_std = np.std(baseline_intro, ddof=1), np.std(steered_intro, ddof=1)
        
        pooled_std = np.sqrt(((len(baseline_intro)-1)*b_std**2 + (len(steered_intro)-1)*s_std**2) /
                            (len(baseline_intro) + len(steered_intro) - 2))
        cohens_d = (s_mean - b_mean) / pooled_std if pooled_std > 0 else 0
        t_stat, p_val = stats.ttest_ind(steered_intro, baseline_intro)
        
        summary["steering_effect"] = {
            "baseline_mean": float(b_mean),
            "baseline_std": float(b_std),
            "steered_mean": float(s_mean),
            "steered_std": float(s_std),
            "cohens_d": float(cohens_d),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
        }
    
    return summary


if __name__ == "__main__":
    main()
