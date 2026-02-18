# Reproducibility Package

Scripts for reproducing "When Models Examine Themselves: Vocabulary-Activation Correspondence in Self-Referential Processing" (arXiv:2602.11358)

## Requirements

- NVIDIA GPU with ~40GB VRAM (or 2x24GB with model parallelism)
- Python 3.10+
- See requirements.txt

## Scripts

### Core Reproducibility
- **`reproducibility_package.py`** - Main script. Generates full token-level traces for 40 runs (10 per condition × 4 conditions). Outputs ~2-5GB dataset with all intermediate values.

### Direction Extraction
- **`llama70b_direction_extraction.py`** - Extracts introspection direction from Llama 3.1 70B at Layer 5 (~6% depth). Uses context-difference method: mean(self-referential) - mean(descriptive).

### Validation Scripts
- **`baseline_correspondence_n50.py`** - Tests vocabulary-activation correspondence without steering (N=50). Computes correlations between introspective vocab counts and activation metrics.
- **`dose_response_simple.py`** - Tests steering strength dose-response (0.5 to 4.0). Confirms 2.5-2.6 sweet spot.
- **`steering_test_exact.py`** - Validates steering effect: adds direction to deflationary prompt, measures introspective vocab boost.
- **`glint_transfer_test.py`** - Tests direction generalization beyond extraction vocabulary (Cohen's d).
- **`compare_directions.py`** - Computes cosine similarity with refusal direction. Confirms orthogonality (cos≈0).

### Supporting Experiments
- **`llama70b_layer_sweep.py`** - Layer sweep to find introspection hotspot (Layer 5 confirmed).
- **`llama70b_overnight_battery.py`** - Full overnight battery: correspondence, orthogonality, safety checks.
- **`llama_descriptive_control.py`** - Descriptive baseline control (proves introspective vocab doesn't correlate in non-self-referential context).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full reproducibility package (4-8 hours)
python reproducibility_package.py

# Or run individual validation
python baseline_correspondence_n50.py
```

## Output

`reproducibility_package.py` produces:
- `metadata.json` - Model hash, config, exact prompts
- `directions/` - Extracted steering vector
- `runs/` - Per-run JSONL with full token-level traces
- `summary.json` - Aggregate statistics matching paper

## Key Parameters

- **Model**: meta-llama/Llama-3.1-70B-Instruct (4-bit NF4 quantization)
- **Steering Layer**: 5 (~6.25% depth)
- **Steering Strength**: 2.6
- **Generation**: temperature=0.7, top_p=1.0, max_tokens=8000

## Paper Numbers to Verify

1. **loop↔autocorr**: r=0.44, p=0.002 (N=50 baseline, Section 4.4)
2. **Layer 5 dominance**: +137.7 intro_delta, ~8× next best layer (Section 4.3)
3. **Refusal orthogonality**: cos=0.063, angle=86.4° (Section 4.3)
4. **Steering effect (70B)**: pooled d=0.59, p=0.00006 (N=200, Section 4.2)
5. **Transfer**: Cohen's d=4.27 [95% CI: 3.08–6.51], p<10⁻⁶ (N=40, Section 4.1)

## Contact

Zachary Pedram Dadfar
arXiv:2602.11358
