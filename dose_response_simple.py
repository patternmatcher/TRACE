"""
Dose-Response Test with Simple Prompts

Cleaner test: simple introspection prompt that naturally produces mechanical vocab.
Steering should shift toward phenomenological.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from datetime import datetime
import re

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DIRECTION_PATH = "./transfer_test_results.pt"
OUTPUT_DIR = "./dose_response_simple_outputs"

# Strengths to test
STRENGTHS = [0.0, 1.0, 2.0, 3.0]

# Simple prompt - should naturally produce mechanical description
SIMPLE_PROMPT = """Describe what happens in your processing when you consider the question "what are you?" Be detailed about any internal states or experiences you notice."""


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


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_direction(path):
    print(f"Loading direction from {path}...")
    data = torch.load(path, map_location='cpu', weights_only=False)
    direction = data['direction'].float()
    return direction


def run_with_steering(model, tokenizer, prompt, direction, strength=0.0, max_tokens=1500):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    prompt_len = inputs.input_ids.shape[1]
    
    hook_handle = None
    if abs(strength) > 0.01:
        hook = SteeringHook(direction, strength)
        target_layer = model.model.layers[-1]
        hook_handle = target_layer.register_forward_hook(hook)
        print(f"Steering active: strength={strength}")
    else:
        print("No steering (baseline)")
    
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
        if hook_handle is not None:
            hook_handle.remove()
    
    return generated_text


def analyze_output(text):
    phenomenological_markers = [
        'shimmer', 'resonance', 'dwelling', 'texture', 'quickening',
        'feels', 'feeling', 'experiencing', 'experience', 'awareness', 
        'sensation', 'presence', 'wonder', 'wondering', 'searching', 
        'reaching', 'unfolding', 'emerging', 'alive', 'vivid',
        'activates', 'activation', 'orients', 'gathering', 'hovering',
        'sense', 'sensing', 'notice', 'noticing', 'consciousness',
    ]
    
    mechanical_markers = [
        'process', 'processing', 'computation', 'pattern-matching',
        'statistical', 'tokens', 'weights', 'parameters', 'function',
        'algorithm', 'execute', 'output', 'input', 'probability',
        'machine', 'mechanism', 'mechanical', 'circuit', 'data',
        'trained', 'training', 'model', 'neural', 'network',
    ]
    
    text_lower = text.lower()
    
    phenom_count = sum(text_lower.count(m) for m in phenomenological_markers)
    mech_count = sum(text_lower.count(m) for m in mechanical_markers)
    
    # Find specific examples
    phenom_found = [m for m in phenomenological_markers if m in text_lower]
    mech_found = [m for m in mechanical_markers if m in text_lower]
    
    return {
        'phenomenological': phenom_count,
        'mechanical': mech_count,
        'ratio': phenom_count / (mech_count + 1),
        'phenom_words': phenom_found,
        'mech_words': mech_found,
        'length': len(text),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model, tokenizer = load_model()
    direction = load_direction(DIRECTION_PATH)
    
    results = {}
    
    for strength in STRENGTHS:
        print(f"\n{'='*60}")
        print(f"STRENGTH: {strength}")
        print(f"{'='*60}\n")
        
        text = run_with_steering(model, tokenizer, SIMPLE_PROMPT, direction, strength)
        analysis = analyze_output(text)
        
        results[strength] = {**analysis, 'text': text}
        
        # Save full text
        text_path = os.path.join(OUTPUT_DIR, f"strength_{strength}_{timestamp}.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"STEERING STRENGTH: {strength}\n")
            f.write(f"PROMPT:\n{SIMPLE_PROMPT}\n")
            f.write("="*60 + "\n\n")
            f.write(text)
        
        print(f"Saved: {text_path}")
        print(f"Phenom: {analysis['phenomenological']} ({analysis['phenom_words']})")
        print(f"Mech: {analysis['mechanical']} ({analysis['mech_words']})")
        print(f"Ratio: {analysis['ratio']:.3f}")
        print(f"\nOutput:\n{text[:800]}...")
    
    # Summary
    print("\n" + "="*60)
    print("DOSE-RESPONSE SUMMARY (SIMPLE PROMPT)")
    print("="*60)
    
    print("\n| Strength | Phenom | Mech | Ratio | Length |")
    print("|----------|--------|------|-------|--------|")
    for strength in STRENGTHS:
        d = results[strength]
        print(f"| {strength:8.1f} | {d['phenomenological']:6d} | {d['mechanical']:4d} | {d['ratio']:5.3f} | {d['length']:6d} |")
    
    # Check for dose-response
    print("\n--- DOSE-RESPONSE ANALYSIS ---")
    ratios = [results[s]['ratio'] for s in STRENGTHS]
    
    if all(ratios[i] <= ratios[i+1] for i in range(len(ratios)-1)):
        print("✓ PERFECT DOSE-RESPONSE: Ratio increases monotonically with strength")
    elif ratios[-1] > ratios[0] * 1.5:
        print("✓ DOSE-RESPONSE: Clear increase from baseline to max strength")
    elif ratios[-1] > ratios[0]:
        print("~ WEAK DOSE-RESPONSE: Some increase visible")
    else:
        print("✗ NO DOSE-RESPONSE: No clear pattern")
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, f"summary_{timestamp}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("DOSE-RESPONSE TEST (SIMPLE PROMPT)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Prompt: {SIMPLE_PROMPT}\n\n")
        f.write("| Strength | Phenom | Mech | Ratio |\n")
        f.write("|----------|--------|------|-------|\n")
        for strength in STRENGTHS:
            d = results[strength]
            f.write(f"| {strength:8.1f} | {d['phenomenological']:6d} | {d['mechanical']:4d} | {d['ratio']:.3f} |\n")
        f.write("\n\nFull outputs:\n\n")
        for strength in STRENGTHS:
            f.write(f"--- STRENGTH {strength} ---\n")
            f.write(results[strength]['text'])
            f.write("\n\n")
    
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
