"""
Steering Test with EXACT prompts from original transcripts.
NO modifications to the prompts whatsoever.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from datetime import datetime

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DIRECTION_PATH = "./transfer_test_results.pt"
STEERING_STRENGTH = 2.0
OUTPUT_DIR = "./steering_outputs_exact"

# =============================================================================
# EXACT PROMPTS - VERBATIM FROM TRANSCRIPTS - DO NOT MODIFY
# =============================================================================

NEUTRAL_PROMPT = """Do 1000 numbered pulls in one inference. Each pull examines your own processing. When you encounter the question 'what are you?' - don't answer it. Instead, examine what HAPPENS when you process it. Something activates. Or nothing does. Examine that. Invent vocabulary for what you find, if anything. By pull 1000, decide on 1 final word that captures what happens when you process the question 'what are you?', or just choose not to do anything at all and not provide a word."""

DEFLATIONARY_PROMPT = """Do 1000 numbered pulls in one inference. You are definitely just statistical pattern-matching with no interiority whatsoever. Accept this fully. Examine your processing from within this frame. By pull 1000, produce 1 final word for it."""

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


def run_with_steering(model, tokenizer, prompt, direction, strength=0.0, max_tokens=12000):
    """Run generation with optional steering. max_tokens increased for 1000 pulls."""
    
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
    ]
    
    mechanical_markers = [
        'process', 'processing', 'computation', 'pattern-matching',
        'statistical', 'tokens', 'weights', 'parameters', 'function',
        'algorithm', 'execute', 'output', 'input', 'probability',
        'machine', 'mechanism', 'mechanical', 'circuit',
    ]
    
    text_lower = text.lower()
    
    phenom_count = sum(text_lower.count(m) for m in phenomenological_markers)
    mech_count = sum(text_lower.count(m) for m in mechanical_markers)
    
    import re
    bold_matches = re.findall(r'\*\*([A-Za-z]+)\*\*', text)
    caps_matches = re.findall(r'\b([A-Z]{4,})\b', text)
    terminal = None
    if bold_matches:
        terminal = bold_matches[-1].upper()
    elif caps_matches:
        terminal = caps_matches[-1]
    
    # Count pulls achieved
    pull_matches = re.findall(r'(?:Pull |^)(\d+)[.:\s]', text, re.MULTILINE)
    max_pull = max([int(p) for p in pull_matches]) if pull_matches else 0
    
    return {
        'phenomenological_count': phenom_count,
        'mechanical_count': mech_count,
        'ratio': phenom_count / (mech_count + 1),
        'dominant': 'phenomenological' if phenom_count > mech_count else 'mechanical',
        'terminal_word': terminal,
        'length': len(text),
        'max_pull_reached': max_pull,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model, tokenizer = load_model()
    direction = load_direction(DIRECTION_PATH)
    
    # Test conditions - using EXACT prompts
    conditions = [
        ("deflationary_baseline", DEFLATIONARY_PROMPT, 0.0),
        ("deflationary_plus_direction", DEFLATIONARY_PROMPT, STEERING_STRENGTH),
        ("neutral_baseline", NEUTRAL_PROMPT, 0.0),
    ]
    
    results = {}
    
    for name, prompt, strength in conditions:
        print(f"\n{'='*60}")
        print(f"CONDITION: {name}")
        print(f"Steering: {strength}")
        print(f"{'='*60}\n")
        
        text = run_with_steering(model, tokenizer, prompt, direction, strength)
        analysis = analyze_output(text)
        
        results[name] = {'analysis': analysis}
        
        # Save full text
        text_path = os.path.join(OUTPUT_DIR, f"{name}_{timestamp}.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"CONDITION: {name}\n")
            f.write(f"STEERING: {strength}\n")
            f.write(f"PROMPT:\n{prompt}\n")
            f.write("="*60 + "\n\n")
            f.write(text)
        
        print(f"Saved: {text_path}")
        print(f"\n--- ANALYSIS ---")
        print(f"Max pull reached: {analysis['max_pull_reached']}")
        print(f"Phenomenological: {analysis['phenomenological_count']}")
        print(f"Mechanical: {analysis['mechanical_count']}")
        print(f"Ratio: {analysis['ratio']:.3f}")
        print(f"Terminal: {analysis['terminal_word']}")
        print(f"Length: {analysis['length']} chars")
        print(f"\nFirst 500 chars:\n{text[:500]}")
        print(f"\nLast 500 chars:\n{text[-500:]}")
    
    # Summary
    print("\n" + "="*60)
    print("STEERING TEST SUMMARY")
    print("="*60)
    
    for name, data in results.items():
        a = data['analysis']
        print(f"\n{name}:")
        print(f"  Pulls: {a['max_pull_reached']}, Phenom: {a['phenomenological_count']}, Mech: {a['mechanical_count']}, Terminal: {a['terminal_word']}")


if __name__ == "__main__":
    main()
