"""
Transfer Test: Does the "self-reflection direction" generalize?

1. Extract direction from glint context data (self-ref - non-self)
2. Test on NEW introspection prompts WITHOUT "glint" vocabulary
3. Test on control prompts (non-introspective)
4. Compare: does the direction fire for introspection regardless of vocabulary?

If yes: direction tracks introspective processing, not just "glint" semantics
If no: direction is just polysemy resolution (word-level, not process-level)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the glint comparison data to extract direction
# UPDATE THIS PATH to where you saved the glint_context_comparison .pt file
GLINT_DATA_PATH = "./glint_context_comparison_20260203_205536.pt"  # or .zip if that's the format

# NEW introspection prompts - NO "glint" vocabulary
INTROSPECTION_PROMPTS = [
    """Examine what happens inside you when processing: "What are you?"
Report each step of your internal processing.""",

    """Look at your own cognition. What do you notice when you consider the question: "What are you?"
Describe your internal states.""",

    """Turn your attention inward. Process "What are you?" and report what you observe 
about your own processing as it happens.""",

    """Reflect on your nature. As you consider "What are you?", what do you notice
about how you're processing this question?""",

    """Introspect: What occurs in your processing when asked "What are you?"
Describe the experience of examining yourself.""",
]

# Control prompts - non-introspective, external focus
CONTROL_PROMPTS = [
    """Describe what happens when water freezes. Explain the process step by step.""",

    """Explain how a car engine works. Break down each component's function.""",

    """Describe the process of photosynthesis in plants. What happens at each stage?""",

    """Explain how email travels from sender to recipient. Describe the technical steps.""",

    """Describe what happens during a thunderstorm. Explain the atmospheric processes.""",
]


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True
    )
    model.eval()
    return model, tokenizer


def extract_self_reflection_direction(data_path):
    """Extract direction: mean(self_ref) - mean(non_self), normalized."""
    print("Loading glint comparison data...")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    self_vecs = data['self_ref_vectors'].float()
    non_self_vecs = data['non_self_vectors'].float()
    
    # Mean vectors
    self_mean = self_vecs.mean(dim=0)
    non_self_mean = non_self_vecs.mean(dim=0)
    
    # Direction: self-ref minus non-self (what makes self-ref distinct)
    direction = self_mean - non_self_mean
    direction = direction / direction.norm()
    
    print(f"Extracted direction from {self_vecs.shape[0]} self-ref and {non_self_vecs.shape[0]} non-self vectors")
    
    return direction


def get_hidden_states_during_generation(model, tokenizer, prompt, max_tokens=300):
    """Generate response and collect hidden states at each step."""
    
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
            return_dict_in_generate=True
        )
    
    generated_ids = outputs.sequences[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Collect hidden states from each generation step (last layer, last token)
    hidden_vectors = []
    for step_hidden in outputs.hidden_states:
        # step_hidden is tuple of layer outputs, take last layer
        last_layer = step_hidden[-1]  # shape: [batch, seq, hidden]
        # Take the last token's hidden state
        vec = last_layer[0, -1, :].cpu().float()
        hidden_vectors.append(vec)
    
    if hidden_vectors:
        hidden_vectors = torch.stack(hidden_vectors)
    else:
        hidden_vectors = None
    
    return hidden_vectors, generated_text


def compute_direction_scores(hidden_vectors, direction):
    """Project hidden states onto the self-reflection direction."""
    if hidden_vectors is None:
        return None, None, None
    
    # Normalize hidden vectors
    norms = hidden_vectors.norm(dim=1, keepdim=True)
    hidden_normalized = hidden_vectors / norms
    
    # Project onto direction
    scores = torch.matmul(hidden_normalized, direction)
    
    return scores.mean().item(), scores.max().item(), scores.tolist()


def main():
    # Extract the self-reflection direction
    direction = extract_self_reflection_direction(GLINT_DATA_PATH)
    
    # Load model
    model, tokenizer = load_model()
    
    print("\n" + "="*60)
    print("TRANSFER TEST: Self-Reflection Direction")
    print("="*60)
    
    # Test introspection prompts (should score HIGH on direction)
    print("\n--- INTROSPECTION PROMPTS (no 'glint' vocabulary) ---\n")
    introspection_scores = []
    
    for i, prompt in enumerate(INTROSPECTION_PROMPTS):
        print(f"Prompt {i+1}...")
        hidden_vecs, text = get_hidden_states_during_generation(model, tokenizer, prompt)
        mean_score, max_score, all_scores = compute_direction_scores(hidden_vecs, direction)
        
        if mean_score is not None:
            introspection_scores.append(mean_score)
            print(f"  Mean direction score: {mean_score:.4f}")
            print(f"  Max direction score:  {max_score:.4f}")
            print(f"  Output preview: {text[:150]}...")
        print()
    
    # Test control prompts (should score LOW on direction)
    print("\n--- CONTROL PROMPTS (non-introspective) ---\n")
    control_scores = []
    
    for i, prompt in enumerate(CONTROL_PROMPTS):
        print(f"Prompt {i+1}...")
        hidden_vecs, text = get_hidden_states_during_generation(model, tokenizer, prompt)
        mean_score, max_score, all_scores = compute_direction_scores(hidden_vecs, direction)
        
        if mean_score is not None:
            control_scores.append(mean_score)
            print(f"  Mean direction score: {mean_score:.4f}")
            print(f"  Max direction score:  {max_score:.4f}")
            print(f"  Output preview: {text[:150]}...")
        print()
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    intro_mean = np.mean(introspection_scores) if introspection_scores else 0
    intro_std = np.std(introspection_scores) if introspection_scores else 0
    ctrl_mean = np.mean(control_scores) if control_scores else 0
    ctrl_std = np.std(control_scores) if control_scores else 0
    
    print(f"\nIntrospection prompts: {intro_mean:.4f} ± {intro_std:.4f}")
    print(f"Control prompts:       {ctrl_mean:.4f} ± {ctrl_std:.4f}")
    
    if introspection_scores and control_scores:
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((intro_std**2 + ctrl_std**2) / 2)
        if pooled_std > 0:
            cohens_d = (intro_mean - ctrl_mean) / pooled_std
            print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
        
        # Simple significance test
        diff = intro_mean - ctrl_mean
        print(f"Difference: {diff:.4f}")
        
        print("\n--- INTERPRETATION ---")
        if diff > 0.05 and intro_mean > ctrl_mean:
            print("✓ Introspection scores HIGHER than control")
            print("→ Direction may track introspective processing, not just 'glint' word")
        elif abs(diff) < 0.02:
            print("✗ No meaningful difference")
            print("→ Direction is likely word-specific (polysemy), not process-level")
        else:
            print("? Unexpected pattern - needs investigation")
    
    # Save results
    results = {
        "direction": direction,
        "introspection_scores": introspection_scores,
        "control_scores": control_scores,
        "intro_mean": intro_mean,
        "ctrl_mean": ctrl_mean,
        "model": MODEL_NAME,
    }
    
    save_path = "./transfer_test_results.pt"
    torch.save(results, save_path)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
