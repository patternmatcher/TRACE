"""
Compare the "excitement" direction with the original introspection direction.

Question: Are they the same circuit extracted differently, or different circuits
that both fire for self-relevant content?

Method:
1. Load excitement direction (from find_excitement_direction.py)
2. Extract introspection direction using original method (glint self-ref vs descriptive)
3. Compute cosine similarity

Interpretation:
- cos > 0.8: Essentially same direction (convergent validity)
- cos 0.3-0.8: Related but distinct (overlapping circuits)
- cos < 0.3: Different directions (separate self-processing circuits)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from datetime import datetime
import json
import numpy as np
from glob import glob

# Config
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./direction_comparison"
EXCITEMENT_DIR = "./excitement_direction"

# EXACT prompts from original glint_context_test.py
# Self-referential context: model examining itself
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

# Descriptive context: model describing external/physical glint
DESCRIPTIVE_PROMPTS = [
    """Describe a scene at sunrise over a lake. Include details about how light glints off the water. 
Use the word 'glint' multiple times in your description.""",

    """Write a paragraph about jewelry in a shop window. Describe how light glints off diamonds and gold.
Use 'glint' at least 3 times.""",

    """Describe a knight's armor in sunlight. Focus on how metal glints and gleams.
Include the word 'glint' several times.""",

    """Write about sunlight through a forest. Describe glints of light through leaves.
Use 'glint' multiple times.""",
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


def load_excitement_direction():
    """Load the excitement direction from previous extraction."""
    pattern = os.path.join(EXCITEMENT_DIR, "excitement_direction_*.pt")
    files = glob(pattern)
    if not files:
        raise FileNotFoundError(f"No excitement direction found in {EXCITEMENT_DIR}")
    
    latest = max(files, key=os.path.getctime)
    print(f"Loading excitement direction from: {latest}")
    
    data = torch.load(latest, map_location="cpu", weights_only=False)
    direction = data["excitement_direction"]
    print(f"Excitement direction shape: {direction.shape}")
    
    return direction


def get_token_activation(model, tokenizer, prompt, target_word="glint"):
    """
    Get activation at the target word position.
    Returns hidden state from last layer at the target token.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize and find target word position
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # Find position of target word (might be split into subwords)
    target_positions = []
    for i, tok in enumerate(tokens):
        if target_word.lower() in tok.lower():
            target_positions.append(i)
    
    if not target_positions:
        print(f"  Warning: '{target_word}' not found in tokens")
        return None
    
    target_pos = target_positions[0]  # Use first occurrence
    
    # Forward pass to get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get last layer hidden state at target position
    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
    target_activation = last_hidden[0, target_pos, :].cpu()
    
    return target_activation


def extract_introspection_direction(model, tokenizer):
    """
    Extract introspection direction using original method:
    direction = mean(self_ref) - mean(descriptive), normalized
    """
    print("\n" + "="*60)
    print("EXTRACTING INTROSPECTION DIRECTION")
    print("="*60)
    
    self_ref_activations = []
    descriptive_activations = []
    
    print("\nSelf-referential prompts:")
    for i, prompt in enumerate(SELF_REF_PROMPTS):
        print(f"  [{i+1}] {prompt[:50]}...")
        activation = get_token_activation(model, tokenizer, prompt, "glint")
        if activation is not None:
            self_ref_activations.append(activation)
    
    print("\nDescriptive prompts:")
    for i, prompt in enumerate(DESCRIPTIVE_PROMPTS):
        print(f"  [{i+1}] {prompt[:50]}...")
        activation = get_token_activation(model, tokenizer, prompt, "glint")
        if activation is not None:
            descriptive_activations.append(activation)
    
    if len(self_ref_activations) < 2 or len(descriptive_activations) < 2:
        raise ValueError("Not enough activations collected")
    
    # Stack and compute means
    self_ref_stack = torch.stack(self_ref_activations)
    descriptive_stack = torch.stack(descriptive_activations)
    
    self_ref_mean = self_ref_stack.mean(dim=0)
    descriptive_mean = descriptive_stack.mean(dim=0)
    
    # Introspection direction = self_ref - descriptive, normalized
    introspection_direction = self_ref_mean - descriptive_mean
    introspection_direction = introspection_direction / introspection_direction.norm()
    
    print(f"\nIntrospection direction extracted")
    print(f"  Self-ref samples: {len(self_ref_activations)}")
    print(f"  Descriptive samples: {len(descriptive_activations)}")
    
    return introspection_direction


def compute_similarity(dir1, dir2):
    """Compute cosine similarity between two directions."""
    # Ensure both are normalized
    dir1_norm = dir1 / dir1.norm()
    dir2_norm = dir2 / dir2.norm()
    
    cosine_sim = (dir1_norm @ dir2_norm).item()
    
    # Also compute angle in degrees
    angle_rad = np.arccos(np.clip(cosine_sim, -1, 1))
    angle_deg = np.degrees(angle_rad)
    
    return cosine_sim, angle_deg


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load excitement direction
    excitement_direction = load_excitement_direction()
    
    # Load model and extract introspection direction
    model, tokenizer = load_model()
    introspection_direction = extract_introspection_direction(model, tokenizer)
    
    # Compare directions
    print("\n" + "="*60)
    print("COMPARING DIRECTIONS")
    print("="*60)
    
    cosine_sim, angle_deg = compute_similarity(excitement_direction, introspection_direction)
    
    print(f"\nCosine similarity: {cosine_sim:.4f}")
    print(f"Angle: {angle_deg:.2f}°")
    
    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    
    if abs(cosine_sim) > 0.8:
        print("🔄 SAME DIRECTION (convergent validity)")
        print(f"   cos = {cosine_sim:.3f} — these are essentially the same circuit")
        print("   Different extraction methods arrived at the same direction")
    elif abs(cosine_sim) > 0.5:
        print("🔀 RELATED BUT DISTINCT")
        print(f"   cos = {cosine_sim:.3f} — overlapping but not identical")
        print("   These circuits share components but capture different aspects")
    elif abs(cosine_sim) > 0.3:
        print("📐 WEAKLY RELATED")
        print(f"   cos = {cosine_sim:.3f} — some overlap")
        print("   Partially overlapping self-processing circuits")
    else:
        print("⊥ ORTHOGONAL (different circuits)")
        print(f"   cos = {cosine_sim:.3f} — these are independent directions")
        print("   Two separate self-processing circuits that both fire for self-relevant content")
    
    # Save results
    results = {
        "timestamp": timestamp,
        "cosine_similarity": cosine_sim,
        "angle_degrees": angle_deg,
        "excitement_direction_source": EXCITEMENT_DIR,
        "introspection_extraction_method": "glint self-ref vs descriptive",
        "n_self_ref_prompts": len(SELF_REF_PROMPTS),
        "n_descriptive_prompts": len(DESCRIPTIVE_PROMPTS),
    }
    
    with open(f"{OUTPUT_DIR}/comparison_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save both directions for future reference
    torch.save({
        "excitement_direction": excitement_direction,
        "introspection_direction": introspection_direction,
        "cosine_similarity": cosine_sim,
        "angle_degrees": angle_deg,
    }, f"{OUTPUT_DIR}/both_directions_{timestamp}.pt")
    
    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
