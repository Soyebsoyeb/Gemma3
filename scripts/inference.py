#!/usr/bin/env python
"""
Script to run inference with the trained SLM model.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.data import Tokenizer
from src.models import Gemma3Model, GEMMA3_CONFIG_270M

def generate_text(prompt: str, model, tokenizer, max_tokens: int = 200, temperature: float = 1.0):
    """Generate text from a prompt."""
    context = torch.tensor(tokenizer.encode(prompt)).unsqueeze(dim=0)
    generated = model.generate(context, max_tokens, temperature=temperature)
    return tokenizer.decode(generated.squeeze().tolist())

def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = Gemma3Model(GEMMA3_CONFIG_270M)
    model.load_state_dict(torch.load("best_model_params.pt", map_location=torch.device(device)))
    model = model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = Tokenizer()
    
    # Test prompts
    prompts = [
        "Once upon a time there was a pumpkin.",
        "A little girl went to the woods",
        "Grandmother was telling the kids story about a unicorn"
    ]
    
    print("\n" + "="*50)
    print("Generating text from prompts...")
    print("="*50 + "\n")
    
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        print("-" * 50)
        generated_text = generate_text(prompt, model, tokenizer, max_tokens=200)
        print(f"Generated: {generated_text}")
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()
