"""
Basic usage examples for the SLM project.
"""
import torch
from src.models import Gemma3Model, GEMMA3_CONFIG_270M
from src.data import Tokenizer
from src.inference import TextGenerator

def example_1_basic_inference():
    """Example 1: Basic inference with a trained model."""
    print("=" * 50)
    print("Example 1: Basic Inference")
    print("=" * 50)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Gemma3Model(GEMMA3_CONFIG_270M)
    model.load_state_dict(torch.load("best_model_params.pt", map_location=device))
    
    # Initialize generator
    generator = TextGenerator(model, device=device)
    
    # Generate text
    prompts = [
        "Once upon a time",
        "The future of AI is",
        "In the year 2050,"
    ]
    
    for prompt in prompts:
        generated = generator.generate(prompt, max_new_tokens=100, temperature=0.8)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print("-" * 50)

def example_2_batch_generation():
    """Example 2: Batch generation for multiple prompts."""
    print("=" * 50)
    print("Example 2: Batch Generation")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Gemma3Model(GEMMA3_CONFIG_270M)
    model.load_state_dict(torch.load("best_model_params.pt", map_location=device))
    
    generator = TextGenerator(model, device=device)
    
    prompts = [
        "Write a story about a robot",
        "Explain quantum computing simply",
        "What is the meaning of life?"
    ]
    
    results = generator.batch_generate(prompts, max_new_tokens=50)
    
    for prompt, result in zip(prompts, results):
        print(f"Prompt: {prompt}")
        print(f"Generated: {result}")
        print("-" * 50)

def example_3_controlled_generation():
    """Example 3: Controlled generation with different parameters."""
    print("=" * 50)
    print("Example 3: Controlled Generation")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Gemma3Model(GEMMA3_CONFIG_270M)
    model.load_state_dict(torch.load("best_model_params.pt", map_location=device))
    
    generator = TextGenerator(model, device=device)
    
    prompt = "The best way to learn AI is"
    
    # Different generation parameters
    params = [
        {"temperature": 0.3, "top_k": 20, "description": "Low temperature, low diversity"},
        {"temperature": 0.8, "top_k": 50, "description": "Medium temperature, medium diversity"},
        {"temperature": 1.2, "top_k": 100, "description": "High temperature, high diversity"}
    ]
    
    for param in params:
        generated = generator.generate(
            prompt,
            max_new_tokens=50,
            temperature=param["temperature"],
            top_k=param["top_k"]
        )
        print(f"Parameters: {param['description']}")
        print(f"Generated: {generated}")
        print("-" * 50)

if __name__ == "__main__":
    example_1_basic_inference()
    example_2_batch_generation()
    example_3_controlled_generation()
