#!/usr/bin/env python
"""
Script to train the SLM model.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.data import Tokenizer, DatasetProcessor
from src.models import Gemma3Model, GEMMA3_CONFIG_270M
from src.training import Trainer

def main():
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing model...")
    model = Gemma3Model(GEMMA3_CONFIG_270M)
    
    # Initialize dataset processor
    tokenizer = Tokenizer()
    dataset_processor = DatasetProcessor(tokenizer, data_dir="data")
    
    # Training configuration
    training_config = {
        "learning_rate": 1e-4,
        "max_iters": 150000,
        "warmup_steps": 1000,
        "min_lr": 5e-4,
        "eval_iters": 500,
        "batch_size": 32,
        "block_size": 128,
        "gradient_accumulation_steps": 32,
        "dtype": 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    }
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        dataset_processor=dataset_processor,
        config=training_config,
        device=device
    )
    
    # Train
    print("Starting training...")
    trainer.train(save_path="best_model_params.pt")
    
    # Plot losses
    print("Plotting losses...")
    trainer.plot_losses(save_path="training_losses.png")
    print("Training complete!")

if __name__ == "__main__":
    main()
