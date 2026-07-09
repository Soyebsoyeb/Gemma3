#!/usr/bin/env python
"""
Script to prepare and tokenize the dataset.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from src.data import Tokenizer, DatasetProcessor

def main():
    # Load dataset (replace with your dataset)
    print("Loading dataset...")
    # Example with a sample dataset - replace with your actual dataset
    dataset = load_dataset("tiny_shakespeare", split="train")
    
    # Create train/val split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Initialize tokenizer and processor
    tokenizer = Tokenizer()
    processor = DatasetProcessor(tokenizer, data_dir="data")
    
    # Tokenize and save
    print("Tokenizing dataset...")
    processor.tokenize_dataset(dataset, num_proc=8)
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
