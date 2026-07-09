import os
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Dict, Any
import torch
from datasets import DatasetDict

class DatasetProcessor:
    """Handle dataset tokenization and storage as binary files."""
    
    def __init__(self, tokenizer, data_dir: str = "data"):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def tokenize_dataset(self, dataset: DatasetDict, num_proc: int = 8) -> None:
        """
        Tokenize the dataset and save to binary files.
        
        Args:
            dataset: HuggingFace dataset with 'text' column
            num_proc: Number of processes for parallel tokenization
        """
        # Process the dataset
        tokenized = dataset.map(
            self.tokenizer.process_example,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )
        
        # Save each split to binary file
        for split, dset in tokenized.items():
            self._save_split_to_binary(dset, split)
    
    def _save_split_to_binary(self, dataset, split: str) -> None:
        """Save a dataset split to a binary file."""
        arr_len = np.sum(dataset['len'], dtype=np.uint64)
        filename = os.path.join(self.data_dir, f'{split}.bin')
        dtype = np.uint16  # since max token value < 2^16
        
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dataset.shard(
                num_shards=total_batches, 
                index=batch_idx, 
                contiguous=True
            ).with_format('numpy')
            
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        
        arr.flush()
    
    def get_batch(self, split: str, batch_size: int, block_size: int, device: str) -> tuple:
        """
        Get a batch of input-target pairs from the dataset.
        
        Args:
            split: 'train' or 'val'
            batch_size: Number of sequences in batch
            block_size: Context window size
            device: Device to load tensors to
            
        Returns:
            Tuple of (inputs, targets) tensors
        """
        data = np.memmap(
            os.path.join(self.data_dir, f'{split}.bin'), 
            dtype=np.uint16, 
            mode='r'
        )
        
        # Randomly select starting positions
        ix = torch.randint(len(data) - block_size, (batch_size,))
        
        # Create input and target sequences
        x = torch.stack([
            torch.from_numpy((data[i:i+block_size]).astype(np.int64)) 
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) 
            for i in ix
        ])
        
        # Move to device
        if device == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
            
        return x, y
