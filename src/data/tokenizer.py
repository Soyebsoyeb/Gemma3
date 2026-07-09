import tiktoken
import numpy as np
from typing import List, Dict, Any
import os

class Tokenizer:
    """GPT-2 tokenizer wrapper for encoding/decoding text."""
    
    def __init__(self, encoding_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.enc.encode_ordinary(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.enc.decode(token_ids)
    
    def process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Process a dataset example by tokenizing the text."""
        ids = self.encode(example['text'])
        return {'ids': ids, 'len': len(ids)}
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
