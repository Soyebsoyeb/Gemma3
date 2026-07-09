import pytest
import os
import tempfile
import numpy as np
from datasets import Dataset
from src.data import Tokenizer, DatasetProcessor

class TestTokenizer:
    """Test tokenizer functionality."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = Tokenizer()
        assert tokenizer.vocab_size == 50257
        assert tokenizer.enc is not None
    
    def test_encode_decode(self):
        """Test encoding and decoding."""
        tokenizer = Tokenizer()
        text = "Hello, world!"
        
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert decoded.strip() == text.strip()
    
    def test_process_example(self):
        """Test processing dataset example."""
        tokenizer = Tokenizer()
        example = {"text": "Sample text for testing."}
        
        processed = tokenizer.process_example(example)
        
        assert "ids" in processed
        assert "len" in processed
        assert processed["len"] == len(processed["ids"])

class TestDatasetProcessor:
    """Test dataset processing functionality."""
    
    def test_dataset_initialization(self):
        """Test dataset processor initialization."""
        tokenizer = Tokenizer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = DatasetProcessor(tokenizer, data_dir=tmpdir)
            assert processor.data_dir == tmpdir
    
    def test_save_split_to_binary(self):
        """Test saving split to binary file."""
        tokenizer = Tokenizer()
        
        # Create sample dataset
        data = {
            "text": ["Sample text 1", "Sample text 2", "Sample text 3"],
        }
        dataset = Dataset.from_dict(data)
        
        # Tokenize dataset
        tokenized = dataset.map(tokenizer.process_example, remove_columns=['text'])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = DatasetProcessor(tokenizer, data_dir=tmpdir)
            processor._save_split_to_binary(tokenized, "train")
            
            # Check file exists
            assert os.path.exists(os.path.join(tmpdir, "train.bin"))
            
            # Check file size
            file_size = os.path.getsize(os.path.join(tmpdir, "train.bin"))
            assert file_size > 0
    
    def test_get_batch(self):
        """Test batch retrieval."""
        tokenizer = Tokenizer()
        
        # Create sample data
        data = {
            "text": ["Sample text 1", "Sample text 2", "Sample text 3", "Sample text 4"],
        }
        dataset = Dataset.from_dict(data)
        
        # Tokenize and save
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = DatasetProcessor(tokenizer, data_dir=tmpdir)
            tokenized = dataset.map(tokenizer.process_example, remove_columns=['text'])
            processor._save_split_to_binary(tokenized, "train")
            
            # Test batch retrieval
            batch_size = 2
            block_size = 5
            x, y = processor.get_batch("train", batch_size, block_size, "cpu")
            
            assert x.shape == (batch_size, block_size)
            assert y.shape == (batch_size, block_size)
            assert x.dtype == torch.int64
            assert y.dtype == torch.int64
