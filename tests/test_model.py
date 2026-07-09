import pytest
import torch
import numpy as np
from src.models import Gemma3Model, GEMMA3_CONFIG_270M
from src.models.attention import RMSNorm, GroupedQueryAttention
from src.models.rope import compute_rope_params, apply_rope

class TestModelComponents:
    """Test individual model components."""
    
    def test_rms_norm(self):
        """Test RMSNorm layer."""
        emb_dim = 128
        norm = RMSNorm(emb_dim)
        x = torch.randn(4, 10, emb_dim)
        out = norm(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_rope_computation(self):
        """Test RoPE parameter computation."""
        head_dim = 64
        context_length = 128
        
        cos, sin = compute_rope_params(head_dim, context_length=context_length)
        
        assert cos.shape == (context_length, head_dim)
        assert sin.shape == (context_length, head_dim)
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()
    
    def test_apply_rope(self):
        """Test applying RoPE to tensors."""
        batch_size = 2
        num_heads = 4
        seq_len = 16
        head_dim = 64
        
        cos, sin = compute_rope_params(head_dim, context_length=seq_len)
        x = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        x_rotated = apply_rope(x, cos, sin)
        
        assert x_rotated.shape == x.shape
        assert not torch.isnan(x_rotated).any()
        assert not torch.isinf(x_rotated).any()
    
    def test_grouped_query_attention(self):
        """Test Grouped Query Attention layer."""
        d_in = 128
        num_heads = 4
        num_kv_groups = 2
        batch_size = 2
        seq_len = 16
        
        attn = GroupedQueryAttention(d_in, num_heads, num_kv_groups)
        x = torch.randn(batch_size, seq_len, d_in)
        cos, sin = compute_rope_params(attn.head_dim, context_length=seq_len)
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        out = attn(x, mask, cos, sin)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

class TestGemma3Model:
    """Test the full Gemma3 model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        config = GEMMA3_CONFIG_270M.copy()
        config["context_length"] = 128  # Reduce for testing
        
        model = Gemma3Model(config)
        
        assert model is not None
        assert len(model.blocks) == config["n_layers"]
    
    def test_forward_pass(self):
        """Test forward pass."""
        config = GEMMA3_CONFIG_270M.copy()
        config["context_length"] = 128
        config["dtype"] = torch.float32
        
        model = Gemma3Model(config)
        batch_size = 2
        seq_len = 32
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        logits, loss = model(input_ids, targets=input_ids)
        
        assert logits.shape == (batch_size, seq_len, config["vocab_size"])
        assert loss is not None
        assert not torch.isnan(loss)
    
    def test_generation(self):
        """Test text generation."""
        config = GEMMA3_CONFIG_270M.copy()
        config["context_length"] = 128
        config["dtype"] = torch.float32
        
        model = Gemma3Model(config)
        input_ids = torch.randint(0, 1000, (1, 10))
        
        generated = model.generate(input_ids, max_new_tokens=20)
        
        assert generated.shape[0] == 1
        assert generated.shape[1] > 10  # Should have generated new tokens
    
    def test_masks_creation(self):
        """Test attention mask creation."""
        config = GEMMA3_CONFIG_270M.copy()
        config["context_length"] = 32
        model = Gemma3Model(config)
        
        seq_len = 16
        mask_global, mask_local = model._create_masks(seq_len, "cpu")
        
        # Global mask: upper triangular (future tokens masked)
        assert mask_global.shape == (seq_len, seq_len)
        assert mask_global.dtype == torch.bool
        
        # Check that local mask is different from global
        assert not torch.all(mask_global == mask_local)
        
        # Check sliding window property
        for i in range(seq_len):
            for j in range(seq_len):
                # Tokens beyond sliding window should be masked for local attention
                if i - j >= config["sliding_window"]:
                    assert mask_local[i, j]
