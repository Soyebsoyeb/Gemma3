import pytest
import torch
from src.models.attention import GroupedQueryAttention, RMSNorm
from src.models.rope import compute_rope_params

class TestGroupedQueryAttention:
    """Comprehensive tests for Grouped Query Attention."""
    
    @pytest.mark.parametrize("num_heads,num_kv_groups", [
        (4, 1),
        (8, 2),
        (16, 4),
    ])
    def test_attention_shapes(self, num_heads, num_kv_groups):
        """Test attention output shapes with different head configurations."""
        d_in = 128
        batch_size = 2
        seq_len = 16
        
        attn = GroupedQueryAttention(d_in, num_heads, num_kv_groups)
        x = torch.randn(batch_size, seq_len, d_in)
        
        # Create RoPE parameters
        head_dim = attn.head_dim
        cos, sin = compute_rope_params(head_dim, context_length=seq_len)
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        out = attn(x, mask, cos, sin)
        
        assert out.shape == (batch_size, seq_len, d_in)
    
    def test_attention_with_qk_norm(self):
        """Test attention with QK normalization."""
        d_in = 128
        num_heads = 4
        num_kv_groups = 1
        
        attn = GroupedQueryAttention(
            d_in, num_heads, num_kv_groups, 
            qk_norm=True
        )
        
        assert attn.q_norm is not None
        assert attn.k_norm is not None
        
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_in)
        cos, sin = compute_rope_params(attn.head_dim, context_length=seq_len)
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        out = attn(x, mask, cos, sin)
        assert out.shape == (batch_size, seq_len, d_in)
    
    def test_attention_scaling(self):
        """Test attention scaling factors."""
        d_in = 128
        num_heads = 4
        num_kv_groups = 1
        
        # Default scaling (1/sqrt(head_dim))
        attn_default = GroupedQueryAttention(d_in, num_heads, num_kv_groups)
        
        # Custom scaling
        attn_custom = GroupedQueryAttention(
            d_in, num_heads, num_kv_groups,
            query_pre_attn_scalar=128.0
        )
        
        assert attn_default.scaling == (attn_default.head_dim) ** -0.5
        assert attn_custom.scaling == (128.0) ** -0.5
    
    def test_attention_masking(self):
        """Test that attention masking works correctly."""
        d_in = 128
        num_heads = 4
        num_kv_groups = 1
        batch_size = 2
        seq_len = 8
        
        attn = GroupedQueryAttention(d_in, num_heads, num_kv_groups)
        x = torch.randn(batch_size, seq_len, d_in)
        cos, sin = compute_rope_params(attn.head_dim, context_length=seq_len)
        
        # Create mask that masks out all positions
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        
        out = attn(x, mask, cos, sin)
        
        # With all masked, attention should be uniform
        # Check that output is not NaN
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
