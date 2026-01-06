"""
Attention Mechanisms

Modern attention implementations following GPT/BERT/T5 architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Used in all modern transformers. Implements scaled dot-product attention
    with multiple attention heads for capturing different aspects of relationships.
    
    References:
        - "Attention is All You Need" (Vaswani et al., 2017)
        - Flash Attention optimizations (Dao et al., 2022)
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: Optional attention mask
            
        Returns:
            Output: (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape to (batch, num_heads, seq_len, head_dim)
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Attention output
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(context)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    
    Adds position information to embeddings. Essential for transformers
    since self-attention is permutation-invariant.
    
    References:
        - Original Transformer paper (Vaswani et al., 2017)
        - RoPE (Rotary Position Embedding) - modern alternative
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RelativePositionEncoding(nn.Module):
    """
    Relative position encoding (T5-style).
    
    More modern approach than absolute positional encoding.
    Captures relative distances between tokens instead of absolute positions.
    
    References:
        - "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2020)
    """
    
    def __init__(self, num_heads: int, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        
        # Learnable relative position biases
        self.relative_attention_bias = nn.Embedding(2 * max_distance + 1, num_heads)
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Compute relative position bias.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position bias: (num_heads, seq_len, seq_len)
        """
        # Create relative position matrix
        positions = torch.arange(seq_len, dtype=torch.long)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Clip to max distance
        relative_positions = torch.clamp(
            relative_positions, -self.max_distance, self.max_distance
        ) + self.max_distance
        
        # Get bias values
        bias = self.relative_attention_bias(relative_positions)  # (seq_len, seq_len, num_heads)
        bias = bias.permute(2, 0, 1)  # (num_heads, seq_len, seq_len)
        
        return bias


class CrossAttention(nn.Module):
    """
    Cross-attention for encoder-decoder architectures.
    
    Used in models like T5, BART for attending to encoder outputs
    while decoding.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, decoder_hidden: torch.Tensor, encoder_output: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            decoder_hidden: (batch_size, dec_seq_len, d_model)
            encoder_output: (batch_size, enc_seq_len, d_model)
            mask: Optional mask
            
        Returns:
            Output with cross-attention applied
        """
        # Query from decoder, key/value from encoder
        attn_output = self.attention(decoder_hidden, encoder_output, encoder_output, mask)
        
        # Residual connection + layer norm
        output = self.norm(decoder_hidden + self.dropout(attn_output))
        
        return output
