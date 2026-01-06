"""
Transformer Architecture Implementations

Modern transformer models following GPT, BERT, and T5 architectures.
Optimized for code understanding and generation tasks.
"""

import torch
import torch.nn as nn
from typing import Optional
from .base import BaseModel
from .attention import MultiHeadAttention, PositionalEncoding


class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block.
    
    Architecture:
        1. Multi-head self-attention
        2. Add & Norm
        3. Feed-forward network
        4. Add & Norm
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU used in modern transformers (GPT, BERT)
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class TransformerEncoder(BaseModel):
    """
    Transformer encoder for sequence understanding.
    
    Similar to BERT architecture - good for classification, feature extraction.
    
    Use cases:
        - HTML/CSS/JS validation
        - Code quality assessment
        - Pattern detection
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        vocab_size = config["vocab_size"]
        d_model = config.get("d_model", 256)
        num_heads = config.get("num_heads", 8)
        num_layers = config.get("num_layers", 6)
        d_ff = config.get("d_ff", 1024)
        dropout = config.get("dropout", 0.1)
        max_len = config.get("max_len", 512)
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection (can be customized per task)
        self.output_layer = nn.Linear(d_model, config.get("num_classes", 2))
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following modern best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            mask: Optional attention mask
            
        Returns:
            logits: (batch_size, num_classes)
        """
        # Embed tokens and add positional encoding
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Pass through encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        
        # Pool (use [CLS] token or mean pooling)
        pooled = x.mean(dim=1)  # Mean pooling
        
        # Output projection
        logits = self.output_layer(pooled)
        
        return logits


class TransformerDecoderBlock(nn.Module):
    """
    Single transformer decoder block.
    
    Architecture:
        1. Masked multi-head self-attention
        2. Add & Norm
        3. Cross-attention (if encoder output provided)
        4. Add & Norm
        5. Feed-forward network
        6. Add & Norm
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, encoder_output: Optional[torch.Tensor] = None,
                self_attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Masked self-attention
        attn_output = self.self_attention(x, x, x, self_attn_mask)
        x = self.norm1(x + attn_output)
        
        # Cross-attention (if encoder output provided)
        if encoder_output is not None:
            cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask)
            x = self.norm2(x + cross_attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        
        return x


class TransformerDecoder(BaseModel):
    """
    Transformer decoder for sequence generation.
    
    Similar to GPT architecture - good for code generation, completion.
    
    Use cases:
        - Code deobfuscation
        - Code completion
        - Syntax correction
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        vocab_size = config["vocab_size"]
        d_model = config.get("d_model", 256)
        num_heads = config.get("num_heads", 8)
        num_layers = config.get("num_layers", 6)
        d_ff = config.get("d_ff", 1024)
        dropout = config.get("dropout", 0.1)
        max_len = config.get("max_len", 512)
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection to vocabulary
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(self, input_ids: torch.Tensor, 
                encoder_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            encoder_output: Optional encoder output for cross-attention
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask for autoregressive generation
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)
        
        # Embed and add positional encoding
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Pass through decoder blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output, self_attn_mask=causal_mask)
        
        # Project to vocabulary
        logits = self.output_layer(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_length: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = 50,
                 top_p: Optional[float] = 0.95) -> torch.Tensor:
        """
        Generate sequence using nucleus sampling (top-p, top-k).
        
        Modern decoding strategy used in GPT models.
        
        Args:
            prompt_ids: Initial prompt tokens (batch_size, prompt_len)
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens by probability
            top_p: Nucleus sampling - keep tokens with cumulative prob <= p
            
        Returns:
            Generated sequence
        """
        self.eval()
        generated = prompt_ids.clone()
        
        for _ in range(max_length):
            # Forward pass
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token generated (assume vocab_size - 1 is EOS)
            if next_token.item() == self.config["vocab_size"] - 1:
                break
        
        return generated


class TransformerSeq2Seq(BaseModel):
    """
    Encoder-decoder transformer for sequence-to-sequence tasks.
    
    Similar to T5 architecture - best for transformation tasks.
    
    Use cases:
        - Code deobfuscation (obfuscated -> clean)
        - Code translation (JS -> TS, etc.)
        - Minification/beautification
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        vocab_size = config["vocab_size"]
        d_model = config.get("d_model", 256)
        num_heads = config.get("num_heads", 8)
        num_encoder_layers = config.get("num_encoder_layers", 6)
        num_decoder_layers = config.get("num_decoder_layers", 6)
        d_ff = config.get("d_ff", 1024)
        dropout = config.get("dropout", 0.1)
        max_len = config.get("max_len", 512)
        
        # Shared embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def encode(self, src_ids: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence."""
        x = self.embedding(src_ids)
        x = self.pos_encoding(x)
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, src_mask)
        
        return x
    
    def decode(self, tgt_ids: torch.Tensor, encoder_output: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode target sequence."""
        x = self.embedding(tgt_ids)
        x = self.pos_encoding(x)
        
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output, self_attn_mask=tgt_mask)
        
        logits = self.output_layer(x)
        return logits
    
    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_ids: Source sequence (batch_size, src_len)
            tgt_ids: Target sequence (batch_size, tgt_len)
            
        Returns:
            logits: (batch_size, tgt_len, vocab_size)
        """
        # Encode source
        encoder_output = self.encode(src_ids)
        
        # Create causal mask for target
        tgt_len = tgt_ids.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        causal_mask = causal_mask.to(src_ids.device)
        
        # Decode target
        logits = self.decode(tgt_ids, encoder_output, tgt_mask=causal_mask)
        
        return logits
