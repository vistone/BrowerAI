"""
Specialized Parsers for HTML/CSS/JS

Domain-specific models leveraging transformer architectures
for web technology understanding.
"""

import torch
import torch.nn as nn
from .transformer import TransformerEncoder
from typing import Optional, Dict


class HTMLParser(TransformerEncoder):
    """
    HTML parsing model with multi-task learning.
    
    Tasks:
        1. Validity classification (valid/malformed)
        2. Complexity estimation (DOM depth, element count)
        3. Structure prediction (semantic tags)
    
    Modern techniques:
        - Multi-task learning (improves generalization)
        - Auxiliary tasks for better representations
    """
    
    def __init__(self, config: dict):
        # Add HTML-specific config defaults
        config.setdefault("num_classes", 2)  # Binary validity
        config.setdefault("d_model", 256)
        config.setdefault("num_heads", 8)
        config.setdefault("num_layers", 6)
        
        super().__init__(config)
        
        # Multi-task heads
        d_model = config["d_model"]
        self.validity_head = nn.Linear(d_model, 2)  # Valid/invalid
        self.complexity_head = nn.Linear(d_model, 1)  # Complexity score
        self.semantic_head = nn.Linear(d_model, 20)  # Common semantic tags
        
    def forward(self, input_ids: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Multi-task forward pass.
        
        Returns:
            Dict with multiple outputs:
                - validity: (batch_size, 2)
                - complexity: (batch_size, 1)  
                - semantics: (batch_size, 20)
        """
        # Get encoder representation
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        
        # Pool representation
        pooled = x.mean(dim=1)
        
        # Multi-task predictions
        return {
            "validity": self.validity_head(pooled),
            "complexity": self.complexity_head(pooled).squeeze(-1),
            "semantics": self.semantic_head(pooled)
        }


class CSSParser(TransformerEncoder):
    """
    CSS parsing and optimization model.
    
    Tasks:
        1. Rule validation
        2. Selector complexity analysis
        3. Unused style detection
        4. Optimization suggestions
    """
    
    def __init__(self, config: dict):
        config.setdefault("num_classes", 2)
        config.setdefault("d_model", 128)  # CSS typically shorter
        config.setdefault("num_heads", 4)
        config.setdefault("num_layers", 4)
        
        super().__init__(config)
        
        d_model = config["d_model"]
        self.validity_head = nn.Linear(d_model, 2)
        self.complexity_head = nn.Linear(d_model, 1)
        self.optimization_head = nn.Linear(d_model, 10)  # Optimization categories
        
    def forward(self, input_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        
        pooled = x.mean(dim=1)
        
        return {
            "validity": self.validity_head(pooled),
            "complexity": self.complexity_head(pooled).squeeze(-1),
            "optimization": self.optimization_head(pooled)
        }


class JSParser(TransformerEncoder):
    """
    JavaScript parsing and analysis model.
    
    Tasks:
        1. Syntax validation
        2. Obfuscation detection
        3. Pattern recognition (frameworks, libraries)
        4. Complexity metrics
        5. Security risk detection
    
    Modern features:
        - Contrastive learning for obfuscation detection
        - Few-shot learning for framework detection
    """
    
    def __init__(self, config: dict):
        config.setdefault("num_classes", 2)
        config.setdefault("d_model", 384)  # JS more complex
        config.setdefault("num_heads", 12)
        config.setdefault("num_layers", 8)
        
        super().__init__(config)
        
        d_model = config["d_model"]
        self.validity_head = nn.Linear(d_model, 2)
        self.obfuscation_head = nn.Linear(d_model, 5)  # Obfuscation types
        self.framework_head = nn.Linear(d_model, 100)  # Framework detection
        self.complexity_head = nn.Linear(d_model, 1)
        self.security_head = nn.Linear(d_model, 10)  # Security patterns
        
        # Contrastive learning projection head
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128)  # Projection space
        )
        
    def forward(self, input_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_embedding: bool = False) -> Dict[str, torch.Tensor]:
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        
        pooled = x.mean(dim=1)
        
        outputs = {
            "validity": self.validity_head(pooled),
            "obfuscation": self.obfuscation_head(pooled),
            "framework": self.framework_head(pooled),
            "complexity": self.complexity_head(pooled).squeeze(-1),
            "security": self.security_head(pooled)
        }
        
        if return_embedding:
            outputs["embedding"] = self.projection_head(pooled)
        
        return outputs
    
    def compute_contrastive_loss(self, embeddings: torch.Tensor, 
                                  labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """
        Contrastive loss for learning obfuscation-invariant representations.
        
        Modern technique from SimCLR, MoCo for self-supervised learning.
        
        Args:
            embeddings: (batch_size, embedding_dim)
            labels: (batch_size) - same label means same semantic content
            temperature: Temperature parameter
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        # Create positive pair mask
        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float()
        
        # Remove self-similarity
        mask = mask - torch.eye(mask.size(0), device=mask.device)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True) - exp_sim
        
        log_prob = similarity_matrix - torch.log(sum_exp_sim + 1e-6)
        
        # Mean over positive pairs
        loss = -(log_prob * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        
        return loss.mean()


class UnifiedWebParser(nn.Module):
    """
    Unified model for HTML/CSS/JS understanding.
    
    Modern multi-modal approach:
        - Shared backbone for common web syntax understanding
        - Task-specific heads for each language
        - Cross-task knowledge transfer
    
    Inspired by:
        - Multi-task learning in NLP (MT-DNN, T5)
        - Unified vision-language models (CLIP, ALIGN)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        vocab_size = config["vocab_size"]
        d_model = config.get("d_model", 256)
        
        # Shared embedding and encoder
        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Task-specific encoders (lighter than full models)
        self.html_parser = HTMLParser(config)
        self.css_parser = CSSParser(config)
        self.js_parser = JSParser(config)
        
        # Cross-task fusion layer
        self.fusion_layer = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
    def forward(self, html_ids: Optional[torch.Tensor] = None,
                css_ids: Optional[torch.Tensor] = None,
                js_ids: Optional[torch.Tensor] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Process multiple input types and fuse representations.
        
        Returns:
            Dict of predictions for each available input type
        """
        outputs = {}
        
        if html_ids is not None:
            outputs["html"] = self.html_parser(html_ids)
        
        if css_ids is not None:
            outputs["css"] = self.css_parser(css_ids)
        
        if js_ids is not None:
            outputs["js"] = self.js_parser(js_ids)
        
        return outputs
