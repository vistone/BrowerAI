"""
Holistic Website Learning System

Learn websites as complete systems, not isolated HTML/CSS/JS.
Understands:
- Website intent and category
- Loading order and dependencies
- Code style and fingerprints
- Device adaptation strategies
- Company-specific patterns
- Technology stacks

Inspired by:
- Multi-modal learning (CLIP, ALIGN)
- Graph neural networks for dependency modeling
- Meta-learning for style adaptation
- Computer vision scene understanding
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .base import BaseModel
from .transformer import TransformerEncoder
from .attention import MultiHeadAttention


class WebsiteIntentClassifier(nn.Module):
    """
    Classify website intent/category.
    
    Categories:
        - E-commerce (shopping)
        - News/Media
        - Education/Learning
        - Entertainment (video, games)
        - Social Media
        - Business/Corporate
        - Government/Public
        - Personal/Blog
        - Documentation/Reference
        - Tools/Services
    """
    
    CATEGORIES = [
        "ecommerce",
        "news",
        "education",
        "entertainment",
        "social",
        "business",
        "government",
        "personal",
        "documentation",
        "tools"
    ]
    
    def __init__(self, d_model: int, num_categories: int = 10, url_feature_dim: int = 128):
        super().__init__()
        
        self.num_categories = num_categories
        
        # Multi-level feature extraction  
        self.url_analyzer = nn.Sequential(
            nn.Linear(url_feature_dim, 256),  # URL features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        
        self.content_analyzer = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
        self.structure_analyzer = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(128 + 256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_categories)
        )
        
    def forward(self, url_features: torch.Tensor, content_features: torch.Tensor,
                structure_features: torch.Tensor) -> torch.Tensor:
        """
        Classify website intent from multiple signals.
        
        Args:
            url_features: URL patterns (batch_size, 128)
            content_features: Content embeddings (batch_size, d_model)
            structure_features: Structure embeddings (batch_size, d_model)
            
        Returns:
            Category logits (batch_size, num_categories)
        """
        url_repr = self.url_analyzer(url_features)
        content_repr = self.content_analyzer(content_features)
        structure_repr = self.structure_analyzer(structure_features)
        
        # Concatenate all features
        combined = torch.cat([url_repr, content_repr, structure_repr], dim=-1)
        
        # Classify
        logits = self.fusion(combined)
        
        return logits


class CodeStyleAnalyzer(nn.Module):
    """
    Analyze and recognize code style and fingerprints.
    
    Recognizes:
        - Framework signatures (React, Vue, Angular, etc.)
        - Build tool patterns (Webpack, Rollup, Vite)
        - Minification styles (UglifyJS, Terser, Closure)
        - Company-specific patterns (Google, Facebook, Amazon style guides)
        - Obfuscation techniques
        - Code generation patterns (TypeScript, Babel transforms)
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # JS style analysis
        self.js_style_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=2048, batch_first=True),
            num_layers=4
        )
        
        # CSS style analysis
        self.css_style_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model // 2, nhead=4, dim_feedforward=1024, batch_first=True),
            num_layers=3
        )
        
        # HTML patterns
        self.html_pattern_analyzer = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Style fingerprint extractor
        self.fingerprint_extractor = nn.Sequential(
            nn.Linear(d_model + d_model // 2 + 256, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)  # 256-dim fingerprint
        )
        
        # Framework classifier
        self.framework_classifier = nn.Linear(256, 50)  # 50 common frameworks
        
        # Build tool classifier
        self.build_tool_classifier = nn.Linear(256, 20)  # 20 build tools
        
        # Company style classifier
        self.company_style_classifier = nn.Linear(256, 30)  # 30 company styles
        
    def forward(self, js_tokens: torch.Tensor, css_tokens: torch.Tensor,
                html_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze code style across all components.
        
        Returns:
            Dict containing:
                - fingerprint: Unique style embedding
                - framework: Framework probabilities
                - build_tool: Build tool probabilities
                - company_style: Company style probabilities
        """
        # Encode each component
        js_repr = self.js_style_encoder(js_tokens).mean(dim=1)
        css_repr = self.css_style_encoder(css_tokens).mean(dim=1)
        html_repr = self.html_pattern_analyzer(html_features)
        
        # Combine into fingerprint
        combined = torch.cat([js_repr, css_repr, html_repr], dim=-1)
        fingerprint = self.fingerprint_extractor(combined)
        
        # Classify different aspects
        framework_logits = self.framework_classifier(fingerprint)
        build_tool_logits = self.build_tool_classifier(fingerprint)
        company_style_logits = self.company_style_classifier(fingerprint)
        
        return {
            "fingerprint": fingerprint,
            "framework": framework_logits,
            "build_tool": build_tool_logits,
            "company_style": company_style_logits
        }


class DependencyGraphLearner(nn.Module):
    """
    Learn file dependencies and loading order.
    
    Models:
        - Script loading order (async, defer, blocking)
        - CSS dependencies and imports
        - Resource dependencies (images, fonts)
        - Dynamic imports and code splitting
        - Module relationships
    
    Uses Graph Neural Networks (GNN) to model dependency graph.
    """
    
    def __init__(self, d_model: int, num_layers: int = 3):
        super().__init__()
        
        self.d_model = d_model
        
        # Node embedding (for each file/resource)
        self.node_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        
        # Loading order predictor
        self.order_predictor = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Relative order score
        )
        
        # Dependency type classifier
        self.dependency_classifier = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # Import, script, style, resource, dynamic
        )
        
    def forward(self, node_features: torch.Tensor, 
                adjacency_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process dependency graph.
        
        Args:
            node_features: (batch_size, num_nodes, d_model)
            adjacency_matrix: (batch_size, num_nodes, num_nodes)
            
        Returns:
            Dict containing:
                - node_embeddings: Updated node representations
                - loading_order: Predicted loading order
                - dependency_types: Types of dependencies
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Encode nodes
        x = self.node_encoder(node_features)
        
        # Apply graph attention layers
        for gat, norm in zip(self.gat_layers, self.norms):
            # Mask based on adjacency
            attn_mask = (adjacency_matrix == 0).float() * -1e9
            
            # Self-attention with graph structure
            attn_output, _ = gat(x, x, x, attn_mask=attn_mask)
            x = norm(x + attn_output)
        
        # Predict pairwise relationships
        # Create all pairs
        node_i = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # (B, N, N, D)
        node_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # (B, N, N, D)
        pairs = torch.cat([node_i, node_j], dim=-1)  # (B, N, N, 2D)
        
        # Predict loading order
        loading_scores = self.order_predictor(pairs).squeeze(-1)  # (B, N, N)
        
        # Predict dependency types
        dependency_types = self.dependency_classifier(pairs)  # (B, N, N, 5)
        
        return {
            "node_embeddings": x,
            "loading_order": loading_scores,
            "dependency_types": dependency_types
        }


class DeviceAdaptationAnalyzer(nn.Module):
    """
    Analyze how websites adapt to different devices.
    
    Learns:
        - Responsive design patterns
        - Media query strategies
        - Mobile-first vs desktop-first
        - Touch vs mouse interactions
        - Viewport handling
        - Progressive enhancement
        - Adaptive loading (lazy load, code splitting)
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # Analyze responsive CSS
        self.responsive_analyzer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=1024, batch_first=True),
            num_layers=3
        )
        
        # Device strategy classifier
        self.strategy_classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # Different adaptation strategies
        )
        
        # Breakpoint detector
        self.breakpoint_detector = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # Common breakpoints
        )
        
        # Interaction model classifier
        self.interaction_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # Touch-only, mouse-only, hybrid, adaptive
        )
        
    def forward(self, css_features: torch.Tensor, 
                html_features: torch.Tensor,
                js_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze device adaptation strategy.
        """
        # Combine all features
        combined = css_features + html_features + js_features
        
        # Analyze responsive patterns
        responsive_repr = self.responsive_analyzer(combined.unsqueeze(1)).squeeze(1)
        
        return {
            "strategy": self.strategy_classifier(responsive_repr),
            "breakpoints": self.breakpoint_detector(responsive_repr),
            "interaction_model": self.interaction_classifier(responsive_repr)
        }


class HolisticWebsiteLearner(BaseModel):
    """
    Complete website understanding system.
    
    Learns websites as integrated systems, capturing:
        1. Overall intent and purpose
        2. Technology stack and patterns
        3. Code style and fingerprints
        4. File dependencies and loading
        5. Device adaptation strategies
        6. Company-specific conventions
    
    This is a multi-modal, multi-task model that processes
    entire websites holistically rather than individual components.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        vocab_size = config["vocab_size"]
        d_model = config.get("d_model", 512)
        
        # Shared embedding for all code types
        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Component encoders (lightweight, share parameters where possible)
        self.html_encoder = TransformerEncoder({
            **config,
            "d_model": d_model,
            "num_layers": 4,
            "num_classes": d_model  # Output embedding
        })
        
        self.css_encoder = TransformerEncoder({
            **config,
            "d_model": d_model // 2,
            "num_layers": 3,
            "num_classes": d_model // 2
        })
        
        self.js_encoder = TransformerEncoder({
            **config,
            "d_model": d_model,
            "num_layers": 6,
            "num_classes": d_model
        })
        
        # Cross-modal fusion (learn relationships between HTML/CSS/JS)
        self.cross_modal_attention = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads=8)
            for _ in range(3)
        ])
        
        # High-level analyzers
        self.intent_classifier = WebsiteIntentClassifier(d_model, url_feature_dim=128)
        self.style_analyzer = CodeStyleAnalyzer(d_model)
        self.dependency_learner = DependencyGraphLearner(d_model)
        self.device_analyzer = DeviceAdaptationAnalyzer(d_model)
        
        # Global website representation
        self.website_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model * 2, nhead=16, dim_feedforward=4096, batch_first=True),
            num_layers=4
        )
        
        # Final embedding projection
        self.website_projection = nn.Sequential(
            nn.Linear(d_model * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512)  # Final website embedding
        )
        
    def forward(self, html_ids: torch.Tensor, css_ids: torch.Tensor, 
                js_ids: torch.Tensor, url_features: torch.Tensor,
                adjacency_matrix: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process complete website.
        
        Args:
            html_ids: HTML token IDs (batch_size, html_len)
            css_ids: CSS token IDs (batch_size, css_len)
            js_ids: JS token IDs (batch_size, js_len)
            url_features: URL embeddings (batch_size, 128)
            adjacency_matrix: File dependency graph (batch_size, num_files, num_files)
            
        Returns:
            Comprehensive website understanding
        """
        batch_size = html_ids.size(0)
        
        # Encode each component
        html_repr = self.html_encoder(html_ids)  # (B, D)
        css_repr = self.css_encoder(css_ids)     # (B, D/2)
        js_repr = self.js_encoder(js_ids)        # (B, D)
        
        # Pad CSS to match dimension
        css_repr = torch.cat([css_repr, torch.zeros_like(css_repr)], dim=-1)
        
        # Cross-modal fusion (learn HTML-CSS-JS relationships)
        # HTML attends to CSS and JS
        html_css_fusion = self.cross_modal_attention[0](
            html_repr.unsqueeze(1),
            css_repr.unsqueeze(1),
            css_repr.unsqueeze(1)
        ).squeeze(1)
        
        html_js_fusion = self.cross_modal_attention[1](
            html_repr.unsqueeze(1),
            js_repr.unsqueeze(1),
            js_repr.unsqueeze(1)
        ).squeeze(1)
        
        # CSS attends to JS
        css_js_fusion = self.cross_modal_attention[2](
            css_repr.unsqueeze(1),
            js_repr.unsqueeze(1),
            js_repr.unsqueeze(1)
        ).squeeze(1)
        
        # Combine all modalities
        fused_repr = torch.cat([
            html_repr + html_css_fusion + html_js_fusion,
            js_repr + css_js_fusion
        ], dim=-1)  # (B, 2D)
        
        # Global website encoding
        website_repr = self.website_encoder(fused_repr.unsqueeze(1)).squeeze(1)
        website_embedding = self.website_projection(website_repr)
        
        # High-level analysis
        intent_logits = self.intent_classifier(url_features, html_repr, js_repr)
        
        style_analysis = self.style_analyzer(
            js_ids.unsqueeze(-1).float(),  # Simplified for demo
            css_ids.unsqueeze(-1).float(),
            html_repr
        )
        
        device_analysis = self.device_analyzer(css_repr, html_repr, js_repr)
        
        # Dependency analysis (if graph provided)
        dependency_analysis = None
        if adjacency_matrix is not None:
            node_features = torch.stack([html_repr, css_repr, js_repr], dim=1)
            dependency_analysis = self.dependency_learner(node_features, adjacency_matrix)
        
        return {
            # Core representations
            "website_embedding": website_embedding,
            "html_repr": html_repr,
            "css_repr": css_repr,
            "js_repr": js_repr,
            
            # High-level understanding
            "intent": intent_logits,
            "style_fingerprint": style_analysis["fingerprint"],
            "framework": style_analysis["framework"],
            "build_tool": style_analysis["build_tool"],
            "company_style": style_analysis["company_style"],
            
            # Device adaptation
            "adaptation_strategy": device_analysis["strategy"],
            "breakpoints": device_analysis["breakpoints"],
            "interaction_model": device_analysis["interaction_model"],
            
            # Dependencies
            "dependency_graph": dependency_analysis if dependency_analysis else None
        }
    
    def compute_website_similarity(self, embedding1: torch.Tensor, 
                                   embedding2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between two websites.
        
        Useful for:
            - Finding similar websites
            - Detecting template reuse
            - Company style recognition
        """
        return torch.cosine_similarity(embedding1, embedding2, dim=-1)
