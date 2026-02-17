"""
Feature Encoder Module - Encodes 48-dimensional features to latent space
Converts Rust feature vectors into 256-dimensional latent representations
for the code generator to process
"""

import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FeatureEncoder:
    """
    Encodes 48-dimensional website features to latent space
    
    Feature dimensions from Rust:
    [0-9]   : HTML metrics (10)
    [10-17] : CSS metrics (8)
    [18-27] : JavaScript metrics (10)
    [28-35] : Page structure (8)
    [36-42] : Design style (7)
    [43-47] : Complexity metrics (5)
    """
    
    def __init__(self, feature_dim: int = 48, latent_dim: int = 256):
        """
        Initialize feature encoder
        
        Args:
            feature_dim: Input feature dimension (48 from Rust)
            latent_dim: Output latent dimension (256)
        """
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        
        # Initialize encoding weights (learned during training)
        np.random.seed(42)
        self.encoding_matrix = np.random.randn(feature_dim, latent_dim) * 0.01
        self.bias = np.zeros(latent_dim)
        
        # Intent encoding weights
        self.intent_embeddings = {
            "blog": np.random.randn(latent_dim) * 0.1,
            "ecommerce": np.random.randn(latent_dim) * 0.1,
            "documentation": np.random.randn(latent_dim) * 0.1,
            "portfolio": np.random.randn(latent_dim) * 0.1,
            "landing": np.random.randn(latent_dim) * 0.1,
            "social": np.random.randn(latent_dim) * 0.1,
            "news": np.random.randn(latent_dim) * 0.1,
            "unknown": np.random.randn(latent_dim) * 0.1,
        }
        
        # Design style embeddings
        self.style_embeddings = {
            "modern": np.random.randn(latent_dim) * 0.1,
            "minimal": np.random.randn(latent_dim) * 0.1,
            "classic": np.random.randn(latent_dim) * 0.1,
            "playful": np.random.randn(latent_dim) * 0.1,
            "professional": np.random.randn(latent_dim) * 0.1,
            "creative": np.random.randn(latent_dim) * 0.1,
            "unknown": np.random.randn(latent_dim) * 0.1,
        }
        
        logger.info(
            f"FeatureEncoder initialized: "
            f"{feature_dim}-dim → {latent_dim}-dim"
        )
    
    def encode(
        self,
        features: List[float],
        intent: str = "unknown",
        design_style: str = "unknown"
    ) -> np.ndarray:
        """
        Encode feature vector to latent space
        
        Args:
            features: 48-dimensional feature vector from Rust
            intent: Website type/intent (blog, ecommerce, etc.)
            design_style: Design style (modern, minimal, etc.)
        
        Returns:
            256-dimensional latent vector
        """
        try:
            # Convert to numpy array
            feature_vec = np.array(features, dtype=np.float32)
            
            if len(feature_vec) != self.feature_dim:
                raise ValueError(
                    f"Expected {self.feature_dim} features, "
                    f"got {len(feature_vec)}"
                )
            
            # Normalize features
            feature_vec = self._normalize_features(feature_vec)
            
            # Linear encoding: features × encoding_matrix + bias
            latent = feature_vec @ self.encoding_matrix + self.bias
            
            # Add intent embedding
            intent_key = intent.lower() if intent.lower() in self.intent_embeddings else "unknown"
            intent_emb = self.intent_embeddings[intent_key]
            latent += intent_emb * 0.3
            
            # Add design style embedding
            style_key = design_style.lower() if design_style.lower() in self.style_embeddings else "unknown"
            style_emb = self.style_embeddings[style_key]
            latent += style_emb * 0.2
            
            # Apply activation function (ReLU)
            latent = np.maximum(latent, 0)
            
            # Normalize latent space
            latent = self._normalize_latent(latent)
            
            logger.debug(
                f"Encoded features: intent={intent}, "
                f"style={design_style}, latent_norm={np.linalg.norm(latent):.4f}"
            )
            
            return latent
        
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            raise
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature vector"""
        # Min-max normalization to [0, 1]
        feature_min = features.min()
        feature_max = features.max()
        
        if feature_max - feature_min > 1e-6:
            features = (features - feature_min) / (feature_max - feature_min)
        else:
            features = np.ones_like(features) * 0.5
        
        return features
    
    def _normalize_latent(self, latent: np.ndarray) -> np.ndarray:
        """Normalize latent vector"""
        # L2 normalization
        norm = np.linalg.norm(latent)
        if norm > 1e-6:
            latent = latent / norm
        
        return latent
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """
        Decode latent vector back to feature space (for debugging)
        
        Args:
            latent: 256-dimensional latent vector
        
        Returns:
            48-dimensional feature vector
        """
        try:
            # Simple linear decoding (pseudoinverse)
            encoding_pinv = np.linalg.pinv(self.encoding_matrix)
            features = latent @ encoding_pinv.T - self.bias @ encoding_pinv.T
            
            # Normalize to [0, 1]
            features = np.clip(features, 0, 1)
            
            return features
        
        except Exception as e:
            logger.error(f"Decoding error: {e}")
            raise
    
    def get_feature_statistics(self, features: List[float]) -> Dict[str, Any]:
        """Get statistics about the feature vector"""
        feature_vec = np.array(features, dtype=np.float32)
        
        return {
            "count": len(feature_vec),
            "mean": float(feature_vec.mean()),
            "std": float(feature_vec.std()),
            "min": float(feature_vec.min()),
            "max": float(feature_vec.max()),
            "median": float(np.median(feature_vec)),
            "html_metrics_mean": float(feature_vec[0:10].mean()),
            "css_metrics_mean": float(feature_vec[10:18].mean()),
            "js_metrics_mean": float(feature_vec[18:28].mean()),
            "structure_metrics_mean": float(feature_vec[28:36].mean()),
            "design_metrics_mean": float(feature_vec[36:43].mean()),
            "complexity_metrics_mean": float(feature_vec[43:48].mean()),
        }
    
    def update_weights(
        self,
        encoding_matrix: np.ndarray,
        bias: np.ndarray
    ):
        """Update encoder weights (called during training)"""
        if encoding_matrix.shape != self.encoding_matrix.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.encoding_matrix.shape}, "
                f"got {encoding_matrix.shape}"
            )
        
        self.encoding_matrix = encoding_matrix
        self.bias = bias
        logger.info("Encoder weights updated")
