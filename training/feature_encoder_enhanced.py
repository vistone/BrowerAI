"""
Enhanced Feature Encoder Module with Non-linear Activation & Learnable Embeddings
Converts 48-dimensional Rust features to 256-dimensional latent with improved expressiveness

Key improvements:
1. Non-linear layers: ReLU + GELU activations for better feature representation
2. Learnable embeddings: intent & style embeddings can be trained
3. Anomaly detection: detects NaN/Inf/outliers in features
4. Layer normalization: improves training stability
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects anomalies in feature vectors using statistical methods"""
    
    def __init__(self, history_size: int = 100):
        """
        Initialize anomaly detector
        
        Args:
            history_size: Number of recent samples to track for statistics
        """
        self.history = deque(maxlen=history_size)
        self.anomalies_detected = 0
        self.last_check_time = datetime.now()
    
    def detect_numeric_anomalies(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Detect NaN, Inf, and extreme values
        
        Args:
            features: Feature vector
        
        Returns:
            Dictionary with anomaly detection results
        """
        anomalies = {
            'has_nan': False,
            'has_inf': False,
            'has_extreme': False,
            'nan_indices': [],
            'inf_indices': [],
            'extreme_indices': [],
            'is_healthy': True
        }
        
        # Check for NaN
        nan_mask = np.isnan(features)
        if np.any(nan_mask):
            anomalies['has_nan'] = True
            anomalies['nan_indices'] = np.where(nan_mask)[0].tolist()
            anomalies['is_healthy'] = False
        
        # Check for Inf
        inf_mask = np.isinf(features)
        if np.any(inf_mask):
            anomalies['has_inf'] = True
            anomalies['inf_indices'] = np.where(inf_mask)[0].tolist()
            anomalies['is_healthy'] = False
        
        # Check for extreme values (> 100 or < -100)
        extreme_mask = (np.abs(features) > 100)
        if np.any(extreme_mask):
            anomalies['has_extreme'] = True
            anomalies['extreme_indices'] = np.where(extreme_mask)[0].tolist()
        
        return anomalies
    
    def detect_statistical_anomalies(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Detect statistical anomalies using IQR method
        
        Args:
            features: Feature vector
        
        Returns:
            Dictionary with statistical anomaly results
        """
        self.history.append(features)
        
        if len(self.history) < 10:
            return {
                'is_anomaly': False,
                'outlier_indices': [],
                'reason': 'insufficient_history'
            }
        
        # Calculate statistics from history
        historical_data = np.array(list(self.history))
        
        results = {
            'is_anomaly': False,
            'outlier_indices': [],
            'outlier_values': []
        }
        
        # Check each feature for outliers using IQR
        for i in range(features.shape[0]):
            feature_history = historical_data[:, i]
            
            Q1 = np.percentile(feature_history, 25)
            Q3 = np.percentile(feature_history, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Check if current feature is outlier
            if features[i] < lower_bound or features[i] > upper_bound:
                results['is_anomaly'] = True
                results['outlier_indices'].append(i)
                results['outlier_values'].append(float(features[i]))
        
        if results['is_anomaly']:
            self.anomalies_detected += 1
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get anomaly detection statistics"""
        return {
            'total_checks': len(self.history),
            'anomalies_detected': self.anomalies_detected,
            'detection_rate': self.anomalies_detected / max(1, len(self.history)),
            'last_check': self.last_check_time.isoformat()
        }


class LayerNormalization:
    """Layer normalization for improved training stability"""
    
    def __init__(self, feature_dim: int, epsilon: float = 1e-6):
        """
        Initialize layer normalization
        
        Args:
            feature_dim: Dimension of features to normalize
            epsilon: Small value to prevent division by zero
        """
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        
        # Learnable parameters (could be trained)
        self.gamma = np.ones(feature_dim)  # Scale
        self.beta = np.zeros(feature_dim)  # Shift
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization
        
        Args:
            x: Input vector of shape (feature_dim,)
        
        Returns:
            Normalized vector
        """
        mean = np.mean(x)
        var = np.var(x)
        
        x_normalized = (x - mean) / np.sqrt(var + self.epsilon)
        
        # Apply learnable scale and shift
        out = self.gamma * x_normalized + self.beta
        
        return out
    
    def update_parameters(self, gamma: np.ndarray, beta: np.ndarray):
        """Update normalization parameters during training"""
        if gamma.shape != self.gamma.shape:
            raise ValueError(f"Gamma shape mismatch: {gamma.shape} vs {self.gamma.shape}")
        if beta.shape != self.beta.shape:
            raise ValueError(f"Beta shape mismatch: {beta.shape} vs {self.beta.shape}")
        
        self.gamma = gamma
        self.beta = beta


class NonLinearActivation:
    """Non-linear activation functions for improved expressiveness"""
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x)"""
        return np.maximum(x, 0)
    
    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """
        GELU activation: x * Φ(x)
        Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        """
        cdf = 0.5 * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
        ))
        return x * cdf
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation: max(alpha*x, x)"""
        return np.maximum(alpha * x, x)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation"""
        return np.tanh(x)


class EnhancedFeatureEncoder:
    """
    Enhanced feature encoder with non-linear layers and learnable embeddings
    
    Architecture:
    - Input: 48-dimensional raw features
    - Layer 1: Linear (48 → 128) + LayerNorm + ReLU
    - Layer 2: Linear (128 → 256) + LayerNorm + GELU
    - Intent embedding: Learnable (48 intent types)
    - Style embedding: Learnable (7 styles)
    - Output: 256-dimensional latent vector
    """
    
    def __init__(
        self,
        feature_dim: int = 48,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        enable_anomaly_detection: bool = True
    ):
        """
        Initialize enhanced feature encoder
        
        Args:
            feature_dim: Input feature dimension (48)
            hidden_dim: Hidden layer dimension (128)
            latent_dim: Output latent dimension (256)
            enable_anomaly_detection: Whether to detect anomalies
        """
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.enable_anomaly_detection = enable_anomaly_detection
        
        np.random.seed(42)
        
        # Layer 1: Feature → Hidden (48 → 128)
        self.W1 = np.random.randn(feature_dim, hidden_dim) * np.sqrt(2.0 / feature_dim)
        self.b1 = np.zeros(hidden_dim)
        
        # Layer 2: Hidden → Latent (128 → 256)
        self.W2 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(latent_dim)
        
        # Layer normalization for each layer
        self.ln1 = LayerNormalization(hidden_dim)
        self.ln2 = LayerNormalization(latent_dim)
        
        # Learnable intent embeddings (8 intent types)
        self.intent_types = [
            "blog", "ecommerce", "documentation", "portfolio",
            "landing", "social", "news", "unknown"
        ]
        self.intent_embeddings = {
            intent: np.random.randn(latent_dim) * 0.05
            for intent in self.intent_types
        }
        self.intent_learnable = True
        
        # Learnable style embeddings (7 styles)
        self.style_types = [
            "modern", "minimal", "classic", "playful",
            "professional", "creative", "unknown"
        ]
        self.style_embeddings = {
            style: np.random.randn(latent_dim) * 0.05
            for style in self.style_types
        }
        self.style_learnable = True
        
        # Anomaly detection
        self.anomaly_detector = AnomalyDetector() if enable_anomaly_detection else None
        
        # Statistics tracking
        self.encoding_count = 0
        self.anomalies_found = 0
        self.skipped_encodings = 0
        self.last_latent_norm = None
        
        logger.info(
            f"EnhancedFeatureEncoder initialized: "
            f"{feature_dim}D → {hidden_dim}D → {latent_dim}D "
            f"with learnable embeddings and anomaly detection"
        )
    
    def encode(
        self,
        features: List[float],
        intent: str = "unknown",
        design_style: str = "unknown",
        skip_on_anomaly: bool = True
    ) -> Dict[str, Any]:
        """
        Encode feature vector to latent space with full diagnostics
        
        Args:
            features: 48-dimensional feature vector
            intent: Website intent type
            design_style: Design style
            skip_on_anomaly: Whether to skip encoding if anomalies detected
        
        Returns:
            Dictionary with latent vector and diagnostics
        """
        self.encoding_count += 1
        
        try:
            # Convert to numpy array
            feature_vec = np.array(features, dtype=np.float32)
            
            if len(feature_vec) != self.feature_dim:
                raise ValueError(
                    f"Expected {self.feature_dim} features, "
                    f"got {len(feature_vec)}"
                )
            
            # Step 1: Anomaly detection
            anomaly_result = {'numeric': {}, 'statistical': {}, 'is_healthy': True}
            
            if self.enable_anomaly_detection:
                numeric_anom = self.anomaly_detector.detect_numeric_anomalies(feature_vec)
                anomaly_result['numeric'] = numeric_anom
                
                if not numeric_anom['is_healthy']:
                    self.anomalies_found += 1
                    if skip_on_anomaly:
                        self.skipped_encodings += 1
                        return {
                            'latent': None,
                            'confidence': 0.0,
                            'anomaly_detected': True,
                            'reason': 'Numeric anomalies detected',
                            'details': numeric_anom
                        }
                
                stat_anom = self.anomaly_detector.detect_statistical_anomalies(feature_vec)
                anomaly_result['statistical'] = stat_anom
            
            # Step 2: Feature normalization
            feature_vec_norm = self._normalize_features(feature_vec)
            
            # Step 3: Layer 1 (48 → 128) + ReLU + LayerNorm
            hidden = feature_vec_norm @ self.W1 + self.b1
            hidden = NonLinearActivation.relu(hidden)
            hidden = self.ln1.normalize(hidden)
            
            # Step 4: Layer 2 (128 → 256) + GELU + LayerNorm
            latent = hidden @ self.W2 + self.b2
            latent = NonLinearActivation.gelu(latent)
            latent = self.ln2.normalize(latent)
            
            # Step 5: Add learnable embeddings
            intent_key = intent.lower() if intent.lower() in self.intent_embeddings else "unknown"
            intent_emb = self.intent_embeddings[intent_key]
            latent = latent + intent_emb * 0.25
            
            style_key = design_style.lower() if design_style.lower() in self.style_embeddings else "unknown"
            style_emb = self.style_embeddings[style_key]
            latent = latent + style_emb * 0.15
            
            # Step 6: Final normalization
            latent = self._normalize_latent(latent)
            self.last_latent_norm = np.linalg.norm(latent)
            
            # Confidence based on layer norms
            confidence = min(1.0, self.last_latent_norm / 0.5)
            
            logger.debug(
                f"Encoded: intent={intent_key}, style={style_key}, "
                f"latent_norm={self.last_latent_norm:.4f}, "
                f"anomalies={anomaly_result['numeric'].get('is_healthy', True)}"
            )
            
            return {
                'latent': latent,
                'confidence': confidence,
                'anomaly_detected': False,
                'latent_norm': self.last_latent_norm,
                'intent': intent_key,
                'style': style_key,
                'stats': self.get_statistics()
            }
        
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            self.skipped_encodings += 1
            return {
                'latent': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using z-score normalization
        More stable than min-max for ML training
        """
        mean = np.mean(features)
        std = np.std(features)
        
        if std > 1e-6:
            return (features - mean) / std
        else:
            return np.zeros_like(features)
    
    def _normalize_latent(self, latent: np.ndarray) -> np.ndarray:
        """L2 normalization for latent vector"""
        norm = np.linalg.norm(latent)
        if norm > 1e-6:
            return latent / norm
        return latent
    
    def update_weights(
        self,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray
    ) -> bool:
        """
        Update encoder weights during training
        
        Returns:
            True if update successful, False otherwise
        """
        try:
            if W1.shape != (self.feature_dim, self.hidden_dim):
                raise ValueError(f"W1 shape mismatch: {W1.shape}")
            if b1.shape != (self.hidden_dim,):
                raise ValueError(f"b1 shape mismatch: {b1.shape}")
            if W2.shape != (self.hidden_dim, self.latent_dim):
                raise ValueError(f"W2 shape mismatch: {W2.shape}")
            if b2.shape != (self.latent_dim,):
                raise ValueError(f"b2 shape mismatch: {b2.shape}")
            
            self.W1 = W1
            self.b1 = b1
            self.W2 = W2
            self.b2 = b2
            
            logger.info("Encoder weights updated successfully")
            return True
        
        except Exception as e:
            logger.error(f"Weight update failed: {e}")
            return False
    
    def update_embeddings(
        self,
        intent_embeddings: Dict[str, np.ndarray] = None,
        style_embeddings: Dict[str, np.ndarray] = None
    ) -> bool:
        """
        Update learnable embeddings
        
        Returns:
            True if update successful
        """
        try:
            if intent_embeddings:
                self.intent_embeddings = intent_embeddings
            if style_embeddings:
                self.style_embeddings = style_embeddings
            
            logger.info("Embeddings updated successfully")
            return True
        
        except Exception as e:
            logger.error(f"Embedding update failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get encoder statistics"""
        return {
            'total_encodings': self.encoding_count,
            'anomalies_found': self.anomalies_found,
            'skipped_encodings': self.skipped_encodings,
            'detection_rate': self.anomalies_found / max(1, self.encoding_count),
            'last_latent_norm': float(self.last_latent_norm) if self.last_latent_norm else None,
            'anomaly_stats': self.anomaly_detector.get_statistics() if self.anomaly_detector else {}
        }
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        """Get statistics about encoder weights"""
        return {
            'W1_norm': float(np.linalg.norm(self.W1)),
            'W1_mean': float(np.mean(np.abs(self.W1))),
            'W1_max': float(np.max(np.abs(self.W1))),
            'W2_norm': float(np.linalg.norm(self.W2)),
            'W2_mean': float(np.mean(np.abs(self.W2))),
            'W2_max': float(np.max(np.abs(self.W2))),
            'b1_norm': float(np.linalg.norm(self.b1)),
            'b2_norm': float(np.linalg.norm(self.b2)),
            'intent_embeddings_mean_norm': float(
                np.mean([np.linalg.norm(emb) for emb in self.intent_embeddings.values()])
            ),
            'style_embeddings_mean_norm': float(
                np.mean([np.linalg.norm(emb) for emb in self.style_embeddings.values()])
            )
        }
    
    def compare_with_baseline(self, baseline_encoder: 'FeatureEncoder', sample_features: List[float]) -> Dict[str, Any]:
        """
        Compare enhanced encoder with baseline encoder
        
        Args:
            baseline_encoder: Original FeatureEncoder instance
            sample_features: Sample features for comparison
        
        Returns:
            Comparison results
        """
        # Encode with baseline
        baseline_latent = baseline_encoder.encode(sample_features)
        
        # Encode with enhanced
        enhanced_result = self.encode(sample_features)
        enhanced_latent = enhanced_result.get('latent')
        
        if enhanced_latent is None:
            return {'error': 'Enhanced encoding failed'}
        
        # Compare
        latent_diff = np.linalg.norm(baseline_latent - enhanced_latent)
        diversity_score = 1.0 - (latent_diff / (np.linalg.norm(baseline_latent) + np.linalg.norm(enhanced_latent) + 1e-6))
        
        return {
            'baseline_norm': float(np.linalg.norm(baseline_latent)),
            'enhanced_norm': float(np.linalg.norm(enhanced_latent)),
            'latent_difference': float(latent_diff),
            'diversity_score': float(diversity_score),
            'improved': enhanced_result.get('confidence', 0) > 0.8
        }
