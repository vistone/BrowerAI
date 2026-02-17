"""
Online Learning Module - Updates models based on feedback
Processes feedback from rendered websites and updates model weights
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class OnlineLearner:
    """
    Online learning engine for continuous model improvement
    Processes feedback and updates model parameters
    """
    
    def __init__(
        self,
        feature_dim: int = 48,
        latent_dim: int = 256,
        learning_rate: float = 0.001,
        batch_size: int = 32
    ):
        """
        Initialize online learner
        
        Args:
            feature_dim: Input feature dimension (48)
            latent_dim: Latent vector dimension (256)
            learning_rate: Model update learning rate
            batch_size: Batch size for gradient computation
        """
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Model weights (encoding matrix)
        self.encoding_matrix = np.random.randn(feature_dim, latent_dim) * 0.01
        
        # Optimizer state (for Adam-like updates)
        self.m = np.zeros((feature_dim, latent_dim))  # First moment
        self.v = np.zeros((feature_dim, latent_dim))  # Second moment
        self.t = 0  # Timestep
        
        # Training statistics
        self.training_losses: List[float] = []
        self.quality_scores: List[float] = []
        self.update_count = 0
        self.feedback_count = 0
        self.last_update = datetime.now()
        
        logger.info(
            f"OnlineLearner initialized: "
            f"feature_dim={feature_dim}, latent_dim={latent_dim}, "
            f"lr={learning_rate}, batch_size={batch_size}"
        )
    
    def process_feedback(
        self,
        features: np.ndarray,
        generated_latent: np.ndarray,
        feedback_data: Dict[str, Any],
        session_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Process feedback and update model
        
        Args:
            features: 48-dimensional feature vector
            generated_latent: Generated 256-dimensional latent vector
            feedback_data: Quality assessment feedback
            session_id: Session identifier
        
        Returns:
            Dictionary with update metrics
        """
        try:
            # Extract quality scores from feedback
            quality_score = feedback_data.get("quality_score", 0.5)
            html_score = feedback_data.get("html_quality", 0.5)
            css_score = feedback_data.get("css_quality", 0.5)
            js_score = feedback_data.get("js_quality", 0.5)
            
            # Compute loss
            loss = self._compute_loss(
                features,
                generated_latent,
                quality_score,
                html_score,
                css_score,
                js_score
            )
            
            # Update model weights if loss is significant
            if loss > 0.1:
                gradients = self._compute_gradients(
                    features,
                    generated_latent,
                    quality_score
                )
                self._update_weights(gradients)
                self.update_count += 1
            
            # Track metrics
            self.training_losses.append(loss)
            self.quality_scores.append(quality_score)
            self.feedback_count += 1
            self.last_update = datetime.now()
            
            # Compute convergence metrics
            convergence = self._compute_convergence()
            improvement = self._compute_improvement()
            
            logger.info(
                f"Feedback processed (session={session_id}): "
                f"loss={loss:.4f}, quality={quality_score:.3f}, "
                f"updates={self.update_count}, convergence={convergence:.3f}"
            )
            
            return {
                "loss": float(loss),
                "quality_score": float(quality_score),
                "weights_updated": loss > 0.1,
                "update_count": self.update_count,
                "feedback_count": self.feedback_count,
                "convergence": float(convergence),
                "improvement": float(improvement),
                "session_id": session_id,
            }
        
        except Exception as e:
            logger.error(f"Feedback processing error: {e}")
            raise
    
    def _compute_loss(
        self,
        features: np.ndarray,
        generated_latent: np.ndarray,
        quality_score: float,
        html_score: float,
        css_score: float,
        js_score: float
    ) -> float:
        """Compute loss based on feedback"""
        
        # Reconstruction loss: how well features map to latent
        expected_latent = features @ self.encoding_matrix
        reconstruction_error = np.mean((expected_latent - generated_latent) ** 2)
        
        # Quality loss: inverse of quality score
        quality_loss = 1.0 - quality_score
        
        # Component losses
        component_loss = (
            (1.0 - html_score) * 0.35 +
            (1.0 - css_score) * 0.35 +
            (1.0 - js_score) * 0.30
        )
        
        # Combined loss
        total_loss = (
            reconstruction_error * 0.3 +
            quality_loss * 0.4 +
            component_loss * 0.3
        )
        
        return float(np.clip(total_loss, 0.0, 1.0))
    
    def _compute_gradients(
        self,
        features: np.ndarray,
        generated_latent: np.ndarray,
        quality_score: float
    ) -> np.ndarray:
        """Compute gradients for weight updates"""
        
        # Reconstruction gradient
        expected_latent = features @ self.encoding_matrix
        error = expected_latent - generated_latent
        
        # Outer product gradient
        gradient = np.outer(features, error)
        
        # Scale by quality (lower quality = larger update)
        quality_scale = 1.0 - quality_score
        gradient = gradient * quality_scale
        
        return gradient
    
    def _update_weights(self, gradients: np.ndarray) -> None:
        """Update model weights using Adam-like optimizer"""
        
        self.t += 1
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        
        # Adam update rule
        self.m = beta1 * self.m + (1 - beta1) * gradients
        self.v = beta2 * self.v + (1 - beta2) * (gradients ** 2)
        
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        
        # Weight update
        self.encoding_matrix -= (
            self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        
        if not self.training_losses:
            return {
                "average_loss": 0.0,
                "average_quality": 0.5,
                "best_loss": 0.0,
                "update_count": 0,
                "feedback_count": 0,
                "convergence": 0.0,
                "improvement": 0.0,
            }
        
        losses = np.array(self.training_losses)
        qualities = np.array(self.quality_scores)
        
        # Recent window (last 100 samples)
        window = min(100, len(losses))
        recent_losses = losses[-window:]
        recent_qualities = qualities[-window:]
        
        convergence = self._compute_convergence()
        improvement = self._compute_improvement()
        
        return {
            "average_loss": float(np.mean(losses)),
            "recent_average_loss": float(np.mean(recent_losses)),
            "best_loss": float(np.min(losses)),
            "worst_loss": float(np.max(losses)),
            "loss_std": float(np.std(losses)),
            "average_quality": float(np.mean(qualities)),
            "recent_average_quality": float(np.mean(recent_qualities)),
            "best_quality": float(np.max(qualities)),
            "worst_quality": float(np.min(qualities)),
            "quality_std": float(np.std(qualities)),
            "update_count": self.update_count,
            "feedback_count": self.feedback_count,
            "convergence": float(convergence),
            "improvement": float(improvement),
            "learning_rate": self.learning_rate,
            "weight_matrix_norm": float(np.linalg.norm(self.encoding_matrix)),
            "weight_matrix_mean": float(np.mean(np.abs(self.encoding_matrix))),
        }
    
    def _compute_convergence(self) -> float:
        """Compute convergence metric"""
        
        if len(self.training_losses) < 10:
            return 0.0
        
        # Use recent losses to compute convergence
        recent = np.array(self.training_losses[-10:])
        
        # Convergence: how stable are recent losses
        mean_recent = np.mean(recent)
        std_recent = np.std(recent)
        
        if mean_recent < 1e-6:
            return 1.0
        
        convergence = 1.0 / (1.0 + std_recent / mean_recent)
        return float(np.clip(convergence, 0.0, 1.0))
    
    def _compute_improvement(self) -> float:
        """Compute improvement metric"""
        
        if len(self.training_losses) < 2:
            return 0.0
        
        # Compare recent losses to early losses
        early = np.mean(self.training_losses[:min(10, len(self.training_losses))])
        recent = np.mean(self.training_losses[-min(10, len(self.training_losses)):])
        
        if early < 1e-6:
            return 0.0
        
        improvement = (early - recent) / early
        return float(np.clip(improvement, -1.0, 1.0))
    
    def set_learning_rate(self, lr: float) -> None:
        """Dynamically adjust learning rate"""
        if lr <= 0:
            raise ValueError("Learning rate must be positive")
        
        old_lr = self.learning_rate
        self.learning_rate = lr
        
        logger.info(f"Learning rate adjusted: {old_lr:.6f} → {lr:.6f}")
    
    def adaptive_learning_rate(self) -> None:
        """Automatically adjust learning rate based on convergence"""
        
        convergence = self._compute_convergence()
        
        if convergence > 0.8:
            # Converged: reduce learning rate
            new_lr = self.learning_rate * 0.95
        elif convergence < 0.3:
            # Diverging: increase learning rate
            new_lr = self.learning_rate * 1.05
        else:
            # Normal: keep learning rate
            return
        
        self.set_learning_rate(new_lr)
    
    def reset_statistics(self) -> None:
        """Reset training statistics"""
        self.training_losses.clear()
        self.quality_scores.clear()
        self.update_count = 0
        self.feedback_count = 0
        
        logger.info("Training statistics reset")
    
    def get_weight_summary(self) -> Dict[str, Any]:
        """Get weight matrix summary"""
        return {
            "shape": self.encoding_matrix.shape,
            "dtype": str(self.encoding_matrix.dtype),
            "norm": float(np.linalg.norm(self.encoding_matrix)),
            "mean": float(np.mean(self.encoding_matrix)),
            "std": float(np.std(self.encoding_matrix)),
            "min": float(np.min(self.encoding_matrix)),
            "max": float(np.max(self.encoding_matrix)),
            "sparsity": float(np.mean(np.abs(self.encoding_matrix) < 0.001)),
        }


class FeedbackBuffer:
    """
    Buffer for accumulating feedback before batch updates
    Enables batch processing of multiple feedback samples
    """
    
    def __init__(self, batch_size: int = 32, max_buffer_size: int = 1000):
        """
        Initialize feedback buffer
        
        Args:
            batch_size: Size for batch processing
            max_buffer_size: Maximum buffer size before forcing flush
        """
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_timestamp = datetime.now()
        
        logger.info(
            f"FeedbackBuffer initialized: "
            f"batch_size={batch_size}, max_size={max_buffer_size}"
        )
    
    def add(self, feedback: Dict[str, Any]) -> bool:
        """
        Add feedback to buffer
        
        Args:
            feedback: Feedback dictionary
        
        Returns:
            True if buffer is ready for processing
        """
        self.buffer.append(feedback)
        
        # Return True if buffer is full or reached max size
        is_full = len(self.buffer) >= self.batch_size
        is_overflow = len(self.buffer) >= self.max_buffer_size
        
        return is_full or is_overflow
    
    def get_batch(self) -> List[Dict[str, Any]]:
        """Get batch of feedback for processing"""
        batch = self.buffer[:self.batch_size]
        self.buffer = self.buffer[self.batch_size:]
        return batch
    
    def flush(self) -> List[Dict[str, Any]]:
        """Flush all remaining feedback"""
        batch = self.buffer.copy()
        self.buffer.clear()
        return batch
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear buffer"""
        self.buffer.clear()
