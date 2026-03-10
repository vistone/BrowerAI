"""
Online Learning Module - Updates models based on feedback
Processes feedback from rendered websites and updates model weights

Enhanced with:
- Gradient health checks (NaN/Inf/explosion detection)
- Anomaly feedback detection (IQR method)
- Adaptive loss weight adjustment
- Advanced learning rate scheduling
- Weight constraint checking
"""

import numpy as np
import os
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
from collections import deque
import warnings
import time

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in ("1", "true", "yes", "on")


def _get_cuda_device() -> Optional[str]:
    device_id = os.getenv("BROWERAI_GPU_DEVICE", "0").strip()
    if device_id.isdigit():
        return f"cuda:{device_id}"
    if device_id:
        return device_id
    return "cuda:0"


def _is_torch_tensor(value: Any) -> bool:
    return _TORCH_AVAILABLE and torch is not None and isinstance(value, torch.Tensor)


def _tensor_norm(value: Any) -> float:
    if _is_torch_tensor(value):
        return float(torch.linalg.norm(value).item())
    return float(np.linalg.norm(value))


def _tensor_mean_abs(value: Any) -> float:
    if _is_torch_tensor(value):
        return float(torch.mean(torch.abs(value)).item())
    return float(np.mean(np.abs(value)))


def _tensor_mean(value: Any) -> float:
    if _is_torch_tensor(value):
        return float(torch.mean(value).item())
    return float(np.mean(value))


def _tensor_std(value: Any) -> float:
    if _is_torch_tensor(value):
        return float(torch.std(value, unbiased=False).item())
    return float(np.std(value))


def _tensor_max_abs(value: Any) -> float:
    if _is_torch_tensor(value):
        return float(torch.max(torch.abs(value)).item())
    return float(np.max(np.abs(value)))


def _tensor_min(value: Any) -> float:
    if _is_torch_tensor(value):
        return float(torch.min(value).item())
    return float(np.min(value))


def _tensor_max(value: Any) -> float:
    if _is_torch_tensor(value):
        return float(torch.max(value).item())
    return float(np.max(value))


def _tensor_sparsity(value: Any, threshold: float = 0.001) -> float:
    if _is_torch_tensor(value):
        return float(torch.mean((torch.abs(value) < threshold).float()).item())
    return float(np.mean(np.abs(value) < threshold))


def _amp_autocast(device: str):
    if _TORCH_AVAILABLE and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device)
    return torch.cuda.amp.autocast()


def _copy_to_buffer(buffer: "torch.Tensor", source: np.ndarray) -> "torch.Tensor":
    cpu_tensor = torch.from_numpy(source.astype(np.float32, copy=False))
    buffer.copy_(cpu_tensor)
    return buffer


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
        batch_size: int = 32,
        max_gradient_norm: float = 1.0,
        enable_gradient_clip: bool = True,
        enable_anomaly_detection: bool = True,
        loss_weight_mode: str = "adaptive"  # "fixed" or "adaptive"
    ):
        """
        Initialize online learner with stability enhancements
        
        Args:
            feature_dim: Input feature dimension (48)
            latent_dim: Latent vector dimension (256)
            learning_rate: Model update learning rate
            batch_size: Batch size for gradient computation
            max_gradient_norm: Maximum gradient norm for clipping
            enable_gradient_clip: Enable gradient clipping
            enable_anomaly_detection: Enable anomaly feedback detection
            loss_weight_mode: "fixed" or "adaptive" loss weight adjustment
        """
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_gradient_norm = max_gradient_norm
        self.enable_gradient_clip = enable_gradient_clip
        self.enable_anomaly_detection = enable_anomaly_detection
        self.loss_weight_mode = loss_weight_mode

        self.learning_mode = _env_flag("BROWERAI_LEARNING_MODE")
        self.use_gpu = self.learning_mode and _env_flag("BROWERAI_USE_GPU")
        self.use_amp = self.use_gpu and _env_flag("BROWERAI_GPU_AMP")
        self.micro_batch_size = int(os.getenv("BROWERAI_MICRO_BATCH", "1"))
        self.device = None
        if self.use_gpu:
            if not _TORCH_AVAILABLE:
                logger.warning("GPU learning requested but torch is not available. Using CPU.")
                self.use_gpu = False
            elif not torch.cuda.is_available():
                logger.warning("GPU learning requested but CUDA is not available. Using CPU.")
                self.use_gpu = False
            else:
                self.device = _get_cuda_device()
        logger.info(
            "OnlineLearner learning mode: %s, gpu: %s, device: %s",
            self.learning_mode,
            self.use_gpu,
            self.device or "cpu",
        )
        
        if self.use_gpu:
            self.encoding_matrix = torch.randn(
                feature_dim, latent_dim, device=self.device, dtype=torch.float32
            ) * 0.01
            self.encoding_matrix_initial = self.encoding_matrix.clone()
            self.m = torch.zeros((feature_dim, latent_dim), device=self.device, dtype=torch.float32)
            self.v = torch.zeros((feature_dim, latent_dim), device=self.device, dtype=torch.float32)
            self._feature_buffer = torch.empty((feature_dim,), device=self.device, dtype=torch.float32)
            self._latent_buffer = torch.empty((latent_dim,), device=self.device, dtype=torch.float32)
        else:
            # Model weights (encoding matrix)
            self.encoding_matrix = np.random.randn(feature_dim, latent_dim) * 0.01
            self.encoding_matrix_initial = self.encoding_matrix.copy()
            # Optimizer state (for Adam-like updates)
            self.m = np.zeros((feature_dim, latent_dim))  # First moment
            self.v = np.zeros((feature_dim, latent_dim))  # Second moment
            self._feature_buffer = None
            self._latent_buffer = None

        # Micro-batch accumulation
        self.micro_batch_count = 0
        self.accumulated_gradients = None
        self.accumulated_loss = 0.0
        
        self.t = 0  # Timestep
        
        # Training statistics
        self.training_losses: List[float] = []
        self.quality_scores: List[float] = []
        self.gradient_norms: List[float] = []
        self.weight_change_norms: List[float] = []
        self.update_count = 0
        self.feedback_count = 0
        self.skipped_updates = 0
        self.anomaly_count = 0
        self.last_update = datetime.now()
        
        # Learning rate scheduling
        self.lr_schedule = deque(maxlen=20)
        self.lr_changes = []
        
        # Loss weight adaptation
        self.loss_weights = {
            'reconstruction': 0.3,
            'quality': 0.4,
            'component': 0.3
        }
        self.loss_weight_history = deque(maxlen=50)
        
        # Anomaly detection: quality score statistics
        self.quality_history = deque(maxlen=100)
        self.outlier_threshold = 1.5  # IQR multiplier
        
        # Gradient statistics for monitoring
        self.gradient_history = deque(maxlen=100)
        self.weight_divergence_threshold = 10.0  # Max ||W - W_init|| / ||W_init||
        
        logger.info(
            f"OnlineLearner initialized (enhanced): "
            f"feature_dim={feature_dim}, latent_dim={latent_dim}, "
            f"lr={learning_rate}, max_grad_norm={max_gradient_norm}, "
            f"gradient_clip={enable_gradient_clip}, "
            f"anomaly_detection={enable_anomaly_detection}, "
            f"loss_weight_mode={loss_weight_mode}, "
            f"learning_mode={self.learning_mode}, gpu_learning={self.use_gpu}, amp={self.use_amp}, "
            f"micro_batch={self.micro_batch_size}"
        )
    
    def process_feedback(
        self,
        features: np.ndarray,
        generated_latent: np.ndarray,
        feedback_data: Dict[str, Any],
        session_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Process feedback and update model with stability checks
        
        Args:
            features: 48-dimensional feature vector
            generated_latent: Generated 256-dimensional latent vector
            feedback_data: Quality assessment feedback
            session_id: Session identifier
        
        Returns:
            Dictionary with update metrics
        """
        try:
            gpu_timing = {
                "loss_ms": 0.0,
                "grad_ms": 0.0,
                "update_ms": 0.0,
                "total_ms": 0.0,
            }
            total_start = time.perf_counter()
            # Extract quality scores from feedback
            quality_score = feedback_data.get("quality_score", 0.5)
            html_score = feedback_data.get("html_quality", 0.5)
            css_score = feedback_data.get("css_quality", 0.5)
            js_score = feedback_data.get("js_quality", 0.5)
            
            # Anomaly detection
            is_anomaly = False
            if self.enable_anomaly_detection:
                is_anomaly = self._detect_anomaly_feedback(quality_score, feedback_data)
                if is_anomaly:
                    self.anomaly_count += 1
                    logger.warn(
                        f"Anomaly feedback detected (session={session_id}): "
                        f"quality={quality_score:.3f}, anomaly_count={self.anomaly_count}"
                    )
                    # Skip update but still count feedback
                    self.feedback_count += 1
                    return {
                        "loss": 0.0,
                        "quality_score": float(quality_score),
                        "weights_updated": False,
                        "reason": "anomaly_detected",
                        "anomaly_count": self.anomaly_count,
                        "session_id": session_id,
                    }
            
            # Update adaptive loss weights if enabled
            if self.loss_weight_mode == "adaptive":
                self._update_adaptive_loss_weights()
            
            # Compute loss
            gradients = None
            if self.use_gpu:
                loss, gradients, timing = self._compute_loss_and_gradients(
                    features,
                    generated_latent,
                    quality_score,
                    html_score,
                    css_score,
                    js_score
                )
                gpu_timing["loss_ms"] = timing["loss_ms"]
                gpu_timing["grad_ms"] = timing["grad_ms"]
            else:
                loss_start = time.perf_counter()
                loss = self._compute_loss(
                    features,
                    generated_latent,
                    quality_score,
                    html_score,
                    css_score,
                    js_score
                )
                if self.use_gpu:
                    gpu_timing["loss_ms"] = (time.perf_counter() - loss_start) * 1000.0
            
            # Update model weights if loss is significant
            update_applied = False
            should_accumulate = self.micro_batch_size > 1 and self.use_gpu

            if should_accumulate:
                # Accumulate gradients
                if self.accumulated_gradients is None:
                    self.accumulated_gradients = gradients.clone() if _is_torch_tensor(gradients) else gradients.copy()
                else:
                    self.accumulated_gradients += gradients
                self.accumulated_loss += loss
                self.micro_batch_count += 1

                # Apply update when micro-batch is full
                if self.micro_batch_count >= self.micro_batch_size:
                    avg_gradients = self.accumulated_gradients / self.micro_batch_count
                    avg_loss = self.accumulated_loss / self.micro_batch_count

                    if avg_loss > 0.1:
                        gradient_health = self._check_gradient_health(avg_gradients)
                        if gradient_health['is_healthy']:
                            update_start = time.perf_counter()
                            update_applied = self._update_weights(avg_gradients)
                            if self.use_gpu:
                                gpu_timing["update_ms"] = (time.perf_counter() - update_start) * 1000.0
                            if update_applied:
                                self.update_count += 1
                        else:
                            logger.warn(
                                f"Unhealthy gradient detected: {gradient_health['reason']}"
                            )
                            self.skipped_updates += 1

                    # Reset accumulation
                    self.accumulated_gradients = None
                    self.accumulated_loss = 0.0
                    self.micro_batch_count = 0
            elif loss > 0.1:
                if not self.use_gpu:
                    grad_start = time.perf_counter()
                    gradients = self._compute_gradients(
                        features,
                        generated_latent,
                        quality_score
                    )
                    if self.use_gpu:
                        gpu_timing["grad_ms"] = (time.perf_counter() - grad_start) * 1000.0
                
                # Check gradient health
                gradient_health = self._check_gradient_health(gradients)
                if gradient_health['is_healthy']:
                    update_start = time.perf_counter()
                    update_applied = self._update_weights(gradients)
                    if self.use_gpu:
                        gpu_timing["update_ms"] = (time.perf_counter() - update_start) * 1000.0
                    if update_applied:
                        self.update_count += 1
                else:
                    logger.warn(
                        f"Unhealthy gradient detected: {gradient_health['reason']}"
                    )
                    self.skipped_updates += 1
            
            # Track metrics
            self.training_losses.append(loss)
            self.quality_scores.append(quality_score)
            self.quality_history.append(quality_score)
            self.feedback_count += 1
            self.last_update = datetime.now()
            
            # Compute convergence metrics
            convergence = self._compute_convergence()
            improvement = self._compute_improvement()
            
            # Adaptive learning rate adjustment
            self._adaptive_learning_rate_schedule()
            
            if self.use_gpu:
                gpu_timing["total_ms"] = (time.perf_counter() - total_start) * 1000.0
                logger.info(
                    "GPU timing (session=%s): loss=%.3fms, grad=%.3fms, update=%.3fms, total=%.3fms",
                    session_id,
                    gpu_timing["loss_ms"],
                    gpu_timing["grad_ms"],
                    gpu_timing["update_ms"],
                    gpu_timing["total_ms"],
                )

            logger.info(
                f"Feedback processed (session={session_id}): "
                f"loss={loss:.4f}, quality={quality_score:.3f}, "
                f"updated={update_applied}, convergence={convergence:.3f}, "
                f"lr={self.learning_rate:.6f}"
            )
            
            return {
                "loss": float(loss),
                "quality_score": float(quality_score),
                "weights_updated": update_applied,
                "update_count": self.update_count,
                "feedback_count": self.feedback_count,
                "skipped_updates": self.skipped_updates,
                "convergence": float(convergence),
                "improvement": float(improvement),
                "learning_rate": float(self.learning_rate),
                "session_id": session_id,
                "gpu_timing_ms": gpu_timing if self.use_gpu else None,
            }
        
        except Exception as e:
            logger.error(f"Feedback processing error: {e}", exc_info=True)
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
        """Compute loss with adaptive weights"""
        
        if self.use_gpu:
            t_features = _copy_to_buffer(self._feature_buffer, features)
            t_generated = _copy_to_buffer(self._latent_buffer, generated_latent)
            if self.use_amp:
                with _amp_autocast("cuda"):
                    expected_latent = t_features @ self.encoding_matrix
                    reconstruction_error = torch.mean((expected_latent - t_generated) ** 2)
            else:
                expected_latent = t_features @ self.encoding_matrix
                reconstruction_error = torch.mean((expected_latent - t_generated) ** 2)
            reconstruction_error = torch.clamp(reconstruction_error, 0.0, 100.0).item()
        else:
            # Reconstruction loss: how well features map to latent
            expected_latent = features @ self.encoding_matrix
            reconstruction_error = np.mean((expected_latent - generated_latent) ** 2)

            # Clip reconstruction error to prevent explosion
            reconstruction_error = np.clip(reconstruction_error, 0.0, 100.0)
        
        # Quality loss: inverse of quality score
        quality_loss = 1.0 - quality_score
        
        # Component losses
        component_loss = (
            (1.0 - html_score) * 0.35 +
            (1.0 - css_score) * 0.35 +
            (1.0 - js_score) * 0.30
        )
        
        # Get current loss weights
        weights = self.loss_weights
        
        # Combined loss
        total_loss = (
            reconstruction_error * weights['reconstruction'] +
            quality_loss * weights['quality'] +
            component_loss * weights['component']
        )
        
        # Clip to [0, 1]
        total_loss = float(np.clip(total_loss, 0.0, 1.0))
        
        return total_loss

    def _compute_loss_and_gradients(
        self,
        features: np.ndarray,
        generated_latent: np.ndarray,
        quality_score: float,
        html_score: float,
        css_score: float,
        js_score: float,
    ) -> Tuple[float, "torch.Tensor", Dict[str, float]]:
        timing = {
            "loss_ms": 0.0,
            "grad_ms": 0.0,
        }

        t_features = _copy_to_buffer(self._feature_buffer, features)
        t_generated = _copy_to_buffer(self._latent_buffer, generated_latent)

        loss_start = time.perf_counter()
        if self.use_amp:
            with _amp_autocast("cuda"):
                expected_latent = t_features @ self.encoding_matrix
                error = expected_latent - t_generated
                reconstruction_error = torch.mean(error ** 2)
        else:
            expected_latent = t_features @ self.encoding_matrix
            error = expected_latent - t_generated
            reconstruction_error = torch.mean(error ** 2)

        reconstruction_error = torch.clamp(reconstruction_error, 0.0, 100.0).item()
        timing["loss_ms"] = (time.perf_counter() - loss_start) * 1000.0

        quality_loss = 1.0 - quality_score
        component_loss = (
            (1.0 - html_score) * 0.35 +
            (1.0 - css_score) * 0.35 +
            (1.0 - js_score) * 0.30
        )

        weights = self.loss_weights
        total_loss = (
            reconstruction_error * weights['reconstruction'] +
            quality_loss * weights['quality'] +
            component_loss * weights['component']
        )
        total_loss = float(np.clip(total_loss, 0.0, 1.0))

        grad_start = time.perf_counter()
        error = torch.clamp(error, -10.0, 10.0)
        gradient = torch.outer(t_features, error)
        quality_scale = 1.0 - quality_score
        gradient = gradient * quality_scale
        gradient = torch.clamp(gradient, -1.0, 1.0)
        timing["grad_ms"] = (time.perf_counter() - grad_start) * 1000.0

        return total_loss, gradient, timing
    
    def _compute_gradients(
        self,
        features: np.ndarray,
        generated_latent: np.ndarray,
        quality_score: float
    ) -> np.ndarray:
        """Compute gradients for weight updates with stability checks"""
        
        if self.use_gpu:
            t_features = _copy_to_buffer(self._feature_buffer, features)
            t_generated = _copy_to_buffer(self._latent_buffer, generated_latent)
            if self.use_amp:
                with _amp_autocast("cuda"):
                    expected_latent = t_features @ self.encoding_matrix
                    error = expected_latent - t_generated
                    error = torch.clamp(error, -10.0, 10.0)
                    gradient = torch.outer(t_features, error)
            else:
                expected_latent = t_features @ self.encoding_matrix
                error = expected_latent - t_generated
                error = torch.clamp(error, -10.0, 10.0)
                gradient = torch.outer(t_features, error)
            quality_scale = 1.0 - quality_score
            gradient = gradient * quality_scale
            gradient = torch.clamp(gradient, -1.0, 1.0)
        else:
            # Reconstruction gradient
            expected_latent = features @ self.encoding_matrix
            error = expected_latent - generated_latent

            # Clip error to prevent explosion
            error = np.clip(error, -10.0, 10.0)

            # Outer product gradient
            gradient = np.outer(features, error)

            # Scale by quality (lower quality = larger update)
            quality_scale = 1.0 - quality_score
            gradient = gradient * quality_scale

            # Additional clipping for safety
            gradient = np.clip(gradient, -1.0, 1.0)
        
        return gradient
    
    def _update_weights(self, gradients: np.ndarray) -> bool:
        """Update model weights using Adam-like optimizer with stability checks"""
        
        self.t += 1
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        
        # Gradient clipping if enabled
        if self.enable_gradient_clip:
            grad_norm = _tensor_norm(gradients)
            self.gradient_norms.append(grad_norm)
            self.gradient_history.append({
                'norm': grad_norm,
                'mean': _tensor_mean(gradients),
                'std': _tensor_std(gradients),
                'max': _tensor_max_abs(gradients)
            })
            
            if grad_norm > self.max_gradient_norm:
                gradients = gradients * (self.max_gradient_norm / (grad_norm + epsilon))
                logger.debug(
                    f"Gradient clipped: {grad_norm:.4f} → {self.max_gradient_norm}"
                )
        
        # Store old weights for comparison
        old_weights = self.encoding_matrix.clone() if _is_torch_tensor(self.encoding_matrix) else self.encoding_matrix.copy()

        if self.use_gpu:
            t_grad = gradients if _is_torch_tensor(gradients) else torch.as_tensor(gradients, device=self.device, dtype=torch.float32)
            self.m = beta1 * self.m + (1 - beta1) * t_grad
            self.v = beta2 * self.v + (1 - beta2) * (t_grad ** 2)

            m_hat = self.m / (1 - beta1 ** self.t + epsilon)
            v_hat = self.v / (1 - beta2 ** self.t + epsilon)
            weight_update = self.learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
            weight_update = torch.clamp(weight_update, -0.1, 0.1)

            self.encoding_matrix = self.encoding_matrix - weight_update
        else:
            # Adam update rule
            self.m = beta1 * self.m + (1 - beta1) * gradients
            self.v = beta2 * self.v + (1 - beta2) * (gradients ** 2)

            m_hat = self.m / (1 - beta1 ** self.t + epsilon)
            v_hat = self.v / (1 - beta2 ** self.t + epsilon)

            # Compute weight update
            weight_update = (
                self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            )

            # Clip weight update
            weight_update = np.clip(weight_update, -0.1, 0.1)

            # Weight update
            self.encoding_matrix -= weight_update
        
        # Compute weight change norm
        weight_change = _tensor_norm(self.encoding_matrix - old_weights)
        self.weight_change_norms.append(weight_change)
        
        # Check weight divergence
        weight_div = self._compute_weight_divergence()
        if weight_div > self.weight_divergence_threshold:
            logger.warn(
                f"Weight divergence too high: {weight_div:.3f}, "
                f"reverting to previous weights"
            )
            self.encoding_matrix = old_weights
            return False
        
        logger.debug(
            f"Weights updated: change_norm={weight_change:.6f}, "
            f"divergence={weight_div:.3f}"
        )
        
        return True
    
    def _check_gradient_health(self, gradients: np.ndarray) -> Dict[str, Any]:
        """Check gradient health (NaN/Inf/explosion detection)"""

        if _is_torch_tensor(gradients):
            if torch.any(torch.isnan(gradients)):
                return {
                    'is_healthy': False,
                    'reason': 'NaN values detected',
                    'details': f"NaN count: {int(torch.sum(torch.isnan(gradients)).item())}"
                }

            if torch.any(torch.isinf(gradients)):
                return {
                    'is_healthy': False,
                    'reason': 'Inf values detected',
                    'details': f"Inf count: {int(torch.sum(torch.isinf(gradients)).item())}"
                }
        else:
            if np.any(np.isnan(gradients)):
                return {
                    'is_healthy': False,
                    'reason': 'NaN values detected',
                    'details': f"NaN count: {np.sum(np.isnan(gradients))}"
                }

            if np.any(np.isinf(gradients)):
                return {
                    'is_healthy': False,
                    'reason': 'Inf values detected',
                    'details': f"Inf count: {np.sum(np.isinf(gradients))}"
                }

        grad_norm = _tensor_norm(gradients)
        grad_max = _tensor_max_abs(gradients)

        if grad_norm > 100.0:
            return {
                'is_healthy': False,
                'reason': 'Gradient explosion',
                'details': f"Gradient norm: {grad_norm:.4f}"
            }

        if grad_max > 10.0:
            logger.warn(f"Large gradient detected: max={grad_max:.4f}")

        return {
            'is_healthy': True,
            'grad_norm': float(grad_norm),
            'grad_max': float(grad_max)
        }
    
    def _detect_anomaly_feedback(
        self,
        quality_score: float,
        feedback_data: Dict[str, Any]
    ) -> bool:
        """Detect anomalous feedback using IQR method"""
        
        if len(self.quality_history) < 10:
            return False
        
        # Calculate IQR
        q1 = np.percentile(list(self.quality_history), 25)
        q3 = np.percentile(list(self.quality_history), 75)
        iqr = q3 - q1
        
        if iqr < 0.05:  # Skip if IQR too small
            return False
        
        lower_bound = q1 - self.outlier_threshold * iqr
        upper_bound = q3 + self.outlier_threshold * iqr
        
        is_outlier = quality_score < lower_bound or quality_score > upper_bound
        
        if is_outlier:
            logger.debug(
                f"Outlier detected: score={quality_score:.3f}, "
                f"bounds=[{lower_bound:.3f}, {upper_bound:.3f}]"
            )
        
        return is_outlier
    
    def _update_adaptive_loss_weights(self) -> None:
        """Adaptively adjust loss weights based on performance"""
        
        if len(self.quality_scores) < 20:
            return
        
        # Analyze recent performance
        recent_quality = np.mean(self.quality_scores[-20:])
        recent_html = np.mean([
            score for score in self.quality_scores[-20:] 
            if score > 0.3
        ]) if self.quality_scores else 0.5
        
        # Adjust weights based on quality trends
        if recent_quality < 0.4:
            # Low quality: focus more on component quality
            self.loss_weights = {
                'reconstruction': 0.2,
                'quality': 0.3,
                'component': 0.5
            }
        elif recent_quality < 0.6:
            # Medium quality: balanced approach
            self.loss_weights = {
                'reconstruction': 0.3,
                'quality': 0.4,
                'component': 0.3
            }
        else:
            # High quality: focus on reconstruction
            self.loss_weights = {
                'reconstruction': 0.5,
                'quality': 0.3,
                'component': 0.2
            }
        
        self.loss_weight_history.append(self.loss_weights.copy())
        
        logger.debug(f"Loss weights updated: {self.loss_weights}")
    
    def _adaptive_learning_rate_schedule(self) -> None:
        """Adaptively adjust learning rate based on convergence"""
        
        if len(self.training_losses) < 20:
            return
        
        convergence = self._compute_convergence()
        recent_losses = self.training_losses[-20:]
        loss_trend = recent_losses[-1] - recent_losses[0]
        
        # Track learning rate schedule
        self.lr_schedule.append(self.learning_rate)
        
        # Adjust learning rate
        if convergence > 0.85:
            # Converged: reduce learning rate
            new_lr = self.learning_rate * 0.95
            self.lr_changes.append(('converged', new_lr))
        elif convergence < 0.3 and loss_trend > 0.05:
            # Diverging: reduce learning rate more aggressively
            new_lr = self.learning_rate * 0.85
            self.lr_changes.append(('diverging', new_lr))
        elif loss_trend > 0.1:
            # Loss increasing: reduce learning rate
            new_lr = self.learning_rate * 0.9
            self.lr_changes.append(('increasing_loss', new_lr))
        else:
            return
        
        # Ensure learning rate stays in reasonable range
        new_lr = np.clip(new_lr, self.initial_learning_rate * 0.001, 
                         self.initial_learning_rate * 10)
        
        if new_lr != self.learning_rate:
            old_lr = self.learning_rate
            self.learning_rate = new_lr
            logger.info(
                f"Learning rate adjusted: {old_lr:.6f} → {new_lr:.6f} "
                f"(convergence={convergence:.3f})"
            )
    
    def _compute_weight_divergence(self) -> float:
        """Compute how much weights have diverged from initialization"""
        diff = self.encoding_matrix - self.encoding_matrix_initial
        divergence = _tensor_norm(diff) / (_tensor_norm(self.encoding_matrix_initial) + 1e-8)
        return float(divergence)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics"""
        
        if not self.training_losses:
            return {
                "loss": {
                    "average": 0.0,
                    "recent_average": 0.0,
                    "best": 0.0,
                    "worst": 0.0,
                    "std": 0.0,
                    "trend": "unknown"
                },
                "quality": {
                    "average": 0.5,
                    "recent_average": 0.5,
                    "best": 0.0,
                    "worst": 0.0,
                    "std": 0.0,
                },
                "optimization": {
                    "update_count": 0,
                    "feedback_count": 0,
                    "skipped_updates": 0,
                    "anomaly_count": 0,
                    "convergence": 0.0,
                    "improvement": 0.0,
                    "learning_rate": self.learning_rate,
                    "lr_changes_count": 0,
                },
                "weights": {
                    "norm": _tensor_norm(self.encoding_matrix),
                    "mean": _tensor_mean_abs(self.encoding_matrix),
                    "divergence": 0.0,
                    "divergence_threshold": self.weight_divergence_threshold,
                    "weight_change_history": [],
                },
                "gradients": {},
                "loss_weights": self.loss_weights,
                "status": "no_data"
            }
        
        losses = np.array(self.training_losses)
        qualities = np.array(self.quality_scores)
        
        # Recent window (last 100 samples)
        window = min(100, len(losses))
        recent_losses = losses[-window:]
        recent_qualities = qualities[-window:]
        
        convergence = self._compute_convergence()
        improvement = self._compute_improvement()
        weight_divergence = self._compute_weight_divergence()
        
        # Gradient statistics
        grad_stats = {}
        if self.gradient_norms:
            grad_array = np.array(self.gradient_norms[-100:])
            grad_stats = {
                'mean': float(np.mean(grad_array)),
                'std': float(np.std(grad_array)),
                'max': float(np.max(grad_array)),
                'min': float(np.min(grad_array)),
            }
        
        metrics = {
            "loss": {
                "average": float(np.mean(losses)),
                "recent_average": float(np.mean(recent_losses)),
                "best": float(np.min(losses)),
                "worst": float(np.max(losses)),
                "std": float(np.std(losses)),
                "trend": "increasing" if recent_losses[-1] > recent_losses[0] else "decreasing"
            },
            "quality": {
                "average": float(np.mean(qualities)),
                "recent_average": float(np.mean(recent_qualities)),
                "best": float(np.max(qualities)),
                "worst": float(np.min(qualities)),
                "std": float(np.std(qualities)),
            },
            "optimization": {
                "update_count": self.update_count,
                "feedback_count": self.feedback_count,
                "skipped_updates": self.skipped_updates,
                "anomaly_count": self.anomaly_count,
                "convergence": float(convergence),
                "improvement": float(improvement),
                "learning_rate": float(self.learning_rate),
                "lr_changes_count": len(self.lr_changes),
            },
            "weights": {
                "norm": _tensor_norm(self.encoding_matrix),
                "mean": _tensor_mean_abs(self.encoding_matrix),
                "divergence": float(weight_divergence),
                "divergence_threshold": self.weight_divergence_threshold,
                "weight_change_history": [float(x) for x in list(self.weight_change_norms[-10:])],
            },
            "gradients": grad_stats,
            "loss_weights": self.loss_weights,
            "status": "healthy" if convergence > 0.5 else "unstable"
        }
        
        return metrics

    
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
            "norm": _tensor_norm(self.encoding_matrix),
            "mean": _tensor_mean(self.encoding_matrix),
            "std": _tensor_std(self.encoding_matrix),
            "min": _tensor_min(self.encoding_matrix),
            "max": _tensor_max(self.encoding_matrix),
            "sparsity": _tensor_sparsity(self.encoding_matrix),
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
