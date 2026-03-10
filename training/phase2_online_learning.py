#!/usr/bin/env python3
"""
Phase 2: Online Learning System - Complete Implementation
在线学习系统完整实现，包含梯度计算和参数优化

核心功能:
- 损失函数设计 (重构损失 + 质量损失 + 正则化)
- 梯度计算 (反向传播)
- Adam 优化器实现
- 参数更新和约束
- 完整的学习反馈循环
- 模型版本管理

Architecture:
  Feedback Data (原始代码, 生成代码, 质量评分)
       ↓
  Loss Computation (计算损失)
       ↓
  Gradient Calculation (计算梯度)
       ↓
  Gradient Clipping (梯度裁剪)
       ↓
  Adam Update (参数更新)
       ↓
  Constraint Enforcement (约束检查)
       ↓
  Updated Model (更新的模型)
"""

import sys
import os
sys.path.insert(0, '/home/stone/BrowerAI/training')

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
from collections import deque
import json

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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


class LossFunction:
    """损失函数设计和计算
    
    总损失 = 重构损失 + 质量损失 + 正则化
            = reconstruction_loss + quality_loss + regularization_loss
    """
    
    def __init__(self, 
                 alpha: float = 0.5,      # 重构损失权重
                 beta: float = 0.3,       # 质量损失权重
                 lambda_reg: float = 0.0001,
                 use_gpu: bool = False,
                 device: Optional[str] = None):  # 正则化系数
        """
        Initialize loss function
        
        Args:
            alpha: 重构损失权重 (0-1)
            beta: 质量损失权重 (0-1)
            lambda_reg: L2正则化系数
        """
        self.alpha = alpha
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.compute_count = 0
        self.use_gpu = use_gpu and _TORCH_AVAILABLE
        self.device = device
    
    def compute_reconstruction_loss(self, 
                                   original_features: np.ndarray,
                                   generated_features: np.ndarray) -> float:
        """计算重构损失 (特征空间距离)
        
        使用 L2 距离衡量生成特征与原始特征的差异
        
        Args:
            original_features: 原始48维特征
            generated_features: 生成代码提取的特征
        
        Returns:
            重构损失 (标量)
        """
        if self.use_gpu:
            t_original = torch.as_tensor(original_features, device=self.device, dtype=torch.float32)
            t_generated = torch.as_tensor(generated_features, device=self.device, dtype=torch.float32)
            diff = t_original - t_generated
            reconstruction_loss = float(torch.mean(diff ** 2).item())
        else:
            # L2距离
            diff = original_features - generated_features
            reconstruction_loss = float(np.mean(diff ** 2))
        return reconstruction_loss
    
    def compute_quality_loss(self, quality_score: float) -> float:
        """计算质量损失
        
        质量越高，损失越低
        loss = 1 - quality_score
        
        Args:
            quality_score: 质量评分 (0-1)
        
        Returns:
            质量损失 (0-1)
        """
        quality_loss = 1.0 - float(quality_score)
        return quality_loss
    
    def compute_regularization_loss(self, weights: np.ndarray) -> float:
        """计算正则化损失 (L2正则化)
        
        防止权重过大导致过拟合
        
        Args:
            weights: 编码权重矩阵 (48, 256)
        
        Returns:
            正则化损失 (标量)
        """
        if self.lambda_reg == 0:
            return 0.0
        if self.use_gpu:
            t_weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
            reg_loss = float(self.lambda_reg * torch.sum(t_weights ** 2).item())
        else:
            # L2范数平方
            reg_loss = float(self.lambda_reg * np.sum(weights ** 2))
        return reg_loss
    
    def compute_total_loss(self,
                          original_features: np.ndarray,
                          generated_features: np.ndarray,
                          quality_score: float,
                          weights: np.ndarray) -> Dict[str, float]:
        """计算总损失和各部分损失
        
        Args:
            original_features: 原始特征
            generated_features: 生成特征
            quality_score: 质量评分
            weights: 模型权重
        
        Returns:
            包含所有损失分量的字典
        """
        # 各部分损失
        recon_loss = self.compute_reconstruction_loss(original_features, generated_features)
        quality_loss = self.compute_quality_loss(quality_score)
        reg_loss = self.compute_regularization_loss(weights)
        
        # 加权总损失
        total_loss = (
            self.alpha * recon_loss +
            self.beta * quality_loss +
            (1 - self.alpha - self.beta) * reg_loss
        )
        
        self.compute_count += 1
        
        return {
            'total_loss': float(total_loss),
            'reconstruction_loss': float(recon_loss),
            'quality_loss': float(quality_loss),
            'regularization_loss': float(reg_loss),
            'breakdown': {
                'alpha': self.alpha,
                'beta': self.beta,
                'lambda_reg': self.lambda_reg,
            }
        }


class GradientComputer:
    """梯度计算模块 (反向传播)
    
    计算损失相对于权重矩阵的梯度
    """
    
    def __init__(self, feature_dim: int = 48, latent_dim: int = 256,
                 use_gpu: bool = False, device: Optional[str] = None):
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.compute_count = 0
        self.gradient_history = deque(maxlen=100)
        self.use_gpu = use_gpu and _TORCH_AVAILABLE
        self.device = device
    
    def compute_reconstruction_gradient(self,
                                       original_features: np.ndarray,
                                       generated_features: np.ndarray,
                                       latent_vector: np.ndarray) -> np.ndarray:
        """计算重构损失相对于权重的梯度
        
        Reconstruction Loss = mean((original - generated)^2)
        
        使用链式法则:
        dL/dW = dL/dgenerated * dgenerated/dW
        
        Args:
            original_features: 原始特征 (48,)
            generated_features: 生成特征 (48,)
            latent_vector: 潜在向量 (256,) - 用于反向传播
        
        Returns:
            梯度矩阵 (48, 256)
        """
        if self.use_gpu:
            t_original = torch.as_tensor(original_features, device=self.device, dtype=torch.float32)
            t_generated = torch.as_tensor(generated_features, device=self.device, dtype=torch.float32)
            t_latent = torch.as_tensor(latent_vector, device=self.device, dtype=torch.float32)
            diff = t_generated - t_original
            dL_dgen = 2 * diff / self.feature_dim
            gradient = torch.outer(dL_dgen, t_latent).cpu().numpy()
        else:
            # 损失相对于生成特征的梯度
            diff = generated_features - original_features
            dL_dgen = 2 * diff / self.feature_dim  # 平均化
            # 简单的线性近似: gradient = dL_dgen ⊗ latent_vector
            gradient = np.outer(dL_dgen, latent_vector)  # (48, 256)
        
        return gradient
    
    def compute_quality_gradient(self,
                                latent_vector: np.ndarray,
                                quality_feedback: float = 0.1) -> np.ndarray:
        """计算质量损失相对于权重的梯度
        
        如果质量评分低, 梯度应该推动参数变化
        
        Args:
            latent_vector: 潜在向量 (256,)
            quality_feedback: 质量反馈信号 (0-1)
        
        Returns:
            梯度矩阵 (48, 256)
        """
        # 质量梯度信号
        quality_signal = -quality_feedback  # 负梯度表示改进方向
        
        # 扩展到权重空间
        gradient = quality_signal * np.ones((self.feature_dim, self.latent_dim)) * 0.01
        
        return gradient
    
    def compute_regularization_gradient(self, 
                                       weights: np.ndarray,
                                       lambda_reg: float) -> np.ndarray:
        """计算正则化损失梯度
        
        L2正则化: loss = lambda * sum(W^2)
        gradient = 2 * lambda * W
        
        Args:
            weights: 权重矩阵 (48, 256)
            lambda_reg: 正则化系数
        
        Returns:
            梯度矩阵 (48, 256)
        """
        if self.use_gpu:
            t_weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
            gradient = (2 * lambda_reg * t_weights).cpu().numpy()
        else:
            gradient = 2 * lambda_reg * weights
        return gradient
    
    def compute_total_gradient(self,
                              original_features: np.ndarray,
                              generated_features: np.ndarray,
                              latent_vector: np.ndarray,
                              weights: np.ndarray,
                              quality_score: float,
                              alpha: float = 0.5,
                              beta: float = 0.3,
                              lambda_reg: float = 0.0001) -> Tuple[np.ndarray, Dict[str, Any]]:
        """计算总梯度 (各部分加权)
        
        Args:
            original_features: 原始特征
            generated_features: 生成特征
            latent_vector: 潜在向量
            weights: 权重矩阵
            quality_score: 质量评分
            alpha: 重构损失权重
            beta: 质量损失权重
            lambda_reg: 正则化系数
        
        Returns:
            总梯度矩阵, 梯度详细信息
        """
        # 各部分梯度
        grad_recon = self.compute_reconstruction_gradient(
            original_features, generated_features, latent_vector
        )
        grad_quality = self.compute_quality_gradient(latent_vector, quality_score)
        grad_reg = self.compute_regularization_gradient(weights, lambda_reg)
        
        # 加权总梯度
        total_gradient = (
            alpha * grad_recon +
            beta * grad_quality +
            (1 - alpha - beta) * grad_reg
        )
        
        # 计算梯度统计
        gradient_norm = float(np.linalg.norm(total_gradient))
        self.gradient_history.append(gradient_norm)
        self.compute_count += 1
        
        return total_gradient, {
            'gradient_norm': gradient_norm,
            'recon_norm': float(np.linalg.norm(grad_recon)),
            'quality_norm': float(np.linalg.norm(grad_quality)),
            'reg_norm': float(np.linalg.norm(grad_reg)),
            'mean_gradient': float(np.mean(np.abs(total_gradient))),
            'max_gradient': float(np.max(np.abs(total_gradient))),
        }


class AdamOptimizer:
    """Adam 优化器实现
    
    自适应学习率方法，结合动量和RMSprop
    
    参数更新:
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t       (一阶矩)
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2    (二阶矩)
    m_hat_t = m_t / (1 - beta1^t)                  (偏差修正)
    v_hat_t = v_t / (1 - beta2^t)                  (偏差修正)
    W_t = W_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,          # 一阶矩指数衰减
                 beta2: float = 0.999,        # 二阶矩指数衰减
                 epsilon: float = 1e-8,       # 数值稳定性
                 weight_decay: float = 0.01,
                 use_gpu: bool = False,
                 device: Optional[str] = None): # L2权重衰减
        """
        Initialize Adam optimizer
        
        Args:
            learning_rate: 初始学习率
            beta1: 一阶矩指数衰减系数
            beta2: 二阶矩指数衰减系数
            epsilon: 数值稳定性常数
            weight_decay: L2权重衰减系数
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.use_gpu = use_gpu and _TORCH_AVAILABLE
        self.device = device
        
        # 优化器状态
        self.t = 0  # 时步
        self.m = None  # 一阶矩
        self.v = None  # 二阶矩
        self.update_count = 0
        self.learning_rate_history = deque(maxlen=100)
    
    def initialize(self, weight_shape: Tuple[int, int]):
        """初始化优化器状态
        
        Args:
            weight_shape: 权重矩阵形状 (48, 256)
        """
        self.m = np.zeros(weight_shape)
        self.v = np.zeros(weight_shape)
        self.t = 0
    
    def update(self, weights: np.ndarray, gradient: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """执行一步参数更新
        
        Args:
            weights: 当前权重矩阵
            gradient: 梯度矩阵
        
        Returns:
            更新后的权重, 更新信息
        """
        if self.m is None:
            self.initialize(weights.shape)
        
        # 时步增加
        self.t += 1
        
        if self.use_gpu:
            t_weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
            t_grad = torch.as_tensor(gradient, device=self.device, dtype=torch.float32)
            t_m = torch.as_tensor(self.m, device=self.device, dtype=torch.float32)
            t_v = torch.as_tensor(self.v, device=self.device, dtype=torch.float32)

            t_m = self.beta1 * t_m + (1 - self.beta1) * t_grad
            t_v = self.beta2 * t_v + (1 - self.beta2) * (t_grad ** 2)

            m_hat = t_m / (1 - self.beta1 ** self.t)
            v_hat = t_v / (1 - self.beta2 ** self.t)

            lr = self.learning_rate
            updated_weights = t_weights - lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
            updated_weights = updated_weights * (1 - self.weight_decay)

            self.m = t_m.cpu().numpy()
            self.v = t_v.cpu().numpy()
            updated_weights = updated_weights.cpu().numpy()
        else:
            # 一阶矩更新
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient

            # 二阶矩更新
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

            # 偏差修正
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)

            # 学习率调整 (可选的衰减)
            lr = self.learning_rate

            # 参数更新
            updated_weights = weights - lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # 权重衰减 (L2正则化)
            updated_weights = updated_weights * (1 - self.weight_decay)
        
        self.update_count += 1
        self.learning_rate_history.append(lr)
        
        # 计算权重变化
        weight_change = np.linalg.norm(updated_weights - weights)
        
        return updated_weights, {
            't': self.t,
            'learning_rate': float(lr),
            'weight_change_norm': float(weight_change),
            'mean_m': float(np.mean(np.abs(m_hat))),
            'mean_v': float(np.mean(np.abs(v_hat))),
            'update_ratio': float(weight_change / max(np.linalg.norm(weights), 1e-8)),
        }


class OnlineLearningSystem:
    """完整的在线学习系统
    
    整合所有学习组件:
    - 损失函数
    - 梯度计算
    - 参数优化
    - 完整学习循环
    """
    
    def __init__(self,
                 feature_dim: int = 48,
                 latent_dim: int = 256,
                 learning_rate: float = 0.001):
        """
        Initialize online learning system
        
        Args:
            feature_dim: 特征维度 (48)
            latent_dim: 潜在维度 (256)
            learning_rate: 学习率
        """
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        
        self.learning_mode = _env_flag("BROWERAI_LEARNING_MODE")
        self.use_gpu = self.learning_mode and _env_flag("BROWERAI_USE_GPU")
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

        # 系统组件
        self.loss_function = LossFunction(
            alpha=0.5,
            beta=0.3,
            lambda_reg=0.0001,
            use_gpu=self.use_gpu,
            device=self.device,
        )
        self.gradient_computer = GradientComputer(
            feature_dim,
            latent_dim,
            use_gpu=self.use_gpu,
            device=self.device,
        )
        self.optimizer = AdamOptimizer(
            learning_rate=learning_rate,
            use_gpu=self.use_gpu,
            device=self.device,
        )
        
        # 初始化优化器
        self.optimizer.initialize((feature_dim, latent_dim))
        
        # 学习历史
        self.learning_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)
        self.gradient_norm_history = deque(maxlen=1000)
        
        # 学习统计
        self.total_updates = 0
        self.total_loss = 0.0
        
        logger.info("✓ OnlineLearningSystem initialized")
        logger.info(f"  Feature dimension: {feature_dim}")
        logger.info(f"  Latent dimension: {latent_dim}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Learning mode: {self.learning_mode}")
        logger.info(f"  GPU learning: {self.use_gpu}")
    
    def process_feedback(self,
                        original_features: np.ndarray,
                        generated_features: np.ndarray,
                        latent_vector: np.ndarray,
                        weights: np.ndarray,
                        quality_score: float,
                        session_id: str = "unknown") -> Dict[str, Any]:
        """处理学习反馈，执行完整的学习更新
        
        完整管道:
        1. 计算损失
        2. 计算梯度
        3. 梯度裁剪
        4. 参数更新
        5. 约束检查
        
        Args:
            original_features: 原始48维特征
            generated_features: 生成特征
            latent_vector: 潜在向量
            weights: 编码权重矩阵
            quality_score: 质量评分 (0-1)
            session_id: 会话ID
        
        Returns:
            完整的学习结果字典
        """
        import time
        start_time = time.time()
        
        # 步骤1: 计算损失
        loss_dict = self.loss_function.compute_total_loss(
            original_features, generated_features, quality_score, weights
        )
        
        # 步骤2: 计算梯度
        gradient, grad_info = self.gradient_computer.compute_total_gradient(
            original_features, generated_features, latent_vector, weights,
            quality_score, alpha=0.5, beta=0.3, lambda_reg=0.0001
        )
        
        # 步骤3: 梯度裁剪 (防止梯度爆炸)
        max_grad_norm = 1.0
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > max_grad_norm:
            gradient = gradient * (max_grad_norm / grad_norm)
            clipped = True
        else:
            clipped = False
        
        # 步骤4: 参数更新
        updated_weights, opt_info = self.optimizer.update(weights, gradient)
        
        # 步骤5: 权重约束 (保持在合理范围内)
        updated_weights = np.clip(updated_weights, -1.0, 1.0)
        
        # 记录学习信息
        elapsed_ms = (time.time() - start_time) * 1000
        
        learning_record = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'loss': loss_dict,
            'gradient_info': grad_info,
            'optimizer_info': opt_info,
            'gradient_clipped': clipped,
            'processing_time_ms': elapsed_ms,
        }
        
        self.learning_history.append(learning_record)
        self.loss_history.append(loss_dict['total_loss'])
        self.gradient_norm_history.append(grad_info['gradient_norm'])
        
        self.total_updates += 1
        self.total_loss += loss_dict['total_loss']
        
        return {
            'success': True,
            'updated_weights': updated_weights,
            'learning_record': learning_record,
            'weight_change_norm': opt_info['weight_change_norm'],
        }
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """获取学习摘要统计"""
        if not self.loss_history:
            return {'status': 'no_learning_yet'}
        
        loss_array = np.array(list(self.loss_history))
        grad_array = np.array(list(self.gradient_norm_history))
        
        return {
            'total_updates': self.total_updates,
            'average_loss': float(np.mean(loss_array)),
            'latest_loss': float(loss_array[-1]),
            'loss_trend': 'decreasing' if len(loss_array) > 1 and loss_array[-1] < loss_array[0] else 'unknown',
            'average_gradient_norm': float(np.mean(grad_array)),
            'max_gradient_norm': float(np.max(grad_array)),
            'optimizer_timesteps': self.optimizer.t,
            'learning_rate': self.optimizer.learning_rate,
        }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """获取详细的学习指标"""
        if not self.learning_history:
            return {}
        
        latest = self.learning_history[-1]
        
        return {
            'latest_update': latest,
            'loss_reconstruction': float(np.mean([
                h['loss']['reconstruction_loss'] for h in list(self.learning_history)[-10:]
            ])),
            'loss_quality': float(np.mean([
                h['loss']['quality_loss'] for h in list(self.learning_history)[-10:]
            ])),
            'gradient_norm_trend': list(self.gradient_norm_history)[-10:],
        }


def main():
    """演示 Phase 2 在线学习系统"""
    print("\n" + "="*70)
    print("Phase 2: Online Learning System - Demonstration")
    print("="*70)
    
    # 初始化系统
    system = OnlineLearningSystem(
        feature_dim=48,
        latent_dim=256,
        learning_rate=0.001
    )
    
    # 初始权重
    np.random.seed(42)
    initial_weights = np.random.randn(48, 256) * 0.01
    
    print("\n[1] Processing 10 learning feedback iterations...")
    
    for iteration in range(10):
        # 模拟反馈数据
        original_features = np.random.rand(48)
        generated_features = original_features + np.random.randn(48) * 0.1
        latent_vector = np.random.randn(256)
        quality_score = 0.7 + iteration * 0.03  # 逐渐改进
        
        # 处理反馈
        result = system.process_feedback(
            original_features,
            generated_features,
            latent_vector,
            initial_weights,
            quality_score,
            session_id=f"session_0_{iteration}"
        )
        
        initial_weights = result['updated_weights']
        
        # 定期显示进展
        if (iteration + 1) % 5 == 0:
            summary = system.get_learning_summary()
            print(f"\nIteration {iteration + 1}:")
            print(f"  Average loss: {summary['average_loss']:.4f}")
            print(f"  Latest loss: {summary['latest_loss']:.4f}")
            print(f"  Avg gradient norm: {summary['average_gradient_norm']:.4f}")
            print(f"  Optimizer steps: {summary['optimizer_timesteps']}")
    
    # 最终摘要
    print("\n" + "-"*70)
    print("Final Learning Summary:")
    summary = system.get_learning_summary()
    print(f"  Total updates: {summary['total_updates']}")
    print(f"  Final loss: {summary['latest_loss']:.4f}")
    print(f"  Loss trend: {summary['loss_trend']}")
    print(f"  Average gradient norm: {summary['average_gradient_norm']:.4f}")
    
    print("\n✅ Phase 2 demonstration complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
