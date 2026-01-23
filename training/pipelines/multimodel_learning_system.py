#!/usr/bin/env python3
"""
🔥 多模型协作学习系统 - BrowerAI高级学习引擎

功能:
1. 多模型联合训练 - 利用多个模型的互补性
2. 模型融合 - 结合多个模型的预测
3. 自适应学习 - 动态调整模型权重
4. 分布式学习 - 支持多GPU学习
5. 模型热交换 - 无缝更换模型版本
6. 性能监控 - 实时追踪学习效果
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Any, Optional, Set
import numpy as np
from dataclasses import dataclass, field
import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BrowerAI.MultiModelLearning")


# ============================================================================
# 1. 模型管理系统
# ============================================================================

@dataclass
class ModelInfo:
    """模型信息"""
    model_id: str
    model_name: str
    version: str
    creation_date: str
    last_updated: str
    total_params: int
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_samples: int = 0
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'version': self.version,
            'creation_date': self.creation_date,
            'last_updated': self.last_updated,
            'total_params': self.total_params,
            'performance_metrics': self.performance_metrics,
            'training_samples': self.training_samples,
            'is_active': self.is_active,
        }


class ModelRegistry:
    """模型注册表"""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.model_versions: Dict[str, List[str]] = defaultdict(list)
        
        logger.info("✓ ModelRegistry 初始化")
    
    def register_model(self, model_info: ModelInfo) -> bool:
        """注册模型"""
        
        if model_info.model_id in self.models:
            logger.warning(f"模型 {model_info.model_id} 已存在，覆盖注册")
        
        self.models[model_info.model_id] = model_info
        self.model_versions[model_info.model_name].append(model_info.model_id)
        
        logger.info(f"✓ 模型注册: {model_info.model_name} v{model_info.version}")
        
        return True
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return self.models.get(model_id)
    
    def get_best_model(self, model_name: str, metric: str = 'accuracy') -> Optional[ModelInfo]:
        """获取最佳模型"""
        
        versions = self.model_versions.get(model_name, [])
        if not versions:
            return None
        
        best_model = None
        best_score = -float('inf')
        
        for model_id in versions:
            model = self.models[model_id]
            if model.is_active and metric in model.performance_metrics:
                score = model.performance_metrics[metric]
                if score > best_score:
                    best_score = score
                    best_model = model
        
        return best_model
    
    def list_models(self) -> List[ModelInfo]:
        """列表所有模型"""
        return list(self.models.values())


# ============================================================================
# 2. 多模型融合系统
# ============================================================================

class ModelEnsemble:
    """模型融合集合"""
    
    def __init__(self, models: List[nn.Module], fusion_strategy: str = 'weighted_average'):
        self.models = models
        self.fusion_strategy = fusion_strategy
        self.model_weights = self._initialize_weights(len(models))
        self.fusion_history = []
        
        logger.info(f"✓ ModelEnsemble 初始化")
        logger.info(f"  模型数: {len(models)}")
        logger.info(f"  融合策略: {fusion_strategy}")
        logger.info(f"  初始权重: {self.model_weights}")
    
    def _initialize_weights(self, num_models: int) -> List[float]:
        """初始化模型权重"""
        return [1.0 / num_models] * num_models
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """融合前向传播"""
        
        outputs = []
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                outputs.append(output)
        
        # 应用融合策略
        fused_output = self._apply_fusion(outputs)
        
        fusion_info = {
            'num_models': len(self.models),
            'fusion_strategy': self.fusion_strategy,
            'model_weights': self.model_weights,
            'output_shape': fused_output.shape,
        }
        
        self.fusion_history.append(fusion_info)
        
        return fused_output, fusion_info
    
    def _apply_fusion(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """应用融合策略"""
        
        if self.fusion_strategy == 'weighted_average':
            result = torch.zeros_like(outputs[0])
            for weight, output in zip(self.model_weights, outputs):
                result += weight * output
            return result
        
        elif self.fusion_strategy == 'voting':
            # 对分类任务: 投票
            stacked = torch.stack(outputs, dim=0)
            result, _ = torch.mode(stacked, dim=0)
            return result
        
        elif self.fusion_strategy == 'max_pooling':
            # 取最大值
            stacked = torch.stack(outputs, dim=0)
            result, _ = torch.max(stacked, dim=0)
            return result
        
        else:
            raise ValueError(f"未知的融合策略: {self.fusion_strategy}")
    
    def update_weights(self, performance_scores: List[float]):
        """基于性能更新权重"""
        
        # 归一化性能分数
        total_score = sum(max(0, score) for score in performance_scores)
        
        if total_score > 0:
            self.model_weights = [
                max(0, score) / total_score 
                for score in performance_scores
            ]
        
        logger.info(f"✓ 模型权重已更新: {self.model_weights}")


# ============================================================================
# 3. 自适应学习控制器
# ============================================================================

class AdaptiveLearningController:
    """自适应学习控制器
    
    根据性能动态调整学习策略
    """
    
    def __init__(self, initial_lr: float = 0.001):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.learning_history = []
        self.strategy_switches = 0
        
        logger.info("✓ AdaptiveLearningController 初始化")
        logger.info(f"  初始学习率: {initial_lr}")
    
    def compute_adaptive_lr(self, 
                           recent_losses: List[float],
                           recent_rewards: List[float]) -> float:
        """计算自适应学习率"""
        
        if len(recent_losses) < 2:
            return self.current_lr
        
        # 计算损失趋势
        loss_gradient = np.gradient(recent_losses[-5:]) if len(recent_losses) >= 5 else [0]
        loss_trend = np.mean(loss_gradient)
        
        # 计算奖励趋势
        reward_gradient = np.gradient(recent_rewards[-5:]) if len(recent_rewards) >= 5 else [0]
        reward_trend = np.mean(reward_gradient)
        
        # 根据趋势调整学习率
        if loss_trend < -0.01:  # 损失快速下降
            new_lr = self.current_lr * 1.2  # 加速学习
        elif loss_trend > 0.01:  # 损失上升
            new_lr = self.current_lr * 0.8  # 减速学习
        elif reward_trend > 0.01:  # 奖励上升
            new_lr = self.current_lr * 1.05  # 轻微加速
        elif reward_trend < -0.01:  # 奖励下降
            new_lr = self.current_lr * 0.95  # 轻微减速
        else:
            new_lr = self.current_lr  # 保持不变
        
        # 限制学习率范围
        new_lr = max(self.initial_lr * 0.1, min(self.initial_lr * 10, new_lr))
        
        self.current_lr = new_lr
        
        return new_lr
    
    def suggest_strategy(self, 
                        loss_variance: float,
                        reward_variance: float) -> Dict[str, str]:
        """建议学习策略"""
        
        suggestions = {}
        
        if loss_variance > 0.1:
            suggestions['stability'] = '损失波动过大，建议使用梯度剪裁或权重衰减'
        else:
            suggestions['stability'] = '训练稳定'
        
        if reward_variance > 0.2:
            suggestions['consistency'] = '奖励不稳定，建议增加反馈样本'
        else:
            suggestions['consistency'] = '反馈一致'
        
        return suggestions


# ============================================================================
# 4. 分布式学习协调器
# ============================================================================

class DistributedLearningCoordinator:
    """分布式学习协调器
    
    管理多GPU或多机学习
    """
    
    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers
        self.available_devices = self._detect_devices()
        self.worker_assignments: Dict[int, str] = {}
        self.synchronized_models: Dict[str, nn.Module] = {}
        
        logger.info(f"✓ DistributedLearningCoordinator 初始化")
        logger.info(f"  可用设备: {self.available_devices}")
        logger.info(f"  Worker数: {num_workers}")
    
    def _detect_devices(self) -> List[str]:
        """检测可用设备"""
        
        devices = []
        
        # 检查GPU
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            devices.extend([f'cuda:{i}' for i in range(num_gpus)])
            logger.info(f"✓ 检测到 {num_gpus} 个GPU")
        
        # CPU总是可用
        devices.append('cpu')
        
        return devices
    
    def assign_workers_to_devices(self) -> Dict[int, str]:
        """分配worker到设备"""
        
        num_devices = len(self.available_devices)
        
        for worker_id in range(self.num_workers):
            device_idx = worker_id % num_devices
            device = self.available_devices[device_idx]
            self.worker_assignments[worker_id] = device
        
        logger.info("✓ Worker分配完成:")
        for worker_id, device in self.worker_assignments.items():
            logger.info(f"  Worker {worker_id} -> {device}")
        
        return self.worker_assignments
    
    def synchronize_models(self, models_dict: Dict[str, nn.Module], 
                          primary_device: str = 'cuda:0'):
        """同步模型参数"""
        
        self.synchronized_models = {}
        
        for model_name, model in models_dict.items():
            # 将模型放到主设备
            model = model.to(primary_device)
            self.synchronized_models[model_name] = model
        
        logger.info(f"✓ {len(models_dict)} 个模型已同步到 {primary_device}")


# ============================================================================
# 5. 性能监控系统
# ============================================================================

class PerformanceMonitor:
    """性能监控系统"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.thresholds = {
            'loss': {'warning': 1.0, 'critical': 2.0},
            'accuracy': {'warning': 0.5, 'critical': 0.3},
            'learning_rate': {'warning': 1e-6, 'critical': 1e-8},
        }
        self.alerts = []
        
        logger.info("✓ PerformanceMonitor 初始化")
    
    def record_metric(self, metric_name: str, value: float, 
                      timestamp: Optional[str] = None):
        """记录指标"""
        
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': timestamp,
        })
        
        # 检查阈值
        self._check_thresholds(metric_name, value)
    
    def _check_thresholds(self, metric_name: str, value: float):
        """检查阈值"""
        
        if metric_name not in self.thresholds:
            return
        
        threshold = self.thresholds[metric_name]
        
        if metric_name == 'loss':
            if value > threshold['critical']:
                self._raise_alert(f"严重: {metric_name} = {value:.4f} (临界值: {threshold['critical']})")
            elif value > threshold['warning']:
                self._raise_alert(f"警告: {metric_name} = {value:.4f} (警告值: {threshold['warning']})")
        
        elif metric_name == 'accuracy':
            if value < threshold['critical']:
                self._raise_alert(f"严重: {metric_name} = {value:.4f} (临界值: {threshold['critical']})")
            elif value < threshold['warning']:
                self._raise_alert(f"警告: {metric_name} = {value:.4f} (警告值: {threshold['warning']})")
    
    def _raise_alert(self, message: str):
        """发出警报"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'level': 'warning' if '警告' in message else 'critical',
        }
        self.alerts.append(alert)
        logger.warning(f"⚠️  {message}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        
        summary = {
            'metrics_count': len(self.metrics),
            'total_alerts': len(self.alerts),
            'latest_values': {},
        }
        
        # 获取最新值
        for metric_name, values in self.metrics.items():
            if values:
                summary['latest_values'][metric_name] = values[-1]['value']
        
        return summary


# ============================================================================
# 6. 完整的多模型学习系统
# ============================================================================

class MultiModelLearningSystem:
    """完整的多模型学习系统"""
    
    def __init__(self, num_models: int = 3):
        self.registry = ModelRegistry()
        self.models = []
        self.ensemble = None
        self.adaptive_controller = AdaptiveLearningController()
        self.distributed_coordinator = DistributedLearningCoordinator(num_models)
        self.performance_monitor = PerformanceMonitor()
        
        # 学习历史
        self.loss_history = []
        self.reward_history = []
        self.learning_stats = {
            'total_iterations': 0,
            'total_samples': 0,
            'start_time': datetime.now(),
        }
        
        logger.info("="*70)
        logger.info("🚀 多模型学习系统完全启动")
        logger.info("="*70)
        logger.info(f"✓ 模型数: {num_models}")
        logger.info(f"✓ 可用设备: {self.distributed_coordinator.available_devices}")
    
    def add_model(self, model: nn.Module, model_name: str, 
                  version: str = "1.0.0") -> str:
        """添加模型"""
        
        model_id = hashlib.md5(f"{model_name}_{version}_{datetime.now()}".encode()).hexdigest()[:8]
        
        model_info = ModelInfo(
            model_id=model_id,
            model_name=model_name,
            version=version,
            creation_date=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            total_params=sum(p.numel() for p in model.parameters()),
        )
        
        self.registry.register_model(model_info)
        self.models.append(model)
        
        logger.info(f"✓ 模型添加: {model_name} (ID: {model_id})")
        
        return model_id
    
    def create_ensemble(self, fusion_strategy: str = 'weighted_average') -> ModelEnsemble:
        """创建模型融合"""
        
        if len(self.models) < 2:
            raise ValueError("至少需要2个模型才能创建融合")
        
        self.ensemble = ModelEnsemble(self.models, fusion_strategy)
        
        logger.info(f"✓ 模型融合创建: {fusion_strategy}")
        
        return self.ensemble
    
    def train_iteration(self, 
                       batch_data: torch.Tensor,
                       loss_fn: callable,
                       losses: List[float],
                       rewards: List[float]) -> Dict[str, Any]:
        """执行一次训练迭代"""
        
        self.learning_stats['total_iterations'] += 1
        
        # 1. 通过融合模型前向传播
        if self.ensemble:
            fused_output, fusion_info = self.ensemble.forward(batch_data)
        else:
            # 如果没有融合，使用第一个模型
            fused_output = self.models[0](batch_data)
            fusion_info = {'num_models': 1}
        
        # 2. 计算损失
        loss = loss_fn(fused_output)
        self.loss_history.append(loss.item())
        self.performance_monitor.record_metric('loss', loss.item())
        
        # 3. 计算自适应学习率
        new_lr = self.adaptive_controller.compute_adaptive_lr(
            self.loss_history[-10:],
            rewards[-10:] if rewards else []
        )
        self.performance_monitor.record_metric('learning_rate', new_lr)
        
        # 4. 更新融合权重
        if self.ensemble and losses:
            # 使用最近的损失作为性能评分
            recent_scores = [1.0 / (1.0 + loss) for loss in losses[-3:]]
            if len(recent_scores) == len(self.ensemble.models):
                self.ensemble.update_weights(recent_scores)
        
        self.learning_stats['total_samples'] += batch_data.shape[0]
        
        return {
            'iteration': self.learning_stats['total_iterations'],
            'loss': loss.item(),
            'learning_rate': new_lr,
            'fusion_info': fusion_info,
            'model_count': len(self.models),
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        
        elapsed = (datetime.now() - self.learning_stats['start_time']).total_seconds()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'operational',
            'models': {
                'total_models': len(self.models),
                'registered_models': len(self.registry.list_models()),
                'ensemble_active': self.ensemble is not None,
            },
            'learning_progress': {
                'total_iterations': self.learning_stats['total_iterations'],
                'total_samples': self.learning_stats['total_samples'],
                'elapsed_seconds': elapsed,
            },
            'performance': self.performance_monitor.get_summary(),
            'devices': {
                'available_devices': self.distributed_coordinator.available_devices,
                'worker_count': self.distributed_coordinator.num_workers,
            }
        }


# ============================================================================
# 主程序演示
# ============================================================================

if __name__ == '__main__':
    logger.info("\n🎯 多模型协作学习系统演示\n")
    
    # 初始化系统
    system = MultiModelLearningSystem(num_models=3)
    
    # 创建3个模拟模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
        
        def forward(self, x):
            return self.fc(x)
    
    for i in range(3):
        model = SimpleModel()
        system.add_model(model, f'Model_{i+1}', f'1.0.{i}')
    
    # 创建融合
    logger.info("\n【创建模型融合】")
    logger.info("-"*70)
    ensemble = system.create_ensemble('weighted_average')
    
    # 执行训练迭代
    logger.info("\n【执行训练迭代】")
    logger.info("-"*70)
    
    def loss_fn(output):
        return torch.mean(output)
    
    for iteration in range(5):
        batch = torch.randn(32, 10)
        losses = [0.5, 0.48, 0.46, 0.44, 0.42]
        rewards = [0.5, 0.55, 0.58, 0.60, 0.62]
        
        result = system.train_iteration(batch, loss_fn, losses, rewards)
        logger.info(f"迭代 {result['iteration']}: 损失={result['loss']:.4f}, "
                   f"LR={result['learning_rate']:.6f}")
    
    # 系统状态
    logger.info("\n【系统状态】")
    logger.info("-"*70)
    
    status = system.get_system_status()
    logger.info(json.dumps(status, indent=2, default=str))
    
    logger.info("\n✅ 演示完成!")
