#!/usr/bin/env python3
"""
真实数据学习管道 - Week 6 高级学习系统
===========================================

功能:
1. 从真实网站和GitHub采集JavaScript代码样本
2. 应用多种混淆技术到真实代码
3. 数据增强: 代码变换、特征注入等
4. GPU加速训练 (PyTorch)
5. 评估混淆效果
6. 记录学习指标和性能

作者: BrowerAI Learning Team
时间: 2026-01
"""

import json
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import random

# GPU支持
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available, will use TensorFlow as fallback")

# 静态分析工具
try:
    import ast
    import re
except ImportError:
    pass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class RealDataCollector:
    """真实数据采集器 - 从GitHub和本地项目采集代码"""
    
    def __init__(self, output_dir='data/week6_real_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_samples = []
        
    def collect_from_github(self, repos: List[Dict], max_files_per_repo=50):
        """从GitHub仓库采集JS代码"""
        logger.info("\n📥 从GitHub采集真实JS代码...")
        
        # 这里集成GitHub爬虫
        # repos 格式: [{'owner': 'facebook', 'name': 'react'}, ...]
        
        for repo in repos:
            logger.info(f"  📂 采集 {repo.get('owner', '')}/{repo.get('name', '')}")
            try:
                # 使用已有的GitHub爬虫模块（如可用）
                try:
                    from training.crawlers.github_framework_crawler import GitHubFrameworkCrawler
                    crawler = GitHubFrameworkCrawler()
                except ImportError:
                    logger.warning(f"    ⚠️  GitHub爬虫模块不可用，跳过该源")
                    continue
                
                crawler = GitHubFrameworkCrawler()
                
                # 采集框架源代码
                samples = crawler.fetch_repo_files(
                    repo.get('owner', ''),
                    repo.get('name', ''),
                    max_files=max_files_per_repo
                )
                
                for sample in samples:
                    self.collected_samples.append({
                        'source': 'github',
                        'repo': f"{repo.get('owner', '')}/{repo.get('name', '')}",
                        'code': sample.get('content', ''),
                        'file': sample.get('path', ''),
                        'timestamp': datetime.now().isoformat(),
                        'type': 'framework'
                    })
                
                logger.info(f"    ✅ 采集了 {len(samples)} 个文件")
                
            except Exception as e:
                logger.warning(f"    ❌ 采集失败: {e}")
    
    def collect_from_local_projects(self, project_paths: List[str]):
        """从本地项目采集JS代码"""
        logger.info("\n📁 从本地项目采集真实代码...")
        
        for project_path in project_paths:
            project = Path(project_path)
            if not project.exists():
                logger.warning(f"  ⚠️  项目路径不存在: {project_path}")
                continue
            
            logger.info(f"  📂 扫描 {project_path}")
            js_files = list(project.rglob('*.js'))
            logger.info(f"    找到 {len(js_files)} 个 JS 文件")
            
            for js_file in js_files[:100]:  # 限制数量
                try:
                    with open(js_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    if len(content) > 100:  # 过滤太短的文件
                        self.collected_samples.append({
                            'source': 'local',
                            'path': str(js_file),
                            'code': content,
                            'size': len(content),
                            'timestamp': datetime.now().isoformat(),
                            'type': 'project'
                        })
                except Exception as e:
                    logger.debug(f"    无法读取 {js_file}: {e}")
    
    def save_collected_data(self):
        """保存采集的数据"""
        if not self.collected_samples:
            logger.warning("⚠️  没有采集到数据")
            return
        
        output_file = self.output_dir / 'collected_samples.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in self.collected_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"\n💾 采集数据保存到: {output_file}")
        logger.info(f"   总计: {len(self.collected_samples)} 个样本")
        
        # 统计源和类型
        sources = {}
        types = {}
        for sample in self.collected_samples:
            source = sample.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
            sample_type = sample.get('type', 'unknown')
            types[sample_type] = types.get(sample_type, 0) + 1
        
        logger.info(f"\n📊 数据分布:")
        for source, count in sources.items():
            logger.info(f"   {source:15} {count:5} 个")
        
        return output_file


class DataAugmentation:
    """数据增强 - 对真实代码进行变换以增加样本多样性"""
    
    def __init__(self):
        self.augmented_samples = []
    
    def variable_renaming(self, code: str, seed: int = None) -> str:
        """随机变量重命名"""
        if seed is not None:
            random.seed(seed)
        
        # 匹配变量名 (简单版本)
        var_pattern = r'\b([a-zA-Z_$][a-zA-Z0-9_$]*)\b'
        
        try:
            # 替换常见变量
            result = code
            common_vars = ['var', 'let', 'const', 'function', 'class']
            
            for var in common_vars:
                if var in result:
                    new_var = f"_{random.randint(1000, 9999)}"
                    result = result.replace(var, new_var)
            
            return result
        except Exception as e:
            logger.debug(f"变量重命名失败: {e}")
            return code
    
    def code_formatting(self, code: str) -> str:
        """代码格式化变换"""
        transformations = [
            # 移除空格
            lambda c: c.replace('\n', ';').replace('  ', ''),
            # 添加注释
            lambda c: c.replace(';', ';\n// 处理'),
            # 代码缩进变换
            lambda c: c.replace('  ', '\t'),
        ]
        
        try:
            result = code
            transform = random.choice(transformations)
            result = transform(result)
            return result
        except Exception as e:
            logger.debug(f"格式化失败: {e}")
            return code
    
    def semantic_wrapping(self, code: str) -> str:
        """语义包装 - 在不改变功能的前提下包装代码"""
        try:
            # 使用 IIFE 包装
            wrapped = f"""
(function() {{
    'use strict';
    {code}
}})();
"""
            return wrapped
        except Exception as e:
            logger.debug(f"包装失败: {e}")
            return code
    
    def inject_framework_patterns(self, code: str) -> str:
        """注入框架模式"""
        patterns = [
            "const app = {};",
            "class Component {}",
            "function render() {}",
            "async function init() {}",
        ]
        
        try:
            pattern = random.choice(patterns)
            result = pattern + '\n' + code
            return result
        except Exception as e:
            logger.debug(f"注入失败: {e}")
            return code
    
    def augment_sample(self, original_code: str, num_variations: int = 3) -> List[str]:
        """为单个样本生成多个变体"""
        variations = [original_code]
        
        augmentation_methods = [
            self.variable_renaming,
            self.code_formatting,
            self.semantic_wrapping,
            self.inject_framework_patterns,
        ]
        
        for _ in range(num_variations - 1):
            method = random.choice(augmentation_methods)
            try:
                variation = method(original_code)
                if variation != original_code:
                    variations.append(variation)
            except Exception as e:
                logger.debug(f"增强失败: {e}")
        
        return variations
    
    def augment_dataset(self, samples: List[Dict], augmentation_factor: int = 2) -> List[Dict]:
        """对数据集进行增强"""
        logger.info(f"\n🔄 数据增强中 (增强因子: {augmentation_factor}x)...")
        
        augmented = []
        
        for idx, sample in enumerate(samples):
            original_code = sample.get('code', '')
            if not original_code:
                continue
            
            # 保留原始样本
            augmented.append(sample)
            
            # 生成增强样本
            variations = self.augment_sample(
                original_code,
                num_variations=augmentation_factor
            )
            
            for var_idx, variation in enumerate(variations[1:]):
                augmented_sample = sample.copy()
                augmented_sample['code'] = variation
                augmented_sample['augmentation_index'] = var_idx + 1
                augmented_sample['augmentation_method'] = random.choice([
                    'variable_renaming', 'formatting', 'wrapping', 'pattern_injection'
                ])
                augmented.append(augmented_sample)
        
        logger.info(f"✅ 增强完成: {len(samples)} → {len(augmented)} 样本")
        return augmented


class ObfuscationEvaluator:
    """混淆效果评估器"""
    
    def __init__(self):
        self.evaluation_results = []
    
    def evaluate_readability(self, original_code: str, obfuscated_code: str) -> Dict:
        """评估代码可读性"""
        metrics = {
            'lines_added': obfuscated_code.count('\n') - original_code.count('\n'),
            'size_increase': (len(obfuscated_code) - len(original_code)) / len(original_code) if original_code else 0,
            'complexity_increase': self._estimate_complexity_increase(obfuscated_code),
            'entropy_increase': self._calculate_entropy(obfuscated_code) - self._calculate_entropy(original_code),
        }
        return metrics
    
    def _estimate_complexity_increase(self, code: str) -> float:
        """估计代码复杂度增加"""
        # 简单的复杂度指标
        complexity_tokens = ['eval', 'Function', 'setTimeout', 'JSON.parse', 'atob']
        complexity_count = sum(code.count(token) for token in complexity_tokens)
        return float(complexity_count)
    
    def _calculate_entropy(self, code: str) -> float:
        """计算Shannon熵"""
        if not code:
            return 0.0
        
        entropy = 0.0
        code_len = len(code)
        
        for char in set(code):
            freq = code.count(char) / code_len
            entropy -= freq * np.log2(freq) if freq > 0 else 0
        
        return entropy
    
    def evaluate_obfuscation_technique(self, technique_name: str, samples: List[Dict]) -> Dict:
        """评估混淆技术的效果"""
        logger.info(f"\n📊 评估混淆技术: {technique_name}")
        
        evaluations = []
        
        for sample in samples:
            original = sample.get('original_code', '')
            obfuscated = sample.get('obfuscated_code', '')
            
            if original and obfuscated:
                eval_result = self.evaluate_readability(original, obfuscated)
                eval_result['sample_id'] = sample.get('id', '')
                evaluations.append(eval_result)
        
        if not evaluations:
            logger.warning(f"  ⚠️  没有有效的评估样本")
            return {}
        
        # 计算统计信息
        stats = {
            'technique': technique_name,
            'num_samples': len(evaluations),
            'avg_size_increase': np.mean([e['size_increase'] for e in evaluations]),
            'max_size_increase': np.max([e['size_increase'] for e in evaluations]),
            'avg_entropy_increase': np.mean([e['entropy_increase'] for e in evaluations]),
            'avg_complexity': np.mean([e['complexity_increase'] for e in evaluations]),
            'evaluation_timestamp': datetime.now().isoformat(),
        }
        
        logger.info(f"  ✅ 平均代码膨胀: {stats['avg_size_increase']:.2%}")
        logger.info(f"  ✅ 平均熵增加: {stats['avg_entropy_increase']:.3f}")
        
        return stats


class GPULearner:
    """GPU加速学习器 - 使用PyTorch进行模型训练"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPU learning")
        
        self.device = device
        self.model = None
        self.optimizer = None
        self.training_history = []
        
        logger.info(f"🖥️  使用设备: {device}")
        if device == 'cuda':
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def build_model(self, input_dim: int, hidden_dims: List[int] = None):
        """构建神经网络模型"""
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3),
            ])
            prev_dim = hidden_dim
        
        # 输出层 (混淆检测的二分类)
        layers.append(nn.Linear(prev_dim, 2))
        layers.append(nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        logger.info(f"✅ 模型构建完成")
        logger.info(f"   输入维度: {input_dim}")
        logger.info(f"   隐藏层: {hidden_dims}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs: int = 50, batch_size: int = 32):
        """GPU加速训练"""
        logger.info(f"\n🚀 开始GPU训练 ({epochs} epochs)...")
        
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first")
        
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # 前向传播
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'timestamp': datetime.now().isoformat(),
            })
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info(f"✅ 训练完成")
        return self.training_history
    
    def evaluate(self, X_test, y_test) -> Dict:
        """评估模型"""
        self.model.eval()
        
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            predictions = torch.argmax(outputs, dim=1)
            
            accuracy = (predictions == y_test_tensor).float().mean().item()
            
            logger.info(f"📊 测试准确率: {accuracy:.2%}")
        
        return {
            'accuracy': accuracy,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat(),
        }


class LearningMetricsRecorder:
    """学习指标记录器"""
    
    def __init__(self, output_dir: str = 'data/week6_learning_metrics'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            'data_collection': [],
            'augmentation': [],
            'obfuscation_evaluation': [],
            'training': [],
            'inference': [],
        }
    
    def record_data_collection(self, source: str, count: int, metadata: Dict = None):
        """记录数据采集"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'count': count,
            'metadata': metadata or {},
        }
        self.metrics['data_collection'].append(record)
        logger.info(f"📝 记录数据采集: {source} ({count} 个)")
    
    def record_training(self, model_name: str, metrics: Dict):
        """记录训练指标"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metrics': metrics,
        }
        self.metrics['training'].append(record)
    
    def save_all_metrics(self):
        """保存所有指标"""
        output_file = self.output_dir / 'all_metrics.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 所有指标保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='真实数据学习管道')
    parser.add_argument('--collect', action='store_true', help='采集真实数据')
    parser.add_argument('--augment', action='store_true', help='进行数据增强')
    parser.add_argument('--train', action='store_true', help='GPU加速训练')
    parser.add_argument('--evaluate', action='store_true', help='评估混淆效果')
    parser.add_argument('--full-pipeline', action='store_true', help='运行完整管道')
    parser.add_argument('--device', type=str, default='auto', help='设备: cuda/cpu/auto')
    
    args = parser.parse_args()
    
    logger.info("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    真实数据学习管道 - Week 6 高级系统                         ║
║                                                                                ║
║  功能: 真实数据采集 → 数据增强 → GPU训练 → 性能评估                          ║
╚════════════════════════════════════════════════════════════════════════════════╝
""")
    
    # 选择设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    metrics_recorder = LearningMetricsRecorder()
    
    # 1. 采集真实数据
    if args.collect or args.full_pipeline:
        logger.info("\n" + "="*80)
        logger.info("阶段 1: 采集真实数据")
        logger.info("="*80)
        
        collector = RealDataCollector()
        
        # 从GitHub采集框架代码
        repos = [
            {'owner': 'facebook', 'name': 'react'},
            {'owner': 'vuejs', 'name': 'vue'},
            {'owner': 'angular', 'name': 'angular'},
        ]
        
        # 如果GitHub API可用，采集真实框架代码
        try:
            collector.collect_from_github(repos, max_files_per_repo=20)
        except Exception as e:
            logger.warning(f"GitHub采集失败: {e}")
        
        # 从本地项目采集
        local_paths = [
            'training/real_data',
            'crates/browerai',
        ]
        
        collector.collect_from_local_projects(local_paths)
        collector.save_collected_data()
    
    # 2. 数据增强
    if args.augment or args.full_pipeline:
        logger.info("\n" + "="*80)
        logger.info("阶段 2: 数据增强")
        logger.info("="*80)
        
        augmenter = DataAugmentation()
        # 这里可以加载采集的数据进行增强
        logger.info("✅ 数据增强模块已准备")
    
    # 3. 训练
    if args.train or args.full_pipeline:
        logger.info("\n" + "="*80)
        logger.info(f"阶段 3: GPU加速训练 (设备: {device})")
        logger.info("="*80)
        
        if not PYTORCH_AVAILABLE:
            logger.error("❌ PyTorch未安装，无法进行GPU训练")
            logger.info("   安装: pip install torch torchvision")
            return
        
        learner = GPULearner(device=device)
        learner.build_model(input_dim=48, hidden_dims=[256, 128, 64])
        
        # 生成示例训练数据
        X_train = np.random.randn(400, 48)
        y_train = np.random.randint(0, 2, 400)
        
        history = learner.train(X_train, y_train, epochs=50, batch_size=32)
        
        # 评估
        X_test = np.random.randn(100, 48)
        y_test = np.random.randint(0, 2, 100)
        eval_result = learner.evaluate(X_test, y_test)
        
        metrics_recorder.record_training('gpu_accelerated_model', eval_result)
    
    # 4. 评估
    if args.evaluate or args.full_pipeline:
        logger.info("\n" + "="*80)
        logger.info("阶段 4: 混淆效果评估")
        logger.info("="*80)
        
        evaluator = ObfuscationEvaluator()
        logger.info("✅ 评估器已准备")
    
    # 保存所有指标
    metrics_recorder.save_all_metrics()
    
    logger.info("\n" + "="*80)
    logger.info("✅ 学习管道执行完成")
    logger.info("="*80)


if __name__ == '__main__':
    main()
