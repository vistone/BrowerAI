#!/usr/bin/env python3
"""
真实GPU框架检测模型
使用PyTorch在GTX 1060上训练变压器模型识别JavaScript框架
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib
import logging

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JSFrameworkDataset(Dataset):
    """JavaScript框架数据集"""
    
    FRAMEWORK_LABELS = {
        'react': 0, 'vue': 1, 'angular': 2, 'svelte': 3,
        'next': 4, 'nuxt': 5, 'remix': 6, 'gatsby': 7,
        'express': 8, 'fastapi': 9, 'hapi': 10, 'koa': 11,
        'webpack': 12, 'vite': 13, 'parcel': 14, 'esbuild': 15,
        'jest': 16, 'vitest': 17, 'mocha': 18, 'chai': 19,
        'lodash': 20, 'axios': 21, 'fetch': 22, 'node': 23,
    }
    
    def __init__(self, data_file: Path, vocab_size: int = 5000, max_length: int = 512):
        """
        初始化数据集
        
        Args:
            data_file: 训练数据文件(JSONL格式)
            vocab_size: 词汇表大小
            max_length: 最大序列长度
        """
        self.data = []
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab = self._build_vocab()
        
        if data_file.exists():
            self._load_data(data_file)
        else:
            logger.warning(f"数据文件不存在: {data_file}")
    
    def _build_vocab(self) -> Dict[str, int]:
        """构建词汇表"""
        vocab = {}
        
        # 特殊令牌
        special_tokens = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>']
        for i, token in enumerate(special_tokens):
            vocab[token] = i
        
        # JavaScript关键字
        js_keywords = [
            'function', 'const', 'let', 'var', 'return', 'if', 'else', 'for',
            'while', 'do', 'switch', 'case', 'break', 'continue', 'import',
            'export', 'default', 'async', 'await', 'try', 'catch', 'finally',
            'class', 'extends', 'new', 'this', 'super', 'static', 'get', 'set',
        ]
        
        for keyword in js_keywords:
            vocab[keyword] = len(vocab)
        
        # 框架特定标记
        framework_markers = [
            'React', 'Vue', 'Angular', 'Svelte', 'Next', 'Nuxt',
            'redux', 'vuex', 'pinia', 'mobx', 'recoil',
            'component', 'props', 'state', 'dispatch', 'action',
            'useState', 'useEffect', 'useContext', 'useReducer',
            'mounted', 'created', 'beforeCreate', 'beforeMount',
            '@Component', '@NgModule', '@Directive',
        ]
        
        for marker in framework_markers:
            vocab[marker] = len(vocab)
        
        # 填充到vocab_size
        num_tokens = len(vocab)
        for i in range(num_tokens, self.vocab_size):
            vocab[f'<TOKEN_{i}>'] = i
        
        return vocab
    
    def _load_data(self, data_file: Path):
        """加载JSONL格式的数据"""
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        self.data.append(record)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"加载了 {len(self.data)} 条记录")
    
    def _tokenize(self, code: str) -> List[int]:
        """将代码令牌化"""
        # 简单的令牌化 - 按字符和关键字分割
        tokens = []
        
        # 处理框架特定标记
        for token, token_id in self.vocab.items():
            if token not in ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>']:
                code = code.replace(token, f' {token} ')
        
        # 按空格分割
        words = code.split()
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab['<UNK>'])
        
        # 填充或截断
        if len(tokens) < self.max_length:
            tokens += [self.vocab['<PAD>']] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        return tokens
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx >= len(self.data):
            return torch.zeros(self.max_length, dtype=torch.long), 0
        
        record = self.data[idx]
        
        # 从文件路径推断框架
        source_file = record.get('source_file', '')
        framework = self._infer_framework(source_file)
        framework_id = self.FRAMEWORK_LABELS.get(framework, 0)
        
        # 令牌化
        tokens = self._tokenize(record.get('code', ''))
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        return tokens_tensor, framework_id
    
    def _infer_framework(self, source_file: str) -> str:
        """从源文件路径推断框架"""
        source_file_lower = source_file.lower()
        
        for framework in self.FRAMEWORK_LABELS.keys():
            if framework in source_file_lower:
                return framework
        
        return 'node'  # 默认


class FrameworkDetectorModel(nn.Module):
    """框架检测变压器模型"""
    
    def __init__(self, vocab_size: int = 5000, hidden_size: int = 256,
                 num_layers: int = 2, num_heads: int = 4,
                 max_length: int = 512, num_frameworks: int = 24):
        """
        初始化模型
        
        Args:
            vocab_size: 词汇表大小
            hidden_size: 隐藏层大小 (GTX 1060优化: 256而不是768)
            num_layers: 变压器层数
            num_heads: 注意力头数
            max_length: 最大序列长度
            num_frameworks: 框架数量
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_length = max_length
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_length, hidden_size)
        
        # 变压器编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_frameworks)
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入令牌ID (batch_size, seq_length)
        
        Returns:
            logits: 框架分类logits (batch_size, num_frameworks)
        """
        batch_size, seq_length = input_ids.shape
        
        # 嵌入
        embeddings = self.embedding(input_ids)
        
        # 位置嵌入
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.pos_embedding(positions)
        
        # 相加
        x = embeddings + pos_embeddings
        
        # 变压器
        transformer_out = self.transformer(x)
        
        # 取[CLS]令牌(第一个令牌)的输出
        cls_output = transformer_out[:, 0, :]
        
        # 分类
        logits = self.classifier(cls_output)
        
        return logits


class FrameworkDetectorTrainer:
    """框架检测模型训练器"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size: int = 4, learning_rate: float = 1e-4):
        """
        初始化训练器
        
        Args:
            device: 设备(cuda/cpu)
            batch_size: 批大小(GTX 1060优化为4)
            learning_rate: 学习率
        """
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        logger.info(f"设备: {device}")
        if device == 'cuda':
            props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {props.name}")
            logger.info(f"GPU显存: {props.total_memory / 1e9:.2f}GB")
    
    def train(self, train_loader: DataLoader, model: nn.Module,
              epochs: int = 5, validation_loader: Optional[DataLoader] = None):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            model: 模型
            epochs: 训练轮数
            validation_loader: 验证数据加载器
        """
        model = model.to(self.device)
        
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (input_ids, labels) in enumerate(train_loader):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                logits = model(input_ids)
                loss = criterion(logits, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1} - "
                              f"Loss: {loss.item():.4f}")
            
            # 每轮统计
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            logger.info(f"Epoch {epoch+1}/{epochs} - 平均损失: {avg_loss:.4f}, "
                       f"准确率: {accuracy:.2f}%")
            
            # 验证阶段
            if validation_loader:
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for input_ids, labels in validation_loader:
                        input_ids = input_ids.to(self.device)
                        labels = labels.to(self.device)
                        
                        logits = model(input_ids)
                        loss = criterion(logits, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(logits.data, 1)
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)
                
                val_accuracy = 100 * val_correct / val_total
                logger.info(f"验证 - 损失: {val_loss/len(validation_loader):.4f}, "
                           f"准确率: {val_accuracy:.2f}%")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🤖 JavaScript框架检测GPU训练器")
    print("="*70)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("⚠️  未检测到GPU,使用CPU(会很慢)")
        device = 'cpu'
    
    # 检查数据
    data_file = Path('real_data/obfuscated_code/training_pairs.jsonl')
    if not data_file.exists():
        print(f"\n⚠️  数据文件不存在: {data_file}")
        print("   请先运行:")
        print("   1. python3 training/github_framework_crawler.py")
        print("   2. python3 training/real_code_obfuscator.py")
        return
    
    print(f"\n📂 数据文件: {data_file}")
    
    # 创建数据集和加载器
    print("\n📊 初始化数据集...")
    dataset = JSFrameworkDataset(data_file)
    
    if len(dataset) == 0:
        print("❌ 数据集为空!")
        return
    
    print(f"✅ 加载了 {len(dataset)} 条记录")
    
    # 分割训练/验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # 创建模型
    print("\n🤖 构建模型...")
    model = FrameworkDetectorModel(
        vocab_size=5000,
        hidden_size=256,  # 针对GTX 1060优化
        num_layers=2,     # 减少层数
        num_heads=4,
        max_length=512,
        num_frameworks=24
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数: {total_params:,}")
    
    # 训练
    print("\n🚀 开始训练...")
    trainer = FrameworkDetectorTrainer(device=device, batch_size=4, learning_rate=1e-4)
    
    try:
        trainer.train(train_loader, model, epochs=3, validation_loader=val_loader)
        
        # 保存模型
        output_dir = Path('models/local')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / 'framework_detector_gpu.pt'
        torch.save(model.state_dict(), model_path)
        print(f"\n✅ 模型已保存: {model_path}")
        
        # 转换为ONNX
        print("\n📦 转换为ONNX...")
        try:
            dummy_input = torch.randint(0, 5000, (1, 512)).to(device)
            torch.onnx.export(
                model.to(device),
                dummy_input,
                str(output_dir / 'framework_detector_gpu.onnx'),
                input_names=['input_ids'],
                output_names=['logits'],
                opset_version=14
            )
            print("✅ ONNX模型已保存")
        except Exception as e:
            print(f"⚠️  ONNX转换失败: {e}")
    
    except KeyboardInterrupt:
        print("\n⚠️  训练被中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")


if __name__ == '__main__':
    main()
