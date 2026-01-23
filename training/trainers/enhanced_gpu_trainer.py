#!/usr/bin/env python3
"""
增强版GPU框架检测模型训练器
- 支持更大的数据集
- 更高的模型维度
- 更多训练轮数
- 更好的数据增强
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from collections import Counter
import random
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NPMFrameworkDataset(Dataset):
    """NPM包框架识别数据集"""
    
    PACKAGE_LABELS = {
        # 前端框架
        'react': 0, 'vue': 1, 'angular': 2, 'svelte': 3, 'ember': 4,
        # 全栈框架
        'next': 5, 'nuxt': 6, 'gatsby': 7, 'remix': 8, 'sveltekit': 9,
        # 后端框架
        'express': 10, 'fastify': 11, 'koa': 12, 'nestjs': 13, 'hapi': 14,
        # 构建工具
        'webpack': 15, 'vite': 16, 'rollup': 17, 'esbuild': 18,
        # NPM包
        'lodash': 19, 'axios': 20, 'ramda': 21, 'underscore': 22,
    }
    
    def __init__(self, data_file: Path, vocab_size: int = 10000, max_length: int = 512):
        self.data = []
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab = self._build_vocab()
        self.label_counts = Counter()
        
        if data_file.exists():
            self._load_data(data_file)
        else:
            logger.warning(f"数据文件不存在: {data_file}")
    
    def _build_vocab(self) -> Dict[str, int]:
        """构建更大的词汇表"""
        vocab = {}
        
        # 特殊令牌
        special_tokens = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>', '<NUM>', '<STR>']
        for i, token in enumerate(special_tokens):
            vocab[token] = i
        
        # JavaScript/Python关键字和操作符
        keywords = [
            'function', 'const', 'let', 'var', 'return', 'if', 'else', 'for', 'while',
            'do', 'switch', 'case', 'break', 'continue', 'import', 'export', 'default',
            'async', 'await', 'try', 'catch', 'finally', 'class', 'extends', 'new',
            'this', 'super', 'static', 'get', 'set', 'of', 'in', 'typeof', 'instanceof',
            'true', 'false', 'null', 'undefined', 'void', 'delete', 'yield', 'from',
            'as', 'require', 'module', 'exports', 'constructor', 'prototype',
            'Symbol', 'Promise', 'async', 'await', 'then', 'catch', 'finally',
        ]
        
        for kw in keywords:
            vocab[kw] = len(vocab)
        
        # 框架特定标记
        framework_markers = [
            'React', 'useState', 'useEffect', 'useContext', 'useReducer', 'useCallback',
            'useMemo', 'useRef', 'Hook', 'JSX', 'Component', 'PureComponent',
            'Vue', 'computed', 'watch', 'mounted', 'created', 'beforeCreate', 'beforeMount',
            'template', 'data', 'methods', 'props', 'Directive', 'Mixin',
            'Angular', '@Component', '@NgModule', '@Directive', '@Pipe', 'Injectable',
            'Observable', 'Subject', 'BehaviorSubject', 'ReplaySubject', 'Service',
            'Svelte', 'animation', 'transition', 'store', 'action', 'tick',
            'Next', 'pages', 'api', 'getServerSideProps', 'getStaticProps', 'getStaticPaths',
            'Nuxt', 'middleware', 'plugins', 'layout', 'asyncData', 'fetch',
            'Express', 'app', 'router', 'middleware', 'request', 'response', 'next',
            'Fastify', 'register', 'route', 'handler', 'preHandler', 'onRequest',
            'Koa', 'context', 'ctx', 'state', 'body', 'status', 'header',
            'lodash', 'map', 'filter', 'reduce', 'find', 'some', 'every',
            'axios', 'request', 'get', 'post', 'put', 'delete', 'patch', 'interceptor',
            'ramda', 'compose', 'pipe', 'curry', 'partial', 'flip', 'partial',
        ]
        
        for marker in framework_markers:
            if marker not in vocab:
                vocab[marker] = len(vocab)
        
        # 常见操作符和符号
        symbols = ['+', '-', '*', '/', '%', '=', '==', '===', '!=', '!==',
                  '<', '>', '<=', '>=', '&&', '||', '!', '&', '|', '^', '~',
                  '?', ':', '.', ',', ';', '(', ')', '[', ']', '{', '}',
                  '=>', '...', '++', '--', '+=', '-=', '*=', '/=', '%=', '**']
        
        for sym in symbols:
            if sym not in vocab:
                vocab[sym] = len(vocab)
        
        # 填充到vocab_size
        num_tokens = len(vocab)
        for i in range(num_tokens, self.vocab_size):
            vocab[f'<TOKEN_{i}>'] = i
        
        return vocab
    
    def _load_data(self, data_file: Path):
        """加载JSONL格式数据"""
        with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        self.data.append(record)
                        
                        # 统计标签 - 优先使用 package 字段，备选 source_file
                        source = record.get('package', record.get('source_file', '')).lower()
                        label_assigned = False
                        
                        for pkg, label in self.PACKAGE_LABELS.items():
                            if pkg in source:
                                self.label_counts[label] += 1
                                label_assigned = True
                                break
                        
                        # 如果未分配标签，使用默认标签
                        if not label_assigned:
                            self.label_counts[22] += 1  # 'other' 标签
                    except:
                        pass
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple:
        """返回(输入tokens, 标签)"""
        record = self.data[idx]
        
        # 使用混淆代码作为输入（更难的学习任务）
        code = record.get('obfuscated', record.get('original', ''))
        
        # 分词
        tokens = []
        i = 0
        while i < len(code) and len(tokens) < self.max_length - 2:
            if code[i].isspace():
                i += 1
                continue
            
            # 尝试匹配关键字
            matched = False
            for word in sorted(self.vocab.keys(), key=len, reverse=True):
                if code[i:i+len(word)] == word and code[i:i+len(word)] != '<':
                    tokens.append(self.vocab.get(word, self.vocab['<UNK>']))
                    i += len(word)
                    matched = True
                    break
            
            if not matched:
                # 默认加入unk token
                tokens.append(self.vocab['<UNK>'])
                i += 1
        
        # 填充到max_length
        tokens = [self.vocab['<CLS>']] + tokens + [self.vocab['<SEP>']]
        while len(tokens) < self.max_length:
            tokens.append(self.vocab['<PAD>'])
        tokens = tokens[:self.max_length]
        
        # 推断框架标签
        framework_label = self._infer_framework(record.get('source_file', ''))
        
        return torch.tensor(tokens, dtype=torch.long), framework_label
    
    def _infer_framework(self, source_file: str) -> int:
        """从源文件推断框架标签"""
        source_lower = source_file.lower()
        
        for package, label in self.PACKAGE_LABELS.items():
            if package in source_lower:
                return label
        
        return 0  # 默认React


class EnhancedTransformerModel(nn.Module):
    """增强版变压器模型 - 更高维度"""
    
    def __init__(self, vocab_size: int = 10000, hidden_size: int = 512,
                 num_layers: int = 3, num_heads: int = 8,
                 max_length: int = 512, num_classes: int = 23):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_length = max_length
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_length, hidden_size)
        self.dropout = nn.Dropout(0.15)
        
        # 变压器编码器（更多层，更多头）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=2048,
            batch_first=True,
            dropout=0.15,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        
        # 嵌入
        embeddings = self.embedding(input_ids)
        
        # 位置嵌入
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.pos_embedding(positions)
        
        # 相加+dropout
        x = self.dropout(embeddings + pos_embeddings)
        
        # 变压器
        transformer_out = self.transformer(x)
        
        # 取CLS令牌和平均池化
        cls_output = transformer_out[:, 0, :]
        mean_output = transformer_out.mean(dim=1)
        
        # 合并
        combined = cls_output + mean_output
        
        # 分类
        logits = self.classifier(combined)
        
        return logits


class EnhancedTrainer:
    """增强版训练器"""
    
    def __init__(self, device: str = 'cuda', batch_size: int = 16, 
                 learning_rate: float = 5e-5):
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        logger.info(f"设备: {device}")
        if device == 'cuda':
            props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {props.name}")
    
    def train(self, train_loader: DataLoader, model: nn.Module, 
              epochs: int = 20, validation_loader: Optional[DataLoader] = None):
        """增强版训练"""
        model = model.to(self.device)
        
        # 优化器 + 学习率调度
        optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # 训练
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (input_ids, labels) in enumerate(train_loader):
                input_ids = input_ids.to(self.device)
                labels = torch.tensor(labels, dtype=torch.long).to(self.device)
                
                logits = model(input_ids)
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                if (batch_idx + 1) % 20 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1} - "
                              f"Loss: {loss.item():.4f}")
            
            # 统计
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            logger.info(f"✅ Epoch {epoch+1}/{epochs} - 损失: {avg_loss:.4f}, "
                       f"准确率: {accuracy:.2f}%")
            
            # 验证
            if validation_loader:
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for input_ids, labels in validation_loader:
                        input_ids = input_ids.to(self.device)
                        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
                        
                        logits = model(input_ids)
                        loss = criterion(logits, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(logits.data, 1)
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)
                
                val_accuracy = 100 * val_correct / val_total
                logger.info(f"📊 验证 - 损失: {val_loss/len(validation_loader):.4f}, "
                           f"准确率: {val_accuracy:.2f}%")
                
                # 保存最佳模型
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    torch.save(model.state_dict(), 'models/local/best_framework_detector.pt')
                    logger.info(f"💾 保存最佳模型 (准确率: {val_accuracy:.2f}%)")
            
            # 学习率调度
            scheduler.step()


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='增强版GPU框架检测模型训练')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--augmentation', type=str, default='strong', help='数据增强策略')
    parser.add_argument('--data-file', type=str, default='real_data/obfuscated_code/augmented_training_pairs.jsonl', 
                       help='训练数据文件路径')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='学习率')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 增强版GPU框架检测模型训练")
    print("使用真实NPM数据 + 更高维度 + 更多轮次")
    print("="*70)
    
    # 检查GPU
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    if device == 'cuda':
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  使用CPU(慢)")
    
    # 加载数据
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"\n❌ 数据文件不存在: {data_file}")
        print("   请先运行: python3 training/npm_code_obfuscator.py")
        return
    
    print(f"\n📂 数据文件: {data_file}")
    
    print("\n📊 加载数据集...")
    dataset = NPMFrameworkDataset(data_file, vocab_size=10000)
    
    if len(dataset) == 0:
        print("❌ 数据集为空!")
        return
    
    print(f"✅ 加载了 {len(dataset)} 条记录")
    print(f"📈 标签分布:")
    for label, count in dataset.label_counts.most_common(10):
        print(f"   {label}: {count}")
    
    # 分割数据
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    batch_size = args.batch_size if device == 'cuda' else min(args.batch_size // 4, 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"\n🎯 数据分割:")
    print(f"   训练: {train_size}")
    print(f"   验证: {val_size}")
    
    # 创建模型
    print("\n🤖 构建模型...")
    model = EnhancedTransformerModel(
        vocab_size=10000,
        hidden_size=512,      # 256 → 512
        num_layers=3,         # 2 → 3
        num_heads=8,          # 4 → 8
        max_length=512,
        num_classes=23
    )
    
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数: {params:,}")
    
    # 训练
    print(f"\n🚀 开始训练 ({args.epochs} epochs)...")
    trainer = EnhancedTrainer(device=device, batch_size=batch_size, learning_rate=args.learning_rate)
    
    try:
        trainer.train(train_loader, model, epochs=args.epochs, validation_loader=val_loader)
        
        # 保存最终模型
        output_dir = Path('models/local')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / 'framework_detector_enhanced.pt'
        torch.save(model.state_dict(), model_path)
        print(f"\n✅ 最终模型已保存: {model_path}")
        
        # 转ONNX
        print("\n📦 转换为ONNX...")
        try:
            dummy_input = torch.randint(0, 10000, (1, 512)).to(device)
            torch.onnx.export(
                model.to(device),
                (dummy_input,),
                str(output_dir / 'framework_detector_enhanced.onnx'),
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
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
