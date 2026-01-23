#!/usr/bin/env python3
"""
大规模数据训练器 - 17,542个真实NPM混淆对
优化配置用于大数据集
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class LargeScaleDataset(Dataset):
    """大规模数据集"""
    
    FRAMEWORKS = {
        'react': 0, 'vue': 1, 'angular': 2, 'svelte': 3, 'ember': 4,
        'next': 5, 'nuxt': 6, 'gatsby': 7, 'remix': 8, 'sveltekit': 9,
        'express': 10, 'fastify': 11, 'koa': 12, 'nestjs': 13, 'hapi': 14,
        'webpack': 15, 'vite': 16, 'rollup': 17, 'esbuild': 18,
        'lodash': 19, 'axios': 20, 'ramda': 21, 'underscore': 22,
    }
    
    def __init__(self, data_file: Path, vocab_size: int = 10000, max_length: int = 256):
        self.data = []
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab = self._build_vocab()
        
        logger.info("📂 加载数据集...")
        with open(data_file) as f:
            for line in f:
                try:
                    self.data.append(json.loads(line))
                except:
                    pass
        logger.info(f"✅ 加载了 {len(self.data)} 个样本")
    
    def _build_vocab(self) -> Dict[str, int]:
        """构建词汇表"""
        vocab = {}
        tokens = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>']
        for i, token in enumerate(tokens):
            vocab[token] = i
        
        keywords = [
            'function', 'const', 'let', 'var', 'return', 'if', 'else', 'for',
            'while', 'switch', 'case', 'break', 'import', 'export', 'class',
            'async', 'await', 'try', 'catch', 'new', 'this', 'super'
        ]
        for kw in keywords:
            if kw not in vocab:
                vocab[kw] = len(vocab)
        
        return vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data[idx]
        code = record.get('obfuscated', '')
        
        # 简单分词
        tokens = []
        for word in code.split()[:self.max_length]:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab['<UNK>'])
        
        # 填充
        tokens = [self.vocab['<CLS>']] + tokens
        while len(tokens) < self.max_length:
            tokens.append(self.vocab['<PAD>'])
        tokens = tokens[:self.max_length]
        
        # 推断标签
        package = record.get('package', '')
        label = 0
        for fw, idx in self.FRAMEWORKS.items():
            if fw in package.lower():
                label = idx
                break
        
        return torch.tensor(tokens, dtype=torch.long), label


class LargeScaleModel(nn.Module):
    """大规模优化模型"""
    
    def __init__(self, vocab_size: int = 10000, hidden_size: int = 512,
                 num_layers: int = 2, num_classes: int = 23):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(256, hidden_size)
        
        # 使用LSTM替代Transformer以提高速度
        self.lstm = nn.LSTM(
            hidden_size, hidden_size // 2, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 嵌入
        emb = self.embedding(input_ids)
        
        # 位置编码
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        pos_emb = self.pos_embed(positions)
        emb = emb + pos_emb
        
        # LSTM
        lstm_out, (h, c) = self.lstm(emb)
        
        # 拼接最后一层的双向隐藏状态
        h_combined = torch.cat([h[-2], h[-1]], dim=1)
        
        # 分类
        logits = self.classifier(h_combined)
        
        return logits


def train_large_scale():
    """大规模训练"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*70)
    print("🚀 大规模数据训练 - 17,542个样本")
    print("="*70 + "\n")
    
    logger.info(f"🖥️  设备: {device}")
    if device == 'cuda':
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载数据
    logger.info("\n📊 加载数据...")
    dataset = LargeScaleDataset(Path('real_data/obfuscated_code/training_pairs.jsonl'))
    
    # 分割
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    logger.info(f"   训练: {train_size}")
    logger.info(f"   验证: {val_size}")
    
    # DataLoader
    batch_size = 64 if device == 'cuda' else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    logger.info(f"   批量大小: {batch_size}")
    
    # 模型
    logger.info("\n🤖 创建模型...")
    model = LargeScaleModel().to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"   参数数: {params:,}")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    logger.info("\n🚀 开始训练...\n")
    
    best_val_acc = 0
    
    for epoch in range(30):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(logits, 1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
        
        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                
                logits = model(x)
                loss = criterion(logits, y)
                
                val_loss += loss.item()
                _, preds = torch.max(logits, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1:2d}/30 - "
                   f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/local/large_scale_best.pt')
        
        scheduler.step()
    
    # 保存最终模型
    torch.save(model.state_dict(), 'models/local/large_scale_final.pt')
    
    print("\n" + "="*70)
    print(f"✅ 训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    print("="*70 + "\n")
    
    logger.info("📁 模型已保存:")
    logger.info("   models/local/large_scale_best.pt")
    logger.info("   models/local/large_scale_final.pt")


if __name__ == '__main__':
    train_large_scale()
