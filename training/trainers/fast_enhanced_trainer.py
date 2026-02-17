#!/usr/bin/env python3
"""快速增强训练 - 简化版，直接可用"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import logging
import argparse
import numpy as np
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastFrameworkDataset(Dataset):
    """高速JavaScript框架检测数据集"""
    
    FRAMEWORKS = {
        'react': 0, 'vue': 1, 'angular': 2, 'svelte': 3, 'ember': 4,
        'next': 5, 'nuxt': 6, 'gatsby': 7, 'remix': 8, 'sveltekit': 9,
        'express': 10, 'fastify': 11, 'koa': 12, 'nestjs': 13, 'hapi': 14,
        'webpack': 15, 'vite': 16, 'rollup': 17, 'esbuild': 18,
        'lodash': 19, 'axios': 20, 'ramda': 21, 'underscore': 22,
    }
    
    def __init__(self, data_file: Path, max_length: int = 512):
        self.data = []
        self.max_length = max_length
        self.label_counts = Counter()
        
        logger.info(f"📂 加载数据文件: {data_file}")
        
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        self.data.append(record)
                        
                        # 标签来自 package 字段
                        package = record.get('package', '').lower()
                        label = 23  # 默认
                        
                        for fw, lbl in self.FRAMEWORKS.items():
                            if fw in package:
                                label = lbl
                                break
                        
                        self.label_counts[label] += 1
                    except:
                        pass
        
        logger.info(f"✅ 加载了 {len(self.data)} 条记录")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data[idx]
        code = record.get('obfuscated', '')
        
        # 简单的字符级编码
        tokens = [ord(c) % 256 for c in code[:self.max_length]]
        tokens = tokens + [0] * (self.max_length - len(tokens))
        
        # 获取标签
        package = record.get('package', '').lower()
        label = 23
        for fw, lbl in self.FRAMEWORKS.items():
            if fw in package:
                label = lbl
                break
        
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )


class SimpleModel(nn.Module):
    """简单的框架检测模型"""
    
    def __init__(self, hidden_size=256, num_classes=24):
        super().__init__()
        self.embedding = nn.Embedding(256, 64)
        self.lstm = nn.LSTM(64, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = self.fc(x)
        return x


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        if batch_idx % 10 == 0:
            acc = 100 * total_correct / total_samples
            logger.info(f"  Batch {batch_idx}: Loss={loss.item():.4f}, Acc={acc:.2f}%")
    
    avg_loss = total_loss / len(loader)
    avg_acc = 100 * total_correct / total_samples
    return avg_loss, avg_acc


def eval_epoch(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    avg_acc = 100 * total_correct / total_samples
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data-file', type=str, default='real_data/obfuscated_code/augmented_training_pairs.jsonl')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 快速增强GPU训练 - 实际计算版")
    print("="*70 + "\n")
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"✅ 设备: {device}")
    if device == 'cuda':
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载数据
    data_file = Path(args.data_file)
    if not data_file.exists():
        logger.error(f"❌ 数据文件不存在: {data_file}")
        return
    
    dataset = FastFrameworkDataset(data_file)
    
    if len(dataset) == 0:
        logger.error("❌ 数据集为空")
        return
    
    # 分割
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    
    logger.info(f"\n📊 数据统计:")
    logger.info(f"   总样本: {len(dataset)}")
    logger.info(f"   训练: {train_size}")
    logger.info(f"   验证: {val_size}")
    logger.info(f"   标签分布: {dict(dataset.label_counts.most_common(5))}")
    
    # 模型
    logger.info(f"\n🤖 构建模型...")
    model = SimpleModel(hidden_size=256, num_classes=24).to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"   参数: {params:,}")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # 训练
    logger.info(f"\n🚀 开始训练 ({args.epochs} epochs)...\n")
    
    best_val_acc = 0.0
    best_model_path = Path('models/local/fast_enhanced_best.pt')
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        
        logger.info(f"\n✅ Epoch {epoch+1}/{args.epochs}")
        logger.info(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        logger.info(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"   💾 最佳模型已保存")
    
    logger.info(f"\n✅ 训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    
    # 保存最终模型
    final_path = Path('models/local/fast_enhanced_final.pt')
    torch.save(model.state_dict(), final_path)
    logger.info(f"✅ 最终模型已保存: {final_path}")


if __name__ == '__main__':
    main()
