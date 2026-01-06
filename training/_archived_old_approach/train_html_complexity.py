#!/usr/bin/env python3
"""
BrowerAI - HTML 复杂度预测模型训练脚本

从反馈数据中学习预测 HTML 文档的复杂度（0.0-1.0）
使用 PyTorch + ONNX Export

用法:
    python train_html_complexity.py --data ../data/feedback_*.json --epochs 100
"""

import json
import glob
import os
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FeedbackDataset(Dataset):
    """反馈数据集"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).reshape(-1, 1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class HtmlComplexityModel(nn.Module):
    """HTML 复杂度预测模型
    
    输入: 100 维特征向量
    输出: 复杂度评分 0.0-1.0
    """
    
    def __init__(self, input_size=100, hidden_sizes=[128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_feedback_data(pattern: str) -> List[dict]:
    """加载所有匹配的反馈文件"""
    all_data = []
    files = glob.glob(pattern)
    
    print(f"📂 找到 {len(files)} 个反馈文件")
    
    for file in sorted(files):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"   ✓ {Path(file).name}: {len(data)} 个事件")
        except Exception as e:
            print(f"   ✗ {Path(file).name}: {e}")
    
    return all_data


def extract_html_features(event: dict) -> Tuple[List[float], float]:
    """从 HTML 解析事件中提取特征
    
    特征包括:
    - 复杂度（当前值，用于训练）
    - 成功标志
    - AI 使用标志
    - 时间戳（小时、星期几等时间特征）
    - 错误信息（是否有错误）
    
    实际应用中，需要从真实 HTML 提取更多特征：
    - 标签数量、嵌套深度、文本长度
    - 表格/表单/多媒体元素数量
    - 属性数量、class/id 使用情况
    - 语义标签使用情况等
    """
    
    features = []
    
    # 基础特征
    features.append(1.0 if event.get('success', True) else 0.0)
    features.append(1.0 if event.get('ai_used', False) else 0.0)
    features.append(1.0 if event.get('error') else 0.0)
    
    # 当前复杂度（用于半监督学习）
    current_complexity = event.get('complexity', 0.5)
    features.append(current_complexity)
    
    # 时间特征（从 timestamp 提取）
    # 这里简化处理，实际可以提取更多
    features.append(0.5)  # 小时归一化
    features.append(0.5)  # 星期归一化
    
    # 填充到 100 维（实际应用中用真实特征替换）
    while len(features) < 100:
        features.append(0.0)
    
    # 标签是复杂度值
    label = current_complexity
    
    return features[:100], label


def prepare_dataset(feedback_events: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """准备训练数据集"""
    
    features_list = []
    labels_list = []
    
    for event in feedback_events:
        if event.get('type') == 'html_parsing':
            try:
                features, label = extract_html_features(event)
                features_list.append(features)
                labels_list.append(label)
            except Exception as e:
                print(f"⚠️  跳过事件: {e}")
                continue
    
    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)
    
    return features, labels


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    epochs: int,
    lr: float,
    device: str
) -> nn.Module:
    """训练模型"""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Note: PyTorch 2.9.1's ReduceLROnPlateau does not support `verbose`
    # in some builds; keep a quiet scheduler and log manually when LR steps.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"   ↘️  学习率降低: {old_lr:.6f} -> {new_lr:.6f}")
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), '../models/html_complexity_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\n⏸️  早停于 Epoch {epoch+1}，最佳验证损失: {best_val_loss:.4f}")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('../models/html_complexity_best.pth'))
    return model


def export_onnx(model: nn.Module, output_path: str):
    """导出为 ONNX 格式"""
    
    model.eval()
    dummy_input = torch.randn(1, 100)
    
    # Export to temporary location first
    import tempfile
    import shutil
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_path = os.path.join(tmpdir, "model.onnx")
        
        torch.onnx.export(
            model,
            dummy_input,
            tmp_model_path,
            input_names=['features'],
            output_names=['complexity'],
            dynamic_axes={
                'features': {0: 'batch_size'},
                'complexity': {0: 'batch_size'}
            },
            # Use latest stable opset to avoid version-conversion failures (torch 2.9 emits opset 18)
            opset_version=18,
            do_constant_folding=True,
            export_params=True,
        )
        
        # Load and re-save to embed external data
        import onnx
        onnx_model = onnx.load(tmp_model_path, load_external_data=True)
        onnx.save_model(onnx_model, output_path, save_as_external_data=False)
    
    
    print(f"✅ ONNX 模型已导出到: {output_path}")
    
    # 验证 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX 模型验证通过")
        
        # 打印模型信息
        print(f"   模型版本: {onnx_model.opset_import[0].version}")
        print(f"   输入: {onnx_model.graph.input[0].name}")
        print(f"   输出: {onnx_model.graph.output[0].name}")
    except ImportError:
        print("⚠️  未安装 onnx，跳过验证（建议: pip install onnx）")
    except Exception as e:
        print(f"⚠️  ONNX 验证失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='训练 HTML 复杂度预测模型')
    parser.add_argument('--data', type=str, default='../data/feedback_*.json',
                        help='反馈数据文件模式')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--output', type=str, default='../models/html_complexity_v1.onnx',
                        help='ONNX 输出路径')
    
    args = parser.parse_args()
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用设备: {device}")
    
    # 加载数据
    print("\n📊 加载反馈数据...")
    feedback_data = load_feedback_data(args.data)
    print(f"✅ 总共加载 {len(feedback_data)} 个反馈事件")
    
    # 准备数据集
    print("\n🔍 提取特征...")
    features, labels = prepare_dataset(feedback_data)
    print(f"✅ 特征矩阵: {features.shape}")
    print(f"   标签范围: [{labels.min():.2f}, {labels.max():.2f}]")
    
    if len(features) < 10:
        print("\n❌ 数据量太少（< 10），无法训练！")
        print("   建议: 先运行 'cargo run -- --learn' 收集更多数据")
        return
    
    # 划分训练集和验证集
    val_size = int(len(features) * args.val_split)
    train_size = len(features) - val_size
    
    indices = np.random.permutation(len(features))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = FeedbackDataset(features[train_indices], labels[train_indices])
    val_dataset = FeedbackDataset(features[val_indices], labels[val_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    print("\n🏗️  创建模型...")
    model = HtmlComplexityModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数量: {total_params:,}")
    
    # 训练
    print(f"\n🎓 开始训练（{args.epochs} 轮）...")
    model = train_model(train_loader, val_loader, model, args.epochs, args.lr, device)
    
    # 导出 ONNX
    print("\n💾 导出 ONNX 模型...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, args.output)
    
    print("\n✅ 训练完成！")
    print("\n📋 下一步:")
    print(f"  1. 复制模型: cp {args.output} ../../models/local/")
    print("  2. 更新配置: vim ../../models/model_config.toml")
    print("     添加:")
    print("     [[models]]")
    print("     name = \"html_complexity_v1\"")
    print("     model_type = \"HtmlParser\"")
    print("     path = \"html_complexity_v1.onnx\"")
    print("     version = \"1.0.0\"")
    print("     enabled = true")
    print("  3. 重新编译: cd ../.. && cargo build --release --features ai")
    print("  4. 测试效果: cargo run -- --ai-report")


if __name__ == '__main__':
    main()
