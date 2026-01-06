#!/usr/bin/env python3
"""
深度学习模型训练 - HTML/CSS/JS理解模型
使用PyTorch训练轻量级神经网络，替代简单分类器
"""

import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CodeUnderstandingDataset(Dataset):
    """代码理解数据集"""
    
    def __init__(self, features_file):
        self.samples = []
        
        with open(features_file) as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
        
        print(f"✅ 加载 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 构造特征向量
        features = []
        
        # HTML特征 (10维)
        html_features = [
            sample.get('dom_depth', 0) / 20.0,  # 归一化
            sample.get('element_count', 0) / 100.0,
            sample.get('text_length', 0) / 1000.0,
            sample.get('link_count', 0) / 50.0,
            sample.get('script_count', 0) / 10.0,
            sample.get('style_count', 0) / 10.0,
            sample.get('form_count', 0) / 5.0,
            sample.get('table_count', 0) / 5.0,
            sample.get('img_count', 0) / 20.0,
            sample.get('video_count', 0) / 5.0,
        ]
        features.extend(html_features)
        
        # CSS特征 (8维)
        css_features = [
            sample.get('css_rule_count', 0) / 100.0,
            sample.get('selector_avg_complexity', 0) / 10.0,
            sample.get('property_count', 0) / 200.0,
            sample.get('unique_colors', 0) / 50.0,
            sample.get('media_query_count', 0) / 10.0,
            sample.get('animation_count', 0) / 5.0,
            sample.get('font_family_count', 0) / 10.0,
            sample.get('class_count', 0) / 100.0,
        ]
        features.extend(css_features)
        
        # JS特征 (12维)
        js_features = [
            sample.get('js_statement_count', 0) / 100.0,
            sample.get('function_count', 0) / 20.0,
            sample.get('variable_count', 0) / 50.0,
            sample.get('async_count', 0) / 10.0,
            sample.get('event_listener_count', 0) / 20.0,
            sample.get('class_definition_count', 0) / 5.0,
            sample.get('import_count', 0) / 10.0,
            sample.get('export_count', 0) / 10.0,
            sample.get('avg_line_length', 0) / 100.0,
            sample.get('max_nesting_depth', 0) / 10.0,
            sample.get('complexity_score', 0),  # 已归一化
            sample.get('minified_probability', 0),  # 已归一化
        ]
        features.extend(js_features)
        
        # URL特征 (5维)
        url_features = [
            sample.get('url_length', 0) / 100.0,
            sample.get('subdomain_count', 0) / 5.0,
            sample.get('path_depth', 0) / 10.0,
            sample.get('query_param_count', 0) / 10.0,
            1.0 if sample.get('is_https', False) else 0.0,
        ]
        features.extend(url_features)
        
        # 确保特征长度固定
        while len(features) < 35:
            features.append(0.0)
        features = features[:35]
        
        # 标签：站点类别 (one-hot编码)
        category = sample.get('inferred_category', 'other')
        category_map = {'news': 0, 'ecommerce': 1, 'tech': 2, 'social': 3, 'video': 4, 
                       'education': 5, 'government': 6, 'finance': 7, 'entertainment': 8, 'other': 9}
        label = category_map.get(category, 9)
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class DeepCodeUnderstanding(nn.Module):
    """深度代码理解模型 - 多层神经网络"""
    
    def __init__(self, input_dim=35, hidden_dims=[128, 64, 32], num_classes=10):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建多层感知机
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_embedding(self, x):
        """获取中间层嵌入向量"""
        # 返回倒数第二层的激活值
        for layer in self.network[:-1]:
            x = layer(x)
        return x


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
    
    print(f"\n✅ 训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    return model


def export_to_onnx(model, input_dim=35, output_path='models/code_understanding.onnx'):
    """导出为ONNX格式"""
    model.eval()
    dummy_input = torch.randn(1, input_dim)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['features'],
        output_names=['logits'],
        dynamic_axes={'features': {0: 'batch_size'},
                     'logits': {0: 'batch_size'}}
    )
    
    print(f"💾 ONNX模型已保存: {output_path}")
    
    # 验证ONNX模型
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX模型验证通过")
    
    # 显示模型大小
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"📦 模型大小: {size_mb:.2f} MB")


def main():
    features_file = Path(__file__).parent.parent / 'features' / 'extracted_features.jsonl'
    
    if not features_file.exists():
        print(f"❌ 特征文件不存在: {features_file}")
        print("请先运行: python scripts/extract_features.py")
        return 1
    
    print("=" * 60)
    print("🚀 深度学习模型训练")
    print("=" * 60)
    
    # 1. 加载数据
    dataset = CodeUnderstandingDataset(features_file)
    
    # 2. 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"📊 训练集: {train_size} 样本, 验证集: {val_size} 样本")
    
    # 3. 创建模型
    model = DeepCodeUnderstanding(
        input_dim=35,
        hidden_dims=[128, 64, 32],
        num_classes=10
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数: {total_params:,}")
    
    # 4. 训练模型
    model = train_model(model, train_loader, val_loader, epochs=50)
    
    # 5. 导出ONNX
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    onnx_path = models_dir / 'code_understanding_v1.onnx'
    export_to_onnx(model, input_dim=35, output_path=str(onnx_path))
    
    print("\n🎉 深度学习模型训练完成！")
    print(f"📍 模型位置: {onnx_path}")
    print("\n下一步:")
    print("1. 复制到Rust项目: cp training/models/*.onnx models/local/")
    print("2. 更新model_config.toml配置")
    print("3. 在Rust中加载模型进行推理")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
