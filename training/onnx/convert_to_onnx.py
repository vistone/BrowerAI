#!/usr/bin/env python3
"""
将训练好的PyTorch模型转换为ONNX格式，用于Rust集成
"""

import torch
import torch.nn as nn
import torch.onnx
from pathlib import Path
import json
import sys

# 模型架构定义 (与large_scale_trainer.py保持一致)
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


def convert_model_to_onnx(pt_model_path, onnx_output_path, device='cuda'):
    """转换PyTorch模型到ONNX"""
    print(f"📦 加载PyTorch模型: {pt_model_path}")
    
    # 加载模型
    model = LargeScaleModel(vocab_size=10000, hidden_size=512, num_layers=2, num_classes=23)
    checkpoint = torch.load(pt_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print(f"✅ 模型加载完成")
    print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建虚拟输入
    batch_size = 1
    seq_len = 256
    dummy_input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    
    print(f"\n🔄 转换为ONNX...")
    print(f"   输入形状: input_ids={list(dummy_input_ids.shape)}")
    
    # 导出ONNX
    onnx_output_path = Path(onnx_output_path)
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input_ids,
        str(onnx_output_path),
        input_names=['input_ids'],
        output_names=['logits'],
        opset_version=14,
        do_constant_folding=True,
        verbose=False,
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX模型已保存: {onnx_output_path}")
    
    # 验证ONNX模型
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_output_path))
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX模型验证通过")
    except Exception as e:
        print(f"⚠️ ONNX验证失败: {e}")
    
    return onnx_output_path


def test_onnx_inference(onnx_path, test_input_ids=None):
    """测试ONNX模型推理"""
    try:
        import onnxruntime
    except ImportError:
        print("❌ 需要安装onnxruntime: pip install onnxruntime")
        return
    
    print(f"\n🧪 测试ONNX推理...")
    
    # 创建推理会话
    sess = onnxruntime.InferenceSession(
        str(onnx_path),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    # 创建测试输入
    if test_input_ids is None:
        import numpy as np
        batch_size = 2
        seq_len = 256
        test_input_ids = np.random.randint(0, 10000, (batch_size, seq_len), dtype=np.int64)
        test_lengths = np.full((batch_size,), seq_len, dtype=np.int64)
    else:
        import numpy as np
        test_lengths = np.array([len(ids) for ids in test_input_ids], dtype=np.int64)
        max_len = max(test_lengths)
        padded = np.zeros((len(test_input_ids), max_len), dtype=np.int64)
        for i, ids in enumerate(test_input_ids):
            padded[i, :len(ids)] = ids
        test_input_ids = padded
    
    # 推理
    try:
        inputs = {
            'input_ids': test_input_ids.astype(np.int64),
            'lengths': test_lengths.astype(np.int64)
        }
        outputs = sess.run(None, inputs)
        logits = outputs[0]
        
        print(f"✅ 推理成功!")
        print(f"   输入形状: {test_input_ids.shape}")
        print(f"   输出形状: {logits.shape}")
        print(f"   输出范围: [{logits.min():.3f}, {logits.max():.3f}]")
        
        return logits
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return None


def main():
    # 路径配置
    pt_model_path = "/home/stone/BrowerAI/models/local/large_scale_best.pt"
    onnx_output_path = "/home/stone/BrowerAI/models/local/large_scale_model.onnx"
    
    # 检查PyTorch模型是否存在
    if not Path(pt_model_path).exists():
        print(f"❌ 错误: 模型文件不存在 {pt_model_path}")
        sys.exit(1)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用设备: {device}")
    
    # 转换模型
    onnx_path = convert_model_to_onnx(pt_model_path, onnx_output_path, device)
    
    # 测试推理
    test_onnx_inference(onnx_path)
    
    # 生成元数据
    metadata = {
        "model_name": "large_scale_framework_detector",
        "model_type": "LSTM",
        "version": "1.0.0",
        "input_shapes": {
            "input_ids": [1, 256],  # [batch_size, seq_len]
            "lengths": [1]
        },
        "output_shapes": {
            "logits": [1, 23]  # [batch_size, num_classes]
        },
        "vocab_size": 10000,
        "num_classes": 23,
        "embedding_dim": 512,
        "hidden_dim": 512,
        "num_layers": 2,
        "class_names": [
            "react", "vue", "angular", "svelte", "preact",
            "solid", "alpine", "htmx", "web-components", "lit",
            "ember", "backbone", "marko", "riotjs", "mithril",
            "polymer", "riot", "dojo", "knockout", "inferno",
            "unknown", "vanilla", "jquery"
        ],
        "training_samples": 17542,
        "validation_accuracy": 0.9578,
        "training_accuracy": 0.9838
    }
    
    metadata_path = Path(onnx_output_path).with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📋 元数据已保存: {metadata_path}")
    print(f"\n✨ 转换完成!")
    print(f"   ONNX模型: {onnx_output_path}")
    print(f"   元数据: {metadata_path}")


if __name__ == "__main__":
    main()
