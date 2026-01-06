#!/usr/bin/env python3
"""
导出JS反混淆模型到ONNX格式
用于BrowerAI Rust集成
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# 添加scripts目录到path
sys.path.insert(0, str(Path(__file__).parent))

from train_js_deobfuscator import Seq2SeqModel, VOCAB_SIZE, MAX_LENGTH, PAD_ID, SOS_ID


def export_to_onnx():
    """导出模型到ONNX"""
    checkpoint_path = Path(__file__).parent.parent / 'checkpoints' / 'js_deobfuscator' / 'best_model.pt'
    output_path = Path(__file__).parent.parent.parent / 'models' / 'local' / 'js_deobfuscator_v1.onnx'
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first by running: python scripts/train_js_deobfuscator.py")
        return
    
    # 加载模型
    print(f"Loading model from {checkpoint_path}")
    model = Seq2SeqModel(vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=256, num_layers=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    
    # 创建示例输入 (batch_size=1, seq_len=60)
    dummy_input = torch.randint(0, VOCAB_SIZE, (1, MAX_LENGTH), dtype=torch.long)
    
    # 导出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # 验证
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    print(f"✅ ONNX model exported successfully: {output_path}")
    print(f"   Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"   Input shape: [batch_size, {MAX_LENGTH}] (i64)")
    print(f"   Output shape: [batch_size, {MAX_LENGTH}, {VOCAB_SIZE}] (f32 logits)")
    print()
    print("Integration path: src/ai/integration.rs::JsDeobfuscatorIntegration")
    print("Model config: Add to models/model_config.toml:")
    print(f"""
[[models]]
name = "js_deobfuscator_v1"
model_type = "JsDeobfuscator"
path = "js_deobfuscator_v1.onnx"
version = "1.0.0"
description = "JS反混淆Seq2Seq模型 (混淆JS → 清晰JS)"
""")


if __name__ == '__main__':
    export_to_onnx()
