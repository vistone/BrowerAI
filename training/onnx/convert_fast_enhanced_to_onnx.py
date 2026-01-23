#!/usr/bin/env python3
"""
将fast_enhanced训练的模型转换为ONNX格式
"""

import torch
import torch.nn as nn
import torch.onnx
from pathlib import Path
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    """简单的框架检测模型 - 与fast_enhanced_trainer.py保持一致"""
    
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


def convert_to_onnx(model_path: Path, output_path: Path, device='cuda'):
    """转换模型到ONNX"""
    
    logger.info(f"🔄 开始ONNX转换")
    logger.info(f"   输入模型: {model_path}")
    logger.info(f"   输出路径: {output_path}")
    
    # 加载模型
    logger.info(f"\n📂 加载PyTorch模型...")
    model = SimpleModel(hidden_size=256, num_classes=24)
    
    if not model_path.exists():
        logger.error(f"❌ 模型文件不存在: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ 模型加载成功")
    logger.info(f"   参数数量: {params:,}")
    
    # 创建虚拟输入
    batch_size = 1
    seq_len = 512
    dummy_input = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long, device=device)
    
    logger.info(f"\n🔄 导出ONNX模型...")
    logger.info(f"   输入形状: {list(dummy_input.shape)}")
    logger.info(f"   输入名称: input_ids")
    logger.info(f"   输出名称: logits")
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 导出ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
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
        
        logger.info(f"✅ ONNX模型导出成功")
        
        # 检查文件大小
        file_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"   文件大小: {file_size:.2f} MB")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ ONNX导出失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_onnx_model(onnx_path: Path, device='cuda'):
    """验证ONNX模型"""
    
    logger.info(f"\n🔍 验证ONNX模型...")
    
    try:
        import onnx
        import onnxruntime as ort
        
        # 加载ONNX模型
        onnx_model = onnx.load(str(onnx_path))
        
        # 检查模型
        onnx.checker.check_model(onnx_model)
        logger.info(f"✅ ONNX模型结构验证通过")
        
        # 创建推理会话
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        
        logger.info(f"✅ ONNX Runtime会话创建成功")
        logger.info(f"   执行提供器: {session.get_providers()}")
        
        # 获取输入输出信息
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        logger.info(f"\n📊 模型接口信息:")
        logger.info(f"   输入: {input_info.name}, 形状: {input_info.shape}, 类型: {input_info.type}")
        logger.info(f"   输出: {output_info.name}, 形状: {output_info.shape}, 类型: {output_info.type}")
        
        # 测试推理
        import numpy as np
        test_input = np.random.randint(0, 256, (1, 512), dtype=np.int64)
        
        logger.info(f"\n🧪 执行测试推理...")
        outputs = session.run(None, {input_info.name: test_input})
        
        logger.info(f"✅ 推理成功")
        logger.info(f"   输出形状: {outputs[0].shape}")
        logger.info(f"   输出范围: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
        
        # 预测类别
        predicted_class = outputs[0].argmax(axis=1)[0]
        logger.info(f"   预测类别: {predicted_class}")
        
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️  无法验证ONNX模型: 缺少依赖 ({e})")
        logger.warning(f"   提示: pip install onnx onnxruntime-gpu")
        return False
    except Exception as e:
        logger.error(f"❌ ONNX验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='转换fast_enhanced模型到ONNX')
    parser.add_argument('--model', type=str, default='models/local/fast_enhanced_best.pt',
                       help='PyTorch模型路径')
    parser.add_argument('--output', type=str, default='models/local/fast_enhanced.onnx',
                       help='ONNX输出路径')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='计算设备')
    parser.add_argument('--verify', action='store_true', default=True,
                       help='验证ONNX模型')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 阶段5: ONNX模型导出")
    print("="*70 + "\n")
    
    model_path = Path(args.model)
    output_path = Path(args.output)
    
    # 转换
    result = convert_to_onnx(model_path, output_path, args.device)
    
    if result is None:
        logger.error("\n❌ ONNX转换失败")
        return 1
    
    # 验证
    if args.verify:
        success = verify_onnx_model(output_path, args.device)
        if not success:
            logger.warning("\n⚠️  ONNX验证未完成，但文件已生成")
    
    print("\n" + "="*70)
    print("✅ 阶段5完成!")
    print(f"   ONNX模型: {output_path}")
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
