#!/usr/bin/env python3
"""
ONNX 模型验证和测试脚本

用法:
    python validate_model.py ../models/html_complexity_v1.onnx
"""

import sys
import time
import argparse
import numpy as np

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("❌ 缺少依赖！请安装:")
    print("   pip install onnx onnxruntime")
    sys.exit(1)


def validate_onnx_model(model_path: str):
    """验证 ONNX 模型格式"""
    print(f"📋 验证模型: {model_path}")
    
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print("✅ ONNX 格式验证通过")
        
        # 打印模型信息
        print(f"\n📊 模型信息:")
        print(f"   Opset 版本: {model.opset_import[0].version}")
        print(f"   IR 版本: {model.ir_version}")
        
        # 输入信息
        print(f"\n📥 模型输入:")
        for input_tensor in model.graph.input:
            dims = [d.dim_value if d.dim_value > 0 else 'dynamic' 
                   for d in input_tensor.type.tensor_type.shape.dim]
            print(f"   - {input_tensor.name}: {dims}")
        
        # 输出信息
        print(f"\n📤 模型输出:")
        for output_tensor in model.graph.output:
            dims = [d.dim_value if d.dim_value > 0 else 'dynamic' 
                   for d in output_tensor.type.tensor_type.shape.dim]
            print(f"   - {output_tensor.name}: {dims}")
        
        return True
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False


def benchmark_model(model_path: str, num_runs: int = 1000):
    """性能测试"""
    print(f"\n⚡ 性能测试（{num_runs} 次推理）...")
    
    try:
        # 创建推理会话
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # 获取输入信息
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # 处理动态维度
        batch_size = 1
        feature_dim = input_shape[1] if len(input_shape) > 1 else input_shape[0]
        
        # 生成随机输入
        input_data = np.random.randn(batch_size, feature_dim).astype(np.float32)
        
        # 预热
        for _ in range(10):
            session.run(None, {input_name: input_data})
        
        # 基准测试
        start = time.time()
        for _ in range(num_runs):
            outputs = session.run(None, {input_name: input_data})
        end = time.time()
        
        avg_time_ms = (end - start) / num_runs * 1000
        
        print(f"✅ 平均推理时间: {avg_time_ms:.3f} ms")
        print(f"✅ 吞吐量: {1000/avg_time_ms:.1f} 次/秒")
        
        # 测试批量推理
        batch_sizes = [1, 10, 100]
        print(f"\n📊 批量推理测试:")
        for bs in batch_sizes:
            batch_input = np.random.randn(bs, feature_dim).astype(np.float32)
            start = time.time()
            for _ in range(100):
                session.run(None, {input_name: batch_input})
            elapsed = time.time() - start
            per_sample = elapsed / 100 / bs * 1000
            print(f"   Batch {bs}: {per_sample:.3f} ms/样本")
        
        return True
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False


def test_inference(model_path: str):
    """测试推理功能"""
    print(f"\n🧪 推理功能测试...")
    
    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # 测试不同输入
        test_cases = [
            ("全零输入", np.zeros((1, session.get_inputs()[0].shape[1]), dtype=np.float32)),
            ("全一输入", np.ones((1, session.get_inputs()[0].shape[1]), dtype=np.float32)),
            ("随机输入", np.random.randn(1, session.get_inputs()[0].shape[1]).astype(np.float32)),
        ]
        
        for name, input_data in test_cases:
            outputs = session.run([output_name], {input_name: input_data})
            result = outputs[0][0]
            
            if len(result.shape) == 0:  # 标量
                print(f"   {name}: {result:.4f}")
            else:  # 向量
                print(f"   {name}: {result}")
        
        print("✅ 推理功能正常")
        return True
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='验证 ONNX 模型')
    parser.add_argument('model', type=str, help='ONNX 模型路径')
    parser.add_argument('--benchmark', action='store_true', help='运行性能测试')
    parser.add_argument('--runs', type=int, default=1000, help='基准测试次数')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BrowerAI ONNX 模型验证工具")
    print("=" * 60)
    
    # 验证模型
    if not validate_onnx_model(args.model):
        sys.exit(1)
    
    # 测试推理
    if not test_inference(args.model):
        sys.exit(1)
    
    # 性能测试
    if args.benchmark:
        if not benchmark_model(args.model, args.runs):
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    main()
