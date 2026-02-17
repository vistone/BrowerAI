#!/usr/bin/env python3
"""
任务4: 模型量化优化 - 减小模型体积
将float32模型量化为int8，减小体积约75%
"""

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import os

def quantize_model(input_model_path, output_model_path):
    """动态量化ONNX模型"""
    
    print("\n" + "="*70)
    print("🔧 任务4: 模型量化优化")
    print("="*70 + "\n")
    
    input_path = Path(input_model_path)
    output_path = Path(output_model_path)
    
    if not input_path.exists():
        print(f"❌ 输入模型不存在: {input_path}")
        return None
    
    # 获取原始模型信息
    original_size = input_path.stat().st_size / (1024 * 1024)
    print(f"📊 原始模型信息:")
    print(f"   路径: {input_path}")
    print(f"   大小: {original_size:.2f} MB")
    
    # 执行量化
    print(f"\n🔄 执行动态量化...")
    print(f"   量化类型: QInt8 (Float32 → Int8)")
    print(f"   量化方法: Dynamic (动态量化)")
    
    try:
        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=QuantType.QInt8,
        )
        
        print(f"✅ 量化完成")
        
        # 检查量化后的模型
        quantized_size = output_path.stat().st_size / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        
        print(f"\n📊 量化后模型信息:")
        print(f"   路径: {output_path}")
        print(f"   大小: {quantized_size:.2f} MB")
        print(f"   压缩率: {reduction:.1f}%")
        print(f"   减少: {original_size - quantized_size:.2f} MB")
        
        # 验证模型
        print(f"\n🔍 验证量化模型...")
        try:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print(f"✅ 量化模型验证通过")
        except Exception as e:
            print(f"❌ 模型验证失败: {e}")
            return None
        
        print("\n" + "="*70)
        print(f"✅ 量化优化完成!")
        print("="*70)
        print(f"   原始: {original_size:.2f} MB → 量化: {quantized_size:.2f} MB")
        print(f"   节省: {reduction:.1f}% ({original_size - quantized_size:.2f} MB)")
        print("="*70 + "\n")
        
        return output_path
        
    except Exception as e:
        print(f"❌ 量化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_performance(original_path, quantized_path):
    """比较原始模型和量化模型的性能"""
    
    import onnxruntime as ort
    import numpy as np
    import time
    
    print("\n" + "="*70)
    print("⚡ 性能对比测试")
    print("="*70 + "\n")
    
    # 测试输入
    test_input = np.random.randint(0, 256, (1, 512), dtype=np.int64)
    
    # 测试原始模型
    print("📊 原始模型 (Float32):")
    session_orig = ort.InferenceSession(str(original_path), providers=['CPUExecutionProvider'])
    
    times_orig = []
    for _ in range(50):
        start = time.perf_counter()
        session_orig.run(None, {'input_ids': test_input})
        times_orig.append((time.perf_counter() - start) * 1000)
    
    avg_orig = np.mean(times_orig)
    print(f"   平均推理时间: {avg_orig:.2f} ms")
    
    # 测试量化模型
    print(f"\n📊 量化模型 (Int8):")
    session_quant = ort.InferenceSession(str(quantized_path), providers=['CPUExecutionProvider'])
    
    times_quant = []
    for _ in range(50):
        start = time.perf_counter()
        session_quant.run(None, {'input_ids': test_input})
        times_quant.append((time.perf_counter() - start) * 1000)
    
    avg_quant = np.mean(times_quant)
    speedup = avg_orig / avg_quant
    
    print(f"   平均推理时间: {avg_quant:.2f} ms")
    print(f"\n⚡ 加速比: {speedup:.2f}x")
    print(f"   {'✅ 更快' if speedup > 1 else '⚠️  稍慢' if speedup > 0.9 else '❌ 较慢'}")
    
    print("="*70 + "\n")


def main():
    # 模型路径
    original_model = Path("models/local/fast_enhanced.onnx")
    quantized_model = Path("models/local/fast_enhanced_quantized.onnx")
    
    # 执行量化
    result = quantize_model(original_model, quantized_model)
    
    if result:
        # 性能对比
        compare_performance(original_model, quantized_model)
        
        print("💡 使用建议:")
        print("   - 量化模型适用于CPU推理")
        print("   - 体积减小75%，内存占用更低")
        print("   - 推理速度略有提升或持平")
        print("   - 准确率损失通常 < 1%")
        print()


if __name__ == '__main__':
    main()
