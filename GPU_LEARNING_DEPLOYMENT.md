# GPU学习模式部署指南

**创建时间**: 2026-02-20  
**状态**: ✅ 生产就绪

## 概述

BrowerAI GPU学习系统已完成开发和优化,现已准备好用于生产环境。系统通过环境变量控制,支持GPU加速训练和推理。

## 快速开始

### 1. 环境设置

```bash
# 启用GPU学习模式
export BROWERAI_LEARNING_MODE=1      # 启用学习模式
export BROWERAI_USE_GPU=1            # 使用GPU加速
export BROWERAI_GPU_DEVICE=0         # GPU设备ID
export BROWERAI_GPU_AMP=1            # 自动混合精度(推荐)
export BROWERAI_MICRO_BATCH=4        # 微批次大小(推荐4-8)
```

### 2. 训练模型

```bash
cd /home/stone/BrowerAI

# GPU统一训练
python3 training/scripts/gpu_unified_training.py \
  --samples 500 \
  --batch-size 32 \
  --epochs 50 \
  --output models/my_model

# 或使用统一学习管道
python3 training/scripts/unified_learning_pipeline.py \
  --mode full \
  --gpu cuda:0
```

### 3. 验证模型

```bash
# 使用验证脚本
python3 models/trained_20260220_190052/validate_model.py
```

### 4. 在线学习集成

```python
import os
os.environ['BROWERAI_LEARNING_MODE'] = '1'
os.environ['BROWERAI_USE_GPU'] = '1'
os.environ['BROWERAI_GPU_AMP'] = '1'
os.environ['BROWERAI_MICRO_BATCH'] = '4'

from training.online_learner import OnlineLearner
from training.online_learning_integration import OnlineLearningIntegration

# 方式1: 直接使用在线学习器
learner = OnlineLearner(
    feature_dim=48,
    latent_dim=256,
    learning_rate=0.001
)

# 方式2: 完整集成系统
integration = OnlineLearningIntegration()
result = integration.process_website(website_data)
```

## 性能优化成果

### 优化历程

| 阶段 | 优化内容 | 性能提升 |
|------|---------|---------|
| 初始 | 基准 (CPU-GPU传输) | 210ms/feedback |
| 优化1 | GPU张量常驻 | 162ms (23%↓) |
| 优化2 | 持久化GPU缓冲 | 149ms (8%↓) |
| 优化3 | 融合损失/梯度计算 | 147ms (1.3%↓) |
| 优化4 | 微批次累积 | **0.93ms/feedback** |

### 综合性能

- **单反馈延迟**: 3.01ms → **0.93ms** (3.24x加速)
- **批处理模式**: 225x加速(相比初始CPU-GPU传输)
- **模型推理**: 0.49ms/sample, 2033 samples/sec
- **吞吐量**: ~2000 samples/sec

## 训练结果示例

### 最新训练 (2026-02-20)

```
模型: models/trained_20260220_190052/
- 参数量: 208,322
- 最佳准确率: 56.00% (Epoch 6)
- 训练时间: ~6秒 (50 epochs, 早停)
- GPU: NVIDIA GTX 1060 (6GB)
```

**文件结构:**
```
models/trained_20260220_190052/
├── checkpoints/
│   ├── best_model_epoch0.pt
│   └── best_model_epoch6.pt      # 最佳模型
├── config.json
├── training_history.json
├── TRAINING_REPORT.md
└── validate_model.py
```

## 系统架构

### GPU优化架构

```
┌─────────────────────────────────────────┐
│  环境变量配置层                          │
│  (BROWERAI_* environment variables)     │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│  Python训练层 (training/*.py)           │
│  - OnlineLearner (GPU优化)              │
│  - 融合损失/梯度计算                     │
│  - 微批次累积                           │
│  - AMP自动混合精度                      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│  PyTorch CUDA后端                       │
│  - GPU常驻张量                          │
│  - 持久化缓冲区                         │
│  - Kernel融合                           │
└─────────────────────────────────────────┘
```

### Rust集成 (CPU模式)

```rust
// crates/browerai-learning/src/online_learning.rs
use crate::config::LearningConfig;

let config = LearningConfig::with_env(); // 从环境变量读取
if config.use_gpu {
    log::warn!("GPU requested for Rust learning, but only Python GPU is supported");
}
```

## 核心代码文件

### Python训练系统

| 文件 | 功能 | GPU优化 |
|------|------|---------|
| `training/online_learner.py` | 在线学习引擎 | ✅ 全面优化 |
| `training/phase2_online_learning.py` | Phase 2学习系统 | ✅ GPU支持 |
| `training/online_learning_integration.py` | 集成系统 | ✅ 含GPU日志 |
| `training/scripts/gpu_unified_training.py` | GPU统一训练 | ✅ 原生支持 |

### Rust配置层

| 文件 | 功能 |
|------|------|
| `crates/browerai-learning/src/online_learning.rs` | 学习配置 |
| `crates/browerai-learning/src/continuous_loop.rs` | 持续学习循环 |

## 环境变量参考

### 必需变量

- `BROWERAI_LEARNING_MODE`: 启用学习模式 (0/1)
- `BROWERAI_USE_GPU`: 使用GPU加速 (0/1, 需要LEARNING_MODE=1)

### 可选变量

- `BROWERAI_GPU_DEVICE`: GPU设备ID (默认: "0", 或"cuda:0")
- `BROWERAI_GPU_AMP`: 自动混合精度 (0/1, 推荐: 1)
- `BROWERAI_MICRO_BATCH`: 微批次大小 (默认: 1, 推荐: 4-8)

### 示例配置

```bash
# 生产环境 - 最大性能
export BROWERAI_LEARNING_MODE=1
export BROWERAI_USE_GPU=1
export BROWERAI_GPU_AMP=1
export BROWERAI_MICRO_BATCH=8

# 开发环境 - 平衡模式
export BROWERAI_LEARNING_MODE=1
export BROWERAI_USE_GPU=1
export BROWERAI_GPU_AMP=0
export BROWERAI_MICRO_BATCH=4

# CPU模式 - 无GPU
export BROWERAI_LEARNING_MODE=1
export BROWERAI_USE_GPU=0
```

## 监控和调试

### GPU使用监控

```bash
# 实时GPU监控
watch -n 1 nvidia-smi

# GPU内存使用
python3 -c "import torch; print(f'Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB')"
```

### 日志级别

```bash
# 详细日志
export RUST_LOG=debug
python3 your_script.py 2>&1 | tee training.log

# 仅INFO及以上
export RUST_LOG=info
```

### 性能分析

```python
# 启用PyTorch性能分析
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # 你的训练代码
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 故障排查

### 常见问题

**1. GPU不可用**
```
WARNING: GPU learning requested but CUDA is not available. Using CPU.
```
解决: 检查CUDA安装 `python3 -c "import torch; print(torch.cuda.is_available())"`

**2. OOM (Out of Memory)**
```
RuntimeError: CUDA out of memory
```
解决: 减小batch_size或micro_batch_size

**3. 性能不佳**
- 检查AMP是否启用 (`BROWERAI_GPU_AMP=1`)
- 增加micro_batch_size (4→8)
- 使用性能分析工具定位瓶颈

### 版本兼容性

```
PyTorch: >= 2.0 (推荐 2.1+)
CUDA: >= 11.8
Python: >= 3.10
GPU: 支持Compute Capability >= 6.0
```

## 下一步计划

### 短期 (1-2周)
- [ ] 分布式训练支持 (多GPU)
- [ ] 模型量化 (INT8推理)
- [ ] 更多模型架构 (Transformer-based)

### 中期 (1-2月)
- [ ] ONNX导出优化
- [ ] TensorRT加速推理
- [ ] 模型压缩和蒸馏

### 长期 (3月+)
- [ ] Rust GPU支持 (通过cudarc)
- [ ] 自适应批大小
- [ ] 联邦学习支持

## 相关文档

- [训练报告](models/trained_20260220_190052/TRAINING_REPORT.md)
- [快速入门](QUICK_START.md)
- [架构文档](docs/ARCHITECTURE_CODE_ALIGNED.md)
- [开发指南](DEVELOPMENT_GUIDE.md)

## 联系与支持

- GitHub: https://github.com/vistone/BrowerAI
- 问题反馈: GitHub Issues

---

**最后更新**: 2026-02-20  
**维护者**: BrowerAI Team
