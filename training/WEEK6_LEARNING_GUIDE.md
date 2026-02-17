# Week 6 加强学习系统 - 完整指南

## 🎯 概述

这是BrowerAI的加强学习系统，专为实现以下目标而设计：

- ✅ **真实数据学习**: 从GitHub和本地项目采集真实JavaScript代码
- ✅ **数据增强**: 多种代码变换技术增加样本多样性
- ✅ **12+ 混淆技术**: 扩展混淆检测能力
- ✅ **GPU加速训练**: PyTorch GPU支持
- ✅ **自动评估**: 混淆效果量化评估
- ✅ **性能指标**: 完整的学习过程监控

## 📁 系统架构

```
training/scripts/
├── real_data_learning_pipeline.py       # 真实数据采集和增强
├── advanced_obfuscation_generator.py    # 高级混淆生成器
├── gpu_unified_training.py              # GPU统一训练系统
├── unified_learning_pipeline.py         # 完整集成管道
└── WEEK6_LEARNING_GUIDE.md             # 本指南
```

### 模块说明

#### 1. RealDataCollector (真实数据采集)

采集真实JavaScript代码源：
- **GitHub框架**: React, Vue, Angular, Next.js等
- **本地项目**: 从BrowerAI项目本身采集
- **自动过滤**: 移除过短/无效代码

```python
from real_data_learning_pipeline import RealDataCollector

collector = RealDataCollector()
collector.collect_from_github([{'owner': 'facebook', 'name': 'react'}])
collector.collect_from_local_projects(['crates/browerai'])
collector.save_collected_data()
```

#### 2. DataAugmentation (数据增强)

对真实代码进行智能变换：
- 变量重命名
- 代码格式化
- 语义包装 (IIFE)
- 框架模式注入

```python
from real_data_learning_pipeline import DataAugmentation

augmenter = DataAugmentation()
variations = augmenter.augment_sample("function add(a, b) { return a + b; }", num_variations=3)
```

#### 3. AdvancedObfuscationGenerator (高级混淆生成)

12+种混淆技术：

1. **控制流混淆** - 虚假条件分支
2. **死代码注入** - 永不执行的代码块
3. **字符串编码** - Base64/十六进制编码
4. **变量重命名** - 无意义的变量名
5. **属性加密** - 计算属性访问
6. **函数包装** - IIFE和高阶函数
7. **正则表达式混淆** - 复杂化字符串匹配
8. **数组混淆** - 常量数组存储
9. **Eval混淆** - 动态代码执行
10. **注释混淆** - 迷惑性注释
11. **语义混淆** - 保持功能但改变逻辑
12. **空白混淆** - 不可见字符

```python
from advanced_obfuscation_generator import AdvancedObfuscationGenerator

generator = AdvancedObfuscationGenerator()
generator.generate_samples(num_samples=500, num_techniques=3)
generator.save_samples()
```

#### 4. ObfuscationEvaluator (混淆评估)

自动评估混淆效果：
- 代码膨胀率
- 熵增加量
- 复杂度提升

```python
from real_data_learning_pipeline import ObfuscationEvaluator

evaluator = ObfuscationEvaluator()
stats = evaluator.evaluate_obfuscation_technique('variable_rename', samples)
```

#### 5. PyTorchGPUTrainer (GPU训练)

GPU加速的深度学习模型：
- **自动混合精度** (AMP) - 加速训练
- **学习率调度** - Cosine退火
- **早停机制** - 防止过拟合
- **模型检查点** - 自动保存最佳模型

```python
from gpu_unified_training import PyTorchGPUTrainer

trainer = PyTorchGPUTrainer(device='cuda', batch_size=64, epochs=100)
trainer.build_model(input_dim=48, hidden_dims=[512, 256, 128, 64])
history = trainer.train(X_train, y_train, X_val, y_val)
```

## 🚀 快速开始

### 前置要求

```bash
# 安装必要的依赖
pip install -r training/requirements.txt

# 可选: GPU支持 (强烈推荐)
pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
```

### 完整管道运行

```bash
# 运行完整学习管道 (所有5个阶段)
python training/scripts/unified_learning_pipeline.py --mode full

# 指定GPU设备
python training/scripts/unified_learning_pipeline.py --mode full --gpu cuda:0

# 自定义样本数和轮数
python training/scripts/unified_learning_pipeline.py --mode full --samples 1000 --epochs 150
```

### 分阶段运行

```bash
# 阶段 1: 数据采集
python training/scripts/unified_learning_pipeline.py --mode collect

# 阶段 2: 数据增强
python training/scripts/unified_learning_pipeline.py --mode augment

# 阶段 3: 混淆生成
python training/scripts/unified_learning_pipeline.py --mode generate

# 阶段 4: 混淆评估
python training/scripts/unified_learning_pipeline.py --mode evaluate

# 阶段 5: GPU训练
python training/scripts/unified_learning_pipeline.py --mode train
```

### 运行特定阶段组合

```bash
# 只运行采集和训练
python training/scripts/unified_learning_pipeline.py --stages collect train

# 运行生成、评估、训练
python training/scripts/unified_learning_pipeline.py --stages generate evaluate train
```

## 📊 输出结构

```
data/week6_unified_learning/
├── raw_data/
│   └── collected_samples.jsonl          # 采集的真实代码样本
├── obfuscation_samples/
│   ├── advanced_obfuscation_samples.jsonl  # 混淆样本
│   └── summary.json                     # 混淆技术统计
├── gpu_training/
│   ├── checkpoints/
│   │   └── best_model_epochX.pt         # 保存的最佳模型
│   ├── training_history.json            # 训练历史
│   └── config.json                      # 训练配置
├── evaluation_results.json               # 混淆评估结果
└── pipeline_log.json                     # 完整管道日志
```

## 🎯 使用示例

### 示例 1: 仅采集真实数据

```bash
python training/scripts/real_data_learning_pipeline.py --collect
```

### 示例 2: 使用真实数据生成混淆样本

```bash
python training/scripts/advanced_obfuscation_generator.py \
    --samples 500 \
    --techniques 3 \
    --output data/week6_obfuscation_enhanced
```

### 示例 3: GPU训练

```bash
python training/scripts/gpu_unified_training.py \
    --samples 2000 \
    --batch-size 128 \
    --epochs 200
```

### 示例 4: 检查GPU环境

```bash
python training/scripts/gpu_unified_training.py --check-gpu
```

## 💾 数据格式

### 采集的真实代码样本

```json
{
  "source": "github",
  "repo": "facebook/react",
  "code": "function Component() { ... }",
  "file": "src/index.js",
  "timestamp": "2026-01-31T10:00:00",
  "type": "framework"
}
```

### 混淆样本

```json
{
  "id": "obf_000001",
  "technique": "variable_rename",
  "original_code": "function add(a, b) { return a + b; }",
  "obfuscated_code": "function add(_x123, _y456) { return _x123 + _y456; }",
  "source_framework": "react",
  "features": {
    "original_length": 35,
    "obfuscated_length": 45,
    "length_ratio": 1.29,
    "entropy_original": 4.2,
    "entropy_obfuscated": 4.5
  },
  "timestamp": "2026-01-31T10:30:00"
}
```

## 🔧 高级配置

### 自定义混淆技术

修改 `advanced_obfuscation_generator.py` 添加新的混淆方法：

```python
class AdvancedObfuscationGenerator:
    def custom_obfuscation(self, code: str) -> str:
        """自定义混淆技术"""
        # 实现你的混淆逻辑
        return modified_code
```

### 自定义训练参数

在 `gpu_unified_training.py` 中调整：

```python
trainer = PyTorchGPUTrainer(
    model_dim=48,
    batch_size=128,           # 增加批大小
    learning_rate=5e-4,       # 降低学习率
    epochs=200,               # 更多轮数
    use_amp=True              # 启用AMP
)
```

### 自定义数据增强

在 `real_data_learning_pipeline.py` 中添加新的增强方法：

```python
class DataAugmentation:
    def custom_augmentation(self, code: str) -> str:
        """自定义增强技术"""
        # 实现你的增强逻辑
        return augmented_code
```

## 📈 性能监控

### 训练指标

- **Loss**: 交叉熵损失
- **Accuracy**: 二分类准确率
- **Learning Rate**: 动态学习率曲线

### 混淆指标

- **Length Ratio**: 代码膨胀比 (平均 1.2-2.5x)
- **Entropy Increase**: 熵增加 (平均 0.5-1.5 bits)
- **Complexity Score**: 复杂度提升

## 🐛 故障排查

### GPU不可用

```bash
# 检查GPU
python training/scripts/gpu_unified_training.py --check-gpu

# 强制使用CPU
python training/scripts/unified_learning_pipeline.py --mode train --gpu cpu
```

### 内存不足

```bash
# 减小批大小
python training/scripts/gpu_unified_training.py --batch-size 32

# 减少模型大小
# 在 gpu_unified_training.py 中修改 hidden_dims
```

### GitHub访问限制

```bash
# 使用GitHub Token
export GITHUB_TOKEN="your_token_here"
python training/scripts/unified_learning_pipeline.py --mode collect
```

## 📚 参考资源

- PyTorch 文档: https://pytorch.org/docs/
- TensorFlow 文档: https://www.tensorflow.org/docs
- 混淆技术论文: [Code Obfuscation and Watermarking]
- JavaScript AST: https://github.com/acornjs/acorn

## 🤝 贡献指南

如何扩展这个系统：

1. **添加新的混淆技术**: 在 `AdvancedObfuscationGenerator` 中添加新方法
2. **改进数据采集**: 扩展 `RealDataCollector` 的数据源
3. **优化训练**: 调整 `PyTorchGPUTrainer` 的超参数
4. **增强评估**: 在 `ObfuscationEvaluator` 中添加新指标

## ⚡ 性能优化建议

1. **GPU内存优化**:
   - 使用自动混合精度 (AMP)
   - 调整批大小
   - 使用梯度累积

2. **训练加速**:
   - 使用多GPU (DDP)
   - 启用Pin Memory
   - 增加DataLoader Workers

3. **模型优化**:
   - 知识蒸馏
   - 量化
   - 剪枝

## 📋 检查清单

- [ ] GPU环境已配置
- [ ] PyTorch已安装
- [ ] GitHub Token已设置 (可选)
- [ ] 数据目录已创建
- [ ] 输出目录可写
- [ ] 足够的磁盘空间 (建议 > 10GB)

## 📞 支持

如有问题，请：
1. 检查日志文件
2. 运行GPU检查: `python training/scripts/gpu_unified_training.py --check-gpu`
3. 验证依赖: `pip list | grep -E "torch|tensorflow"`

## 📝 更新日志

### Week 6 更新

- ✅ 新增真实数据采集模块
- ✅ 新增数据增强管道
- ✅ 扩展混淆技术到12+种
- ✅ 实现GPU加速训练
- ✅ 添加自动评估系统
- ✅ 集成统一学习管道

---

**最后更新**: 2026-01-31
**版本**: Week 6 Enhanced Edition
