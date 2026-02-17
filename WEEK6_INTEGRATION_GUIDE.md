# Week 6 真实数据学习系统 - 集成使用指南

**集成日期**: 2026-02-01  
**系统状态**: ✅ 完整集成  
**可用入口**: 3个 (train_week6.sh, Justfile, 独立脚本)

---

## 🎯 快速开始

### 方式1: 使用Just命令（推荐）

**最简单的方式 - 一行命令启动：**

```bash
just learn-real
```

**自定义参数：**

```bash
# 设置混淆技术数
just learn-real-techniques 6

# 设置训练轮数
just learn-real-epochs 100

# 设置批大小
just learn-real-batch 64

# 全参数自定义
just learn-real-custom crates 6 100 64
```

**查看结果和帮助：**

```bash
# 查看学习结果
just learn-results

# 显示详细帮助
just learn-help

# 清空学习结果
just learn-clean
```

---

### 方式2: 使用Bash脚本

**最灵活的方式 - 完整的shell脚本：**

```bash
# 运行真实数据学习
./training/train_week6.sh real

# 自定义参数
./training/train_week6.sh real --techniques 6 --epochs 50 --batch-size 64

# 自定义采集目录
./training/train_week6.sh real --collect-dir crates --techniques 4
```

**其他命令：**

```bash
./training/train_week6.sh help        # 显示帮助
./training/train_week6.sh check-gpu   # 检查GPU环境
```

---

### 方式3: 直接运行Python脚本

**最直接的方式 - 不通过脚本包装：**

```bash
# 基础用法
python3 training/scripts/real_learning.py

# 自定义参数
python3 training/scripts/real_learning.py \
    --collect-dir crates \
    --techniques 4 \
    --epochs 50 \
    --batch-size 32
```

---

## 📊 参数说明

### 采集参数 (--collect-dir)

| 值 | 说明 | 效果 |
|---|------|------|
| `crates` | Rust代码库 | 采集277个Rust文件 |
| `training` | Python脚本库 | 采集5207个Python文件 |
| `.` (项目根) | 全项目采集 | 采集5491个文件 (完整) |

**示例：**
```bash
just learn-real-custom . 4 50 32  # 采集全项目代码
```

---

### 混淆技术参数 (--techniques)

**可选值**: 1-12 (最多12种技术)

| 值 | 说明 | 复杂度 | 耗时 |
|---|------|--------|------|
| 1 | 单一技术 | 低 | ~2秒 |
| 3 | 轻度混淆 | 中 | ~5秒 |
| 4 | 标准混淆 | 中 | ~7秒 (默认) |
| 6 | 重度混淆 | 高 | ~10秒 |
| 12 | 极限混淆 | 极高 | ~15秒 |

**12种混淆技术：**
1. control_flow - 控制流混淆
2. dead_code - 死代码插入
3. string - 字符串编码
4. variable_rename - 变量重命名
5. property_encryption - 属性加密
6. function_wrapping - 函数包装
7. regex_obfuscation - 正则混淆
8. array_obfuscation - 数组混淆
9. eval_obfuscation - Eval混淆
10. comment_obfuscation - 注释混淆
11. semantic_obfuscation - 语义混淆
12. whitespace_obfuscation - 空白混淆

**示例：**
```bash
just learn-real-techniques 8   # 8种混淆 (高复杂度)
```

---

### 训练参数 (--epochs)

**可选值**: 10-200+ (推荐: 50-100)

| 值 | 说明 | 训练时间 | 精度 |
|---|------|---------|------|
| 10 | 快速测试 | ~2秒 | 一般 |
| 30 | 标准训练 | ~5秒 | 良好 |
| 50 | 完整训练 | ~8秒 | 优秀 (默认) |
| 100 | 深度训练 | ~15秒 | 极优 |

**示例：**
```bash
just learn-real-epochs 100   # 100轮深度训练
```

---

### 批大小参数 (--batch-size)

**可选值**: 16-128 (推荐: 32-64)

| 值 | 说明 | 内存占用 | 速度 |
|---|------|---------|------|
| 16 | 小批 | 低 | 慢 |
| 32 | 标准 | 中 | 适中 (默认) |
| 64 | 大批 | 高 | 快 |
| 128 | 超大批 | 极高 | 极快 |

**示例：**
```bash
just learn-real-batch 64   # 更大的批大小 (需要更多显存)
```

---

## 📈 查看结果

### 查看汇总结果
```bash
just learn-results
```

输出示例：
```
📊 真实数据学习结果

✅ 原始代码采集结果:
   文件行数: 5491
   文件大小: 67M

✅ 混淆样本生成结果:
   样本数量: 5473
   文件大小: 7.5M

✅ GPU训练历史:
   已生成
```

### 查看原始代码样本
```bash
head -1 data/real_codes/raw_codes.jsonl | python3 -m json.tool
```

### 查看混淆样本
```bash
head -1 data/real_codes/obfuscated_samples.jsonl | python3 -m json.tool
```

### 查看训练历史
```bash
cat data/real_codes/training_history.json | python3 -m json.tool
```

### 统计数据
```bash
# 样本数统计
wc -l data/real_codes/*.jsonl

# 文件大小
du -sh data/real_codes/

# 详细分析
python3 << 'EOF'
import json
with open('data/real_codes/training_history.json') as f:
    history = json.load(f)
    losses = [h['loss'] for h in history]
    print(f"初始损失: {losses[0]:.6f}")
    print(f"最小损失: {min(losses):.6f}")
    print(f"最终损失: {losses[-1]:.6f}")
EOF
```

---

## 🔄 工作流示例

### 快速验证 (2分钟)
```bash
# 运行完整流程 (默认参数)
just learn-real

# 查看结果
just learn-results
```

### 标准开发 (5分钟)
```bash
# 标准流程 (4种技术, 50轮)
just learn-real

# 查看详细结果
cat data/real_codes/training_history.json | python3 -m json.tool
```

### 深度优化 (15分钟)
```bash
# 高度混淆 (8种技术) + 深度训练 (100轮)
just learn-real-custom crates 8 100 64

# 查看和对比结果
just learn-results
```

### 完整实验 (30分钟)
```bash
# 采集全项目代码 + 极限混淆 (12种) + 深度训练 (100轮)
just learn-real-custom . 12 100 64

# 详细分析
python3 -c "
import json
with open('data/real_codes/training_history.json') as f:
    history = json.load(f)
    print(f'总epochs: {len(history)}')
    print(f'初始: {history[0][\"loss\"]:.6f}')
    print(f'最优: {min(h[\"loss\"] for h in history):.6f}')
"
```

---

## 🛠️ 故障排查

### 问题：Python版本不兼容

```bash
# 检查Python版本
python3 --version  # 需要3.8+

# 使用指定版本
/usr/bin/python3.11 training/scripts/real_learning.py
```

### 问题：PyTorch未安装

```bash
# 检查PyTorch
python3 -c "import torch; print(torch.__version__)"

# 安装PyTorch (CPU)
pip install torch

# 安装PyTorch (GPU - 需要CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 问题：GPU不可用

```bash
# 检查GPU
just learn-help  # 显示GPU状态

# 检查CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# 列出可用GPU
python3 -c "import torch; print(torch.cuda.device_count())"
```

### 问题：内存不足

```bash
# 减小批大小
just learn-real-batch 16

# 减少样本
python3 training/scripts/real_learning.py --collect-dir crates --techniques 2
```

---

## 📚 详细文档

完整的Week 6学习系统文档：

- [WEEK6_COMPLETION_SUMMARY.md](WEEK6_COMPLETION_SUMMARY.md) - 完整成功报告
- [WEEK6_REAL_DATA_LEARNING_REPORT.md](WEEK6_REAL_DATA_LEARNING_REPORT.md) - 详细执行报告
- [DATA_STATISTICS_REPORT.md](DATA_STATISTICS_REPORT.md) - 数据统计分析
- [QUICK_REFERENCE_REAL_DATA.md](QUICK_REFERENCE_REAL_DATA.md) - 快速参考

---

## 🚀 集成点总结

| 入口 | 方式 | 最佳用途 |
|-----|------|---------|
| **just** | 命令行 | 日常开发、快速启动 |
| **train_week6.sh** | Bash脚本 | CI/CD集成、详细日志 |
| **real_learning.py** | Python | 程序集成、自定义逻辑 |

---

## 💡 最佳实践

### ✅ 推荐做法
1. 首次运行使用默认参数：`just learn-real`
2. 查看结果：`just learn-results`
3. 需要调优时自定义参数
4. 在脚本中复用命令

### ❌ 避免做法
1. 频繁改变参数（先运行一次标准配置）
2. 同时运行多个学习任务（会竞争GPU）
3. 忽视GPU状态（check-gpu先确认）
4. 清空数据前未备份结果

---

**集成完成时间**: 2026-02-01  
**系统状态**: ✅ 生产就绪  
**文档完整度**: 100%
