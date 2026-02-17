# 🌐 BrowerAI Training - 真实数据驱动架构

**核心原则**：完全基于真实网站数据，无任何虚拟/模拟数据生成

AI 网站解析和优化训练系统：
- 输入：真实网站代码（来自爬虫）
- 处理：框架检测、代码分析、混淆检测、反混淆
- 输出：代码优化、特征提取、性能分析

## 🎯 设计哲学

### 真实数据驱动
- ✅ **只使用真实网站数据**（来自爬虫）
- ✅ **完整的数据处理流程**（爬取→解析→训练→评估）
- ✅ **生产级模型**（在真实网站验证过）
- ❌ **不生成虚拟数据**（删除所有模拟数据生成器）

### 整体网站学习
不学习孤立的技术点（JS/HTML/CSS分开），而是将完整网站（HTML+CSS+JS）作为一个整体来学习。

> "学习应该是整个网站的思想去学习，而不是单独的某个技术层面学习"

### 完整的处理流程
- 数据获取：爬虫系统获取真实网站
- 特征提取：检测框架、识别混淆、分析结构
- 模型训练：使用真实数据训练模型
- 性能评估：在真实网站样本上验证

## 📁 目录结构

## 📁 目录结构（仅真实数据模块）

```
training/
├── README.md                    # 本文件
│
├── real_data/                   # ⭐ 60M 真实网站数据库
│   ├── annotated/               真实网站+人工标注
│   ├── expanded/                扩展的真实样本
│   ├── final/                   最终处理后的数据
│   ├── scaleable/               可扩展真实数据
│   └── websites/                原始网站爬取数据
│
├── crawlers/                    # 数据获取层
│   ├── real_website_crawler.py
│   ├── scaleable_website_crawler.py
│   ├── github_framework_crawler.py
│   └── npm_package_crawler.py
│
├── detectors/                   # 框架检测层
│   ├── high_precision_detector.py
│   ├── hybrid_framework_detector.py
│   ├── production_hybrid_detector.py
│   └── gpu_framework_detector.py
│
├── trainers/                    # 模型训练层
│   ├── real_data_trainer.py
│   ├── production_trainer.py
│   ├── fast_enhanced_trainer.py
│   └── enhanced_gpu_trainer.py
│
├── pipelines/                   # 端到端处理流程
│   ├── complete_system.py
│   ├── implementation_pipeline.py
│   ├── multimodel_learning_system.py
│   └── final_production_system.py
│
├── obfuscation/                 # 混淆/反混淆系统
│   ├── global_js_obfuscation_deobfuscation_system.py
│   ├── enhanced_deobfuscation_rules.py
│   └── end_to_end_deobfuscation_demo.py
│
├── models/                      # 模型管理
│   ├── train_all_models.py
│   └── export_to_onnx.py
│
├── evaluation/                  # 模型评估
├── optimization/                # 模型优化
├── onnx/                       # ONNX 转换
├── metrics/                    # 监控指标
├── services/                   # API 服务
├── utils/                      # 工具函数
└── scripts/                    # 脚本工具
    ├── data_tools/             数据处理工具
    ├── export/                 模型导出工具
    └── legacy/                 遗留脚本（已弃用）
```

## 🚀 快速开始

### 1. 使用真实网站爬虫
```python
from training.crawlers.real_website_crawler import RealWebsiteCrawler

crawler = RealWebsiteCrawler()
websites = crawler.crawl(['https://example.com'])
# 直接保存到 training/real_data/
```

### 2. 加载真实数据训练
```python
from training.trainers.production_trainer import ProductionTrainer

# 加载真实数据
dataset = load_real_dataset('training/real_data/')

# 训练模型
trainer = ProductionTrainer(use_gpu=True)
model = trainer.train(dataset, epochs=50)
```

### 3. 框架检测
```python
from training.detectors.high_precision_detector import HighPrecisionDetector

detector = HighPrecisionDetector()
framework = detector.detect(code_sample)
print(f"检测框架: {framework['name']} (置信度: {framework['confidence']})")
```

### 4. 实战反混淆
```python
from training.obfuscation.end_to_end_deobfuscation_demo import PracticalDeobfuscator

deobfuscator = PracticalDeobfuscator()
original = deobfuscator.deobfuscate(obfuscated_code)
```

## 📊 真实数据集规模

| 数据源 | 大小 | 样本数 | 状态 |
|--------|------|--------|------|
| real_data/ | 60M | 10,000+ | ✅ 就绪 |
| GitHub 项目 | 扩展 | 可扩展 | 通过爬虫 |
| NPM 包 | 扩展 | 可扩展 | 通过爬虫 |

## ✨ 系统特点

- 🌍 **完全真实数据**：无虚拟数据生成
- 🔄 **完整工作流**：爬虫→训练→评估→部署
- 📈 **生产级质量**：已在真实网站验证
- 🚀 **可扩展设计**：轻松添加新数据源
- 🔬 **实战反混淆**：基于真实混淆代码

## 🛠️ 开发工作流

### 新增数据源
1. 在 `training/crawlers/` 创建爬虫
2. 爬取数据保存到 `training/real_data/`
3. 集成到训练脚本

### 改进检测器
1. 用真实数据在 `trainers/` 中训练
2. 在真实网站样本上评估
3. 导出为 ONNX 供 Rust 使用

### 添加反混淆技术
1. 在 `obfuscation/` 实现新技术
2. 用真实混淆代码测试
3. 集成到完整系统

## 📖 详细文档

- [../docs/architecture/ARCHITECTURE.md](../docs/architecture/ARCHITECTURE.md) - 项目架构
- [../docs/guides/](../docs/guides/) - 集成指南
- [../docs/references/](../docs/references/) - 快速参考

## ✅ 清理历史

**2026年1月31日**：删除了以下模拟/学习模块：
- ❌ `generators/` - 模拟数据生成器
- ❌ `semantic_learning/` - 学习日志
- ❌ `validation/` - 验证模块
- ❌ `core/` - 实验性基础模型

现在项目 **100% 基于真实数据运行**。
````
- 输入完整网站，输出优化版本
- 功能相同，代码更简洁
- 用于双渲染对比

## 🔧 技术栈

- **模型**：Transformer Encoder-Decoder
- **vocab_size**：229（字符级）
- **架构**：d_model=256, nhead=8, layers=3
- **训练**：30 epochs, batch_size=2
- **输出**：ONNX（用于Rust集成）
