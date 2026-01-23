# 📁 BrowerAI 项目结构总览

本文档描述 BrowerAI 项目的完整目录结构和文件组织。

## 根目录结构

```
/home/stone/BrowerAI/
├── 📄 核心文档 (6 个)
│   ├── README.md ............................. 项目入门
│   ├── QUICK_START.md ....................... 快速开始指南
│   ├── VERIFICATION_CHECKLIST.md ........... 项目验证清单
│   ├── CHANGELOG.md ......................... 版本历史
│   ├── CONTRIBUTING.md ..................... 贡献指南
│   └── DEPLOYMENT_GUIDE.md ................. 部署指南
│
├── 🦀 Rust 项目文件
│   ├── Cargo.toml .......................... 主项目配置
│   ├── Cargo.lock .......................... 依赖锁定
│   ├── src/ ............................... 源代码目录
│   ├── crates/ ............................. Cargo 工作区
│   ├── tests/ ............................. 集成测试
│   ├── examples/ ........................... 示例程序
│   └── target/ ............................ 编译输出
│
├── 🐍 Python 训练/工具
│   ├── training/ ........................... 模型训练脚本
│   ├── real_data/ .......................... 真实数据
│   └── models/ ............................ 模型文件
│
├── 🐳 容器和部署
│   ├── Dockerfile.prod .................... 生产 Dockerfile
│   ├── Dockerfile.api ..................... API Dockerfile
│   ├── docker-compose.yml ................. Docker 编排
│   ├── docker-compose-week3.yml .......... 周期性配置
│   └── Justfile ........................... 任务脚本
│
├── 🔧 工具和配置
│   ├── deny.toml .......................... 依赖检查配置
│   ├── codecov.json ....................... 代码覆盖配置
│   ├── grafana-dashboard.json ............ Grafana 仪表板
│   ├── PHASE_2_SUMMARY.json .............. 阶段总结
│   └── format_with_prettier.sh ........... 代码格式化脚本
│
└── 📚 文档和参考
    └── docs/ ............................ 文档目录（见下文）
```

## 📚 docs 目录详细结构

```
docs/
├── 📄 核心文档（可快速访问）
│   ├── README.md ......................... 📍 文档导航索引（从这里开始）
│   ├── PROJECT_STATUS.md ................. 项目状态
│   ├── USAGE.md .......................... 功能使用指南
│   ├── CLEANUP_SUMMARY.md ............... 项目整理说明
│   ├── ONNX_RUST_INTEGRATION_GUIDE.md .. 集成指南
│   ├── RUST_MODULE_REGISTRATION.md .... 模块注册
│   ├── ARCHITECTURE.md .................. 系统架构
│   ├── JS_CENTRIC_ARCHITECTURE.md ..... JavaScript 处理
│   ├── COMPREHENSIVE_TESTING.md ........ 测试策略
│   └── TODO.md .......................... 任务追踪
│
├── 📦 archives/ ......................... 历史文档存档（280+ 文件）
│   ├── ANALYSIS_ARCHIVE.md .............. 分析报告索引
│   ├── CSS_LEARNING_ARCHIVE.md ......... CSS 学习文档
│   ├── WEEK_REPORTS_ARCHIVE.md ........ 周报告
│   ├── PHASE_*.md ....................... 历史阶段文档
│   ├── TASK_*.md ........................ 历史任务文档
│   ├── WEEK*.md ......................... 历史周文档
│   └── ... (280+ 其他历史文档)
│
├── 🔧 scripts/ .......................... 辅助脚本（15+ 个）
│   ├── run_complete_pipeline.sh ........ 完整训练管道
│   ├── fast_pipeline.sh ................ 快速训练管道
│   ├── run_gpu_training.sh ............ GPU 训练脚本
│   ├── monitor_progress.sh ............ 进度监控
│   ├── monitor_obfuscation.sh ........ 混淆监控
│   ├── simple_pipeline.sh ............ 简单管道
│   └── ... (10+ 其他脚本)
│
├── 📚 book/ ............................. 完整技术文档
│   ├── README.md
│   ├── summary.md
│   ├── chapters/
│   └── ... (完整书籍结构)
│
├── 🌐 en/ .............................. 英文文档
│   ├── GETTING_STARTED.md
│   ├── QUICKREF.md
│   ├── ONNX_TRAINING_GUIDE.md
│   └── ... (英文版本)
│
├── 🇨🇳 zh-CN/ .......................... 中文文档
│   ├── 快速开始.md
│   ├── 快速参考.md
│   └── ... (中文版本)
│
├── 📊 phases/ ........................... 项目阶段文档
│   ├── PHASE1_*.md ...................... 第一阶段
│   ├── PHASE2_*.md ...................... 第二阶段
│   ├── PHASE3_*.md ...................... 第三阶段
│   └── ... (各阶段详细文档)
│
└── 🔓 deobfuscation/ ................... 反混淆技术文档
    ├── strategies.md
    ├── implementations.md
    └── ... (反混淆专题)
```

## 📊 关键目录说明

### src/ - Rust 源代码
```
src/
├── lib.rs .............................. 主库入口
├── main.rs ............................. 示例主程序
├── ai/ ................................ AI 模块
│   ├── inference.rs ................... ONNX 推理
│   ├── model_loader.rs ............... 模型加载
│   ├── hot_reload.rs ................. 热重载
│   └── integration.rs ................ 集成
├── parser/ ............................ 解析器模块
│   ├── html.rs ....................... HTML 解析
│   ├── css.rs ........................ CSS 解析
│   ├── js.rs ........................ JavaScript 解析
│   └── js_analyzer/ ................. JS 深度分析
├── renderer/ ......................... 渲染器模块
├── dom/ ............................. DOM 模块
├── network/ .......................... 网络模块
├── learning/ ......................... 学习系统
└── plugins/ .......................... 插件系统
```

### crates/ - Cargo 工作区
```
crates/
├── browerai-core/ ................... 核心库
├── browerai-ai-integration/ ......... AI 集成（框架检测）
├── browerai-cli/ .................... CLI 工具
└── ... (其他 crate)
```

### training/ - Python 训练脚本
```
training/
├── scripts/ .......................... 训练脚本
│   ├── train_html_parser.py ......... HTML 训练
│   ├── train_css_parser.py ......... CSS 训练
│   ├── train_js_analyzer.py ........ JS 分析训练
│   └── gpu_framework_detector.py ... 框架检测训练
├── models/ .......................... 输出模型
├── data/ ............................ 训练数据
└── ... (其他文件)
```

### real_data/ - 真实数据
```
real_data/
├── obfuscated_code/ ................. 混淆代码数据
│   ├── training_pairs.jsonl ......... 训练对
│   ├── validation_pairs.jsonl ....... 验证对
│   └── ... (其他数据)
├── github_repos/ .................... GitHub 仓库
├── npm_packages/ .................... NPM 包
└── ... (其他数据)
```

### models/ - 模型文件
```
models/
├── local/ ........................... 本地模型
│   ├── large_scale_model.onnx ...... 大规模模型（34MB）
│   ├── large_scale_best.pt ......... PyTorch 版本
│   ├── model_config.toml .......... 模型配置
│   └── ... (其他模型)
├── pretrained/ ..................... 预训练模型
└── ... (其他模型)
```

## 🗂️ 文件组织原则

### 根目录
- ✅ 仅保留必要的配置文件和核心文档
- ✅ 项目的直接入口和指南
- ✅ Cargo.toml, Dockerfile 等配置文件
- ❌ 历史文档、日志、临时文件

### docs/ 目录
- ✅ 所有项目文档集中在此
- ✅ 按功能和用途分类
- ✅ 历史文档统一存放在 archives/
- ✅ 脚本和工具在 scripts/

### 特殊目录
- **docs/archives/** - 280+ 历史文档（不经常访问）
- **docs/scripts/** - 辅助脚本（运维用）
- **docs/book/** - 完整技术文档（深入学习）
- **docs/phases/** - 项目阶段报告（项目管理）

## 📍 导航快速参考

| 用途 | 路径 |
|------|------|
| 快速开始 | [QUICK_START.md](../QUICK_START.md) |
| 验证项目 | [VERIFICATION_CHECKLIST.md](../VERIFICATION_CHECKLIST.md) |
| 文档导航 | [docs/README.md](./README.md) |
| 项目状态 | [docs/PROJECT_STATUS.md](./PROJECT_STATUS.md) |
| 使用指南 | [docs/USAGE.md](./USAGE.md) |
| 集成指南 | [docs/ONNX_RUST_INTEGRATION_GUIDE.md](./ONNX_RUST_INTEGRATION_GUIDE.md) |
| 部署说明 | [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) |
| 历史文档 | [docs/archives/](./archives/) |
| 运维脚本 | [docs/scripts/](./scripts/) |

## 📊 整洁度指标

| 指标 | 当前值 | 目标 |
|------|--------|------|
| 根目录 .md 文件 | 6 个 | < 10 个 ✅ |
| 根目录脚本 | 1 个 | < 5 个 ✅ |
| 历史文档归档率 | 100% | > 90% ✅ |
| 文档分类完整性 | 100% | 100% ✅ |

## 🔄 维护建议

### 添加新文档
1. 确定文档类型（指南、参考、报告等）
2. 根据类型放入相应目录
3. 如果是历史性文档，直接放在 archives/
4. 更新 docs/README.md 的导航索引

### 清理过期文档
1. 移到 docs/archives/
2. 如果有多个相似文档，考虑合并
3. 更新导航索引

### 脚本管理
1. 常用脚本保留在根目录（如 format_with_prettier.sh）
2. 其他脚本放在 docs/scripts/
3. 更新脚本文档中的调用方式

---

**上次更新**: 2024-01-29  
**维护者**: GitHub Copilot  
**整洁度**: ✅ 5/5 星

