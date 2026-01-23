# 📚 BrowerAI 文档中心

BrowerAI 项目的所有文档、指南和资源。

## 🚀 快速导航（必读）

### 新用户必读（5-15 分钟）

1. **[../QUICK_START.md](../QUICK_START.md)** - 30 秒快速了解项目
2. **[../VERIFICATION_CHECKLIST.md](../VERIFICATION_CHECKLIST.md)** - 验证项目真实性
3. **[USAGE.md](USAGE.md)** - 学习如何使用

### 核心文档（保留在根目录）

| 文档 | 用途 |
|------|------|
| [../README.md](../README.md) | 项目入门 |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | 贡献指南 |
| [../CHANGELOG.md](../CHANGELOG.md) | 版本历史 |
| [../DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) | 部署说明 |

## 📊 项目状态和使用

### 项目现状

| 文档 | 内容 |
|------|------|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | 项目真实状态总结 |
| [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) | 项目整理说明 |
| [USAGE.md](USAGE.md) | API 和功能使用 |

### 技术集成

| 文档 | 内容 |
|------|------|
| [ONNX_RUST_INTEGRATION_GUIDE.md](ONNX_RUST_INTEGRATION_GUIDE.md) | ONNX 和 Rust 集成 |
| [RUST_MODULE_REGISTRATION.md](RUST_MODULE_REGISTRATION.md) | 模块注册细节 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 系统设计与架构 |
| [JS_CENTRIC_ARCHITECTURE.md](JS_CENTRIC_ARCHITECTURE.md) | JavaScript 处理管道 |
| [COMPREHENSIVE_TESTING.md](COMPREHENSIVE_TESTING.md) | 测试策略 |

## 📁 目录结构

```
docs/
├── README.md ................................... 本文档导航索引
├── PROJECT_STATUS.md ........................... 项目状态总结
├── USAGE.md .................................... 功能使用示例
├── CLEANUP_SUMMARY.md .......................... 项目整理说明
├── ONNX_RUST_INTEGRATION_GUIDE.md ............ 集成技术指南
├── RUST_MODULE_REGISTRATION.md ............... 模块注册细节
├── ARCHITECTURE.md ............................. 系统设计
├── JS_CENTRIC_ARCHITECTURE.md ................ JavaScript 处理
├── COMPREHENSIVE_TESTING.md ................... 测试策略
├── TODO.md ..................................... 任务追踪
│
├── archives/ ................................... 历史文档存档 (280+ 文件)
│   ├── ANALYSIS_ARCHIVE.md ................... 分析报告汇总
│   ├── CSS_LEARNING_ARCHIVE.md .............. CSS 学习汇总
│   ├── WEEK_REPORTS_ARCHIVE.md .............. 周报告汇总
│   └── ... (更多历史文档)
│
├── scripts/ .................................... 辅助脚本 (15+ 脚本)
│   ├── run_complete_pipeline.sh ............. 完整管道脚本
│   ├── fast_pipeline.sh ..................... 快速管道脚本
│   ├── run_gpu_training.sh .................. GPU 训练脚本
│   ├── monitor_progress.sh .................. 进度监控
│   └── ... (更多脚本)
│
├── phases/ ..................................... 项目阶段文档
├── book/ ....................................... 完整技术文档
├── en/ ......................................... 英文文档
└── zh-CN/ ...................................... 中文文档
```

## 🎯 按用途查找文档

### 入门与学习
- 📝 [../QUICK_START.md](../QUICK_START.md) - 快速开始（**从这里开始！**）
- 🔍 [../VERIFICATION_CHECKLIST.md](../VERIFICATION_CHECKLIST.md) - 验证项目
- 📊 [PROJECT_STATUS.md](PROJECT_STATUS.md) - 项目现状
- 📚 [USAGE.md](USAGE.md) - 功能使用

### 集成与开发
- 🔧 [ONNX_RUST_INTEGRATION_GUIDE.md](ONNX_RUST_INTEGRATION_GUIDE.md) - ONNX 集成
- 📋 [RUST_MODULE_REGISTRATION.md](RUST_MODULE_REGISTRATION.md) - 模块注册
- 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md) - 系统架构
- 🚀 [../DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) - 部署说明

### 项目管理
- 📜 [../README.md](../README.md) - 项目概述
- ✨ [../CHANGELOG.md](../CHANGELOG.md) - 版本历史
- 🤝 [../CONTRIBUTING.md](../CONTRIBUTING.md) - 贡献指南
- 📋 [TODO.md](TODO.md) - 任务追踪

### 历史与档案
- 🗂️ [archives/](./archives/) - 280+ 历史文档
- 📊 [archives/ANALYSIS_ARCHIVE.md](./archives/ANALYSIS_ARCHIVE.md) - 分析报告
- 🎓 [archives/CSS_LEARNING_ARCHIVE.md](./archives/CSS_LEARNING_ARCHIVE.md) - 学习文档
- 📅 [archives/WEEK_REPORTS_ARCHIVE.md](./archives/WEEK_REPORTS_ARCHIVE.md) - 周报告

## 💻 脚本和工具

所有脚本都在 [docs/scripts/](./scripts/) 中：

```bash
# 主要脚本
docs/scripts/run_complete_pipeline.sh    # 完整训练管道
docs/scripts/fast_pipeline.sh            # 快速训练管道
docs/scripts/run_gpu_training.sh         # GPU 训练

# 监控和工具脚本
docs/scripts/monitor_progress.sh         # 进度监控
docs/scripts/monitor_obfuscation.sh      # 混淆监控
docs/scripts/simple_pipeline.sh          # 简单管道

# 其他脚本
ls docs/scripts/  # 查看所有脚本
```

## 📊 清理统计

| 项目 | 数值 |
|------|------|
| 根目录核心文档 | 6 个 |
| 移到 docs/archives/ 的历史文档 | 280+ 个 |
| 移到 docs/scripts/ 的脚本 | 15+ 个 |
| 保留在根目录的脚本 | 1 个 |

## 🎓 学习路径

```
新用户
  ↓
QUICK_START.md (30 秒)
  ↓
VERIFICATION_CHECKLIST.md (5 分钟)
  ↓
USAGE.md (10 分钟)
  ↓
ONNX_RUST_INTEGRATION_GUIDE.md (可选)
  ↓
开始使用或贡献
```

## 🔗 相关链接

- **项目主页**: [../README.md](../README.md)
- **快速开始**: [../QUICK_START.md](../QUICK_START.md)
- **验证清单**: [../VERIFICATION_CHECKLIST.md](../VERIFICATION_CHECKLIST.md)
- **贡献指南**: [../CONTRIBUTING.md](../CONTRIBUTING.md)
- **部署指南**: [../DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)

---

**上次更新**: 2024-01-29  
**状态**: ✅ 文档已整理和索引化  
**根目录整洁度**: ✅ 完整
