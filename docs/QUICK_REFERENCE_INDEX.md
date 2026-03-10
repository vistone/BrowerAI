# 🗺️ BrowerAI 文档快速索引

**版本**: 1.0  
**日期**: 2026-02-17  
**用途**: 快速找到所需文档

---

## 🎯 按角色导航

### 👶 我是新手（第一次了解BrowerAI）

**推荐阅读顺序**:
1. [README.md](../README.md) - 5分钟了解项目概况
2. [CORE_DESIGN_PHILOSOPHY.md](CORE_DESIGN_PHILOSOPHY.md) - 10分钟理解核心理念
3. [GETTING_STARTED.md](../GETTING_STARTED.md) - 30分钟快速上手
4. [PROJECT_EVOLUTION_STORY.md](PROJECT_EVOLUTION_STORY.md) - 15分钟了解项目历史

**总时间**: 约1小时

---

### 💻 我是开发者（想使用BrowerAI）

#### 快速开始
- [GETTING_STARTED.md](../GETTING_STARTED.md) - 安装与基础使用
- [examples/](../examples/) - 代码示例
  - [basic_usage.rs](../examples/basic_usage.rs) - 基础解析示例
  - [js_deobfuscator_demo.rs](../examples/js_deobfuscator_demo.rs) - 反混淆演示
  - [dual_rendering_demo.rs](../examples/dual_rendering_demo.rs) - 渲染演示

#### API文档
- [在线API文档](https://docs.rs/browerai) - 完整Rust API
- [crates/README.md](../crates/) - 各crate功能说明

#### 问题排查
- [docs/maintenance/TROUBLESHOOTING.md](maintenance/TROUBLESHOOTING.md) - 常见问题
- [GitHub Issues](https://github.com/YOUR_USERNAME/BrowerAI/issues) - 提问

**总时间**: 2-3小时上手

---

### 🔬 我是研究者（想深入理解技术）

#### 深度学习路径
1. **第一周：基础理解**
   - [LEARNING_PATH.md](LEARNING_PATH.md) - 完整学习路径（必读）
   - [CORE_DESIGN_PHILOSOPHY.md](CORE_DESIGN_PHILOSOPHY.md) - 设计哲学
   - [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) - 架构总览

2. **第二周：技术深入**
   - [TECHNICAL_IMPLEMENTATION.md](TECHNICAL_IMPLEMENTATION.md) - 核心算法实现
   - [docs/phases/](phases/) - 各开发阶段文档
   - [Source Code Reading](LEARNING_PATH.md#第5阶段-源码精读建议阅读顺序) - 源码阅读顺序

3. **第三周：实验验证**
   - [tests/](../tests/) - 测试用例学习
   - [training/](../training/) - 训练流程研究
   - [docs/testing/](testing/) - 测试方法论

#### 关键技术论文
- [docs/references/](references/) - 参考文献
- [docs/deobfuscation/](deobfuscation/) - 反混淆技术

**总时间**: 3周深度学习

---

### 🤝 我是贡献者（想为项目贡献代码）

#### 必读文档
1. [CONTRIBUTING.md](../CONTRIBUTING.md) - 贡献指南
2. [DEVELOPMENT_GUIDE.md](../DEVELOPMENT_GUIDE.md) - 开发规范
3. [docs/PROJECT_STANDARDS.md](PROJECT_STANDARDS.md) - 项目标准

#### 开发流程
1. **环境搭建**
   - [GETTING_STARTED.md](../GETTING_STARTED.md#开发环境)
   - [.github/workflows/](../.github/workflows/) - CI配置

2. **代码风格**
   - [DEVELOPMENT_GUIDE.md#代码风格](../DEVELOPMENT_GUIDE.md) - Rust风格指南
   - [.github/copilot-instructions.md](../.github/copilot-instructions.md) - AI辅助规范

3. **测试要求**
   - [docs/testing/](testing/) - 测试策略
   - [tests/](../tests/) - 测试示例

4. **提交PR**
   - Pull Request模板
   - Code Review流程

**Good First Issues**:
- 查看[GitHub Issues标签"good first issue"](https://github.com/YOUR_USERNAME/BrowerAI/labels/good%20first%20issue)

---

### 🎓 我是教师/学生（用于教学/学习）

#### 教学资源
- [LEARNING_PATH.md](LEARNING_PATH.md) - 5阶段渐进式学习
- [PROJECT_EVOLUTION_STORY.md](PROJECT_EVOLUTION_STORY.md) - 项目演进案例研究
- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) - 技术选型决策分析

#### 课程模块建议
1. **Week 1-2: Rust基础** - 通过Parser代码学习Rust
2. **Week 3-4: 算法与数据结构** - DFS/BFS在CFG分析中的应用
3. **Week 5-6: 机器学习** - ONNX模型训练与部署
4. **Week 7-8: 软件工程** - 模块化架构设计
5. **Week 9-10: 项目实战** - 实现自己的反混淆策略

#### 作业与项目
- [examples/](../examples/) - 参考实现
- [tests/](../tests/) - 验证标准

---

## 📚 按主题导航

### 核心概念

| 主题 | 文档 | 难度 | 时间 |
|-----|------|------|------|
| 什么是BrowerAI？ | [README.md](../README.md) | ⭐ | 5min |
| 核心设计理念 | [CORE_DESIGN_PHILOSOPHY.md](CORE_DESIGN_PHILOSOPHY.md) | ⭐⭐ | 15min |
| "保功能、换体验" | [CORE_DESIGN_PHILOSOPHY.md#核心口号](CORE_DESIGN_PHILOSOPHY.md) | ⭐⭐ | 10min |
| 项目历史 | [PROJECT_EVOLUTION_STORY.md](PROJECT_EVOLUTION_STORY.md) | ⭐⭐ | 20min |

### 架构与设计

| 主题 | 文档 | 难度 | 时间 |
|-----|------|------|------|
| 总体架构 | [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) | ⭐⭐⭐ | 30min |
| 27个crate说明 | [crates/README.md](../crates/) | ⭐⭐ | 20min |
| 模块依赖关系 | [LEARNING_PATH.md#依赖关系图](LEARNING_PATH.md) | ⭐⭐⭐ | 15min |
| 为什么这样设计？ | [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) | ⭐⭐⭐⭐ | 45min |

### 技术实现

| 主题 | 文档 | 难度 | 时间 |
|-----|------|------|------|
| 7阶段JS分析 | [TECHNICAL_IMPLEMENTATION.md#第2章](TECHNICAL_IMPLEMENTATION.md) | ⭐⭐⭐⭐ | 60min |
| DFS/BFS算法 | [TECHNICAL_IMPLEMENTATION.md#第1章](TECHNICAL_IMPLEMENTATION.md) | ⭐⭐⭐⭐ | 45min |
| 18种反混淆策略 | [TECHNICAL_IMPLEMENTATION.md#1.4](TECHNICAL_IMPLEMENTATION.md) | ⭐⭐⭐⭐⭐ | 90min |
| 智能渲染系统 | [TECHNICAL_IMPLEMENTATION.md#第3章](TECHNICAL_IMPLEMENTATION.md) | ⭐⭐⭐⭐ | 60min |
| 学习系统 | [TECHNICAL_IMPLEMENTATION.md#第4章](TECHNICAL_IMPLEMENTATION.md) | ⭐⭐⭐⭐ | 45min |
| 性能优化 | [TECHNICAL_IMPLEMENTATION.md#第6章](TECHNICAL_IMPLEMENTATION.md) | ⭐⭐⭐ | 30min |

### 数据与训练

| 主题 | 文档 | 难度 | 时间 |
|-----|------|------|------|
| 训练快速开始 | [training/QUICKSTART.md](../training/QUICKSTART.md) | ⭐⭐ | 20min |
| 数据收集流程 | [PROJECT_EVOLUTION_STORY.md#5.1](PROJECT_EVOLUTION_STORY.md) | ⭐⭐⭐ | 30min |
| 48维特征工程 | [TECHNICAL_IMPLEMENTATION.md#5.2](TECHNICAL_IMPLEMENTATION.md) | ⭐⭐⭐⭐ | 45min |
| 模型训练配置 | [training/](../training/) | ⭐⭐⭐⭐ | 60min |
| ONNX导出 | [models/README.md](../models/README.md) | ⭐⭐⭐ | 30min |

### 开发与贡献

| 主题 | 文档 | 难度 | 时间 |
|-----|------|------|------|
| 快速上手 | [GETTING_STARTED.md](../GETTING_STARTED.md) | ⭐⭐ | 30min |
| 开发环境搭建 | [DEVELOPMENT_GUIDE.md](../DEVELOPMENT_GUIDE.md) | ⭐⭐ | 20min |
| 代码风格规范 | [DEVELOPMENT_GUIDE.md#风格](../DEVELOPMENT_GUIDE.md) | ⭐⭐ | 15min |
| 测试方法 | [docs/testing/](testing/) | ⭐⭐⭐ | 40min |
| CI/CD流程 | [docs/CICD_USAGE_GUIDE.md](CICD_USAGE_GUIDE.md) | ⭐⭐⭐ | 30min |
| 如何提交PR | [CONTRIBUTING.md](../CONTRIBUTING.md) | ⭐⭐ | 20min |

### 部署与运维

| 主题 | 文档 | 难度 | 时间 |
|-----|------|------|------|
| API服务器部署 | [config/README.md](../config/README.md) | ⭐⭐⭐ | 30min |
| Docker部署 | [Dockerfile.prod](../Dockerfile.prod) | ⭐⭐⭐ | 20min |
| Kubernetes | [k8s/](../k8s/) | ⭐⭐⭐⭐ | 60min |
| 监控与告警 | [config/prometheus.yml](../config/prometheus.yml) | ⭐⭐⭐ | 30min |
| 性能调优 | [docs/maintenance/](maintenance/) | ⭐⭐⭐⭐ | 45min |

---

## 🔍 按问题导航

### 安装与配置

**Q: 如何安装BrowerAI？**  
→ [GETTING_STARTED.md#安装](../GETTING_STARTED.md)

**Q: 需要什么依赖？**  
→ [README.md#依赖](../README.md)

**Q: 如何配置开发环境？**  
→ [DEVELOPMENT_GUIDE.md#环境搭建](../DEVELOPMENT_GUIDE.md)

**Q: 为什么编译失败？**  
→ [docs/maintenance/TROUBLESHOOTING.md](maintenance/TROUBLESHOOTING.md)

### 使用与功能

**Q: 如何解析HTML/CSS/JS？**  
→ [examples/basic_usage.rs](../examples/basic_usage.rs)

**Q: 如何反混淆JavaScript？**  
→ [examples/js_deobfuscator_demo.rs](../examples/js_deobfuscator_demo.rs)

**Q: 如何生成不同风格的网站？**  
→ [examples/dual_rendering_demo.rs](../examples/dual_rendering_demo.rs)

**Q: 如何训练自己的模型？**  
→ [training/QUICKSTART.md](../training/QUICKSTART.md)

**Q: "保功能、换体验"是什么意思？**  
→ [CORE_DESIGN_PHILOSOPHY.md#核心口号](CORE_DESIGN_PHILOSOPHY.md)

### 技术细节

**Q: 7阶段JS分析是什么？**  
→ [TECHNICAL_IMPLEMENTATION.md#第2章](TECHNICAL_IMPLEMENTATION.md)

**Q: DFS循环检测如何工作？**  
→ [TECHNICAL_IMPLEMENTATION.md#1.1节](TECHNICAL_IMPLEMENTATION.md)

**Q: 18种反混淆策略有哪些？**  
→ [TECHNICAL_IMPLEMENTATION.md#1.4节](TECHNICAL_IMPLEMENTATION.md)

**Q: 多层缓存如何实现53.77x加速？**  
→ [TECHNICAL_IMPLEMENTATION.md#6.1节](TECHNICAL_IMPLEMENTATION.md)

**Q: 为什么选择Rust而非C++？**  
→ [DESIGN_DECISIONS.md#决策1](DESIGN_DECISIONS.md)

**Q: 为什么27个crate？**  
→ [DESIGN_DECISIONS.md#决策4](DESIGN_DECISIONS.md)

### 贡献与开发

**Q: 如何为项目贡献代码？**  
→ [CONTRIBUTING.md](../CONTRIBUTING.md)

**Q: 有哪些Good First Issue？**  
→ [GitHub Issues](https://github.com/YOUR_USERNAME/BrowerAI/labels/good%20first%20issue)

**Q: 如何运行测试？**  
→ [DEVELOPMENT_GUIDE.md#测试](../DEVELOPMENT_GUIDE.md)

**Q: 代码风格规范是什么？**  
→ [docs/PROJECT_STANDARDS.md](PROJECT_STANDARDS.md)

**Q: 如何添加新的反混淆策略？**  
→ [docs/deobfuscation/](deobfuscation/)

### 部署与运维

**Q: 如何部署到生产环境？**  
→ [config/README.md](../config/README.md)

**Q: 如何监控系统性能？**  
→ [grafana/](../grafana/)

**Q: 如何处理大流量？**  
→ [docs/maintenance/SCALABILITY.md](maintenance/)

**Q: 如何备份与恢复？**  
→ [docs/maintenance/](maintenance/)

---

## 📖 完整文档列表

### 根目录文档

| 文件 | 描述 | 适合人群 |
|-----|------|---------|
| [README.md](../README.md) | 项目概览与快速开始 | 所有人 |
| [GETTING_STARTED.md](../GETTING_STARTED.md) | 详细安装与使用教程 | 新手、开发者 |
| [DEVELOPMENT_GUIDE.md](../DEVELOPMENT_GUIDE.md) | 开发者指南 | 贡献者 |
| [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) | 项目结构说明 | 开发者、研究者 |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | 贡献指南 | 贡献者 |
| [CHANGELOG.md](../CHANGELOG.md) | 版本历史 | 所有人 |
| [LICENSE](../LICENSE) | 许可证 | 所有人 |

### docs/ 核心文档

| 文件 | 描述 | 难度 | 时间 |
|-----|------|------|------|
| [CORE_DESIGN_PHILOSOPHY.md](CORE_DESIGN_PHILOSOPHY.md) | 核心设计哲学（必读） | ⭐⭐ | 20min |
| [LEARNING_PATH.md](LEARNING_PATH.md) | 完整学习路径 | ⭐⭐ | 30min |
| [TECHNICAL_IMPLEMENTATION.md](TECHNICAL_IMPLEMENTATION.md) | 技术实现细节 | ⭐⭐⭐⭐ | 120min |
| [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) | 设计决策日志 | ⭐⭐⭐ | 60min |
| [PROJECT_EVOLUTION_STORY.md](PROJECT_EVOLUTION_STORY.md) | 项目演进故事 | ⭐⭐ | 30min |
| [QUICK_REFERENCE_INDEX.md](QUICK_REFERENCE_INDEX.md) | 本文档 | ⭐ | 5min |

### docs/ 子目录

| 目录 | 内容 | 适合人群 |
|-----|------|---------|
| [docs/api/](api/) | API设计文档 | 开发者 |
| [docs/architecture/](architecture/) | 架构设计细节 | 架构师、研究者 |
| [docs/deobfuscation/](deobfuscation/) | 反混淆技术文档 | 研究者 |
| [docs/guides/](guides/) | 使用指南 | 开发者 |
| [docs/learning/](learning/) | 学习资源 | 新手、学生 |
| [docs/maintenance/](maintenance/) | 运维文档 | 运维人员 |
| [docs/phases/](phases/) | 开发阶段记录 | 研究者、贡献者 |
| [docs/references/](references/) | 参考文献 | 研究者 |
| [docs/testing/](testing/) | 测试文档 | 开发者、贡献者 |
| [docs/zh-CN/](zh-CN/) | 中文文档 | 中文用户 |

### 训练相关

| 文件/目录 | 描述 | 适合人群 |
|----------|------|---------|
| [training/QUICKSTART.md](../training/QUICKSTART.md) | 训练快速开始 | ML工程师 |
| [training/README.md](../training/README.md) | 训练流程详解 | ML工程师、研究者 |
| [training/scripts/](../training/scripts/) | 训练脚本 | ML工程师 |
| [models/README.md](../models/README.md) | 模型说明 | 开发者、研究者 |
| [models/MODEL_ZOO.md](../models/MODEL_ZOO.md) | 模型库 | 所有人 |

### 示例代码

| 文件 | 描述 | 难度 |
|-----|------|------|
| [examples/basic_usage.rs](../examples/basic_usage.rs) | 基础使用示例 | ⭐⭐ |
| [examples/comprehensive_demo.rs](../examples/comprehensive_demo.rs) | 完整流程演示 | ⭐⭐⭐ |
| [examples/js_deobfuscator_demo.rs](../examples/js_deobfuscator_demo.rs) | 反混淆演示 | ⭐⭐⭐ |
| [examples/dual_rendering_demo.rs](../examples/dual_rendering_demo.rs) | 双渲染演示 | ⭐⭐⭐ |
| [examples/framework_detection_demo.rs](../examples/framework_detection_demo.rs) | 框架检测演示 | ⭐⭐ |

### 测试文件

| 目录 | 描述 | 适合人群 |
|-----|------|---------|
| [tests/](../tests/) | 集成测试 | 开发者、贡献者 |
| [crates/*/tests/](../crates/) | 单元测试 | 开发者 |

---

## 🎓 学习路径推荐

### 路径A：我想快速使用（2小时）

```
1. README.md（5min）
   ↓
2. GETTING_STARTED.md（30min）
   ↓
3. examples/basic_usage.rs（实践30min）
   ↓
4. examples/js_deobfuscator_demo.rs（实践30min）
   ↓
5. API文档查阅（30min）
```

### 路径B：我想深入理解（2周）

```
Week 1:
  Day 1-2: LEARNING_PATH.md + CORE_DESIGN_PHILOSOPHY.md
  Day 3-4: PROJECT_STRUCTURE.md + 架构图理解
  Day 5-7: TECHNICAL_IMPLEMENTATION.md（逐章精读）

Week 2:
  Day 8-10: 源码阅读（按LEARNING_PATH顺序）
  Day 11-12: 训练流程实践
  Day 13-14: 实现自己的反混淆策略
```

### 路径C：我想贡献代码（1周）

```
Day 1: CONTRIBUTING.md + DEVELOPMENT_GUIDE.md
       ↓
Day 2-3: 环境搭建 + 运行测试 + 代码风格学习
       ↓
Day 4-5: 选择Issue + 实现功能
       ↓
Day 6: 编写测试 + 文档
       ↓
Day 7: 提交PR + Code Review
```

---

## 🆘 获取帮助

### 文档内找不到？

1. **搜索工具**:
   ```bash
   # 在项目根目录
   grep -r "关键词" docs/
   ```

2. **查看索引**:
   - 本文档的[按问题导航](#按问题导航)部分
   - 各文档内部的目录

3. **API文档**:
   ```bash
   cargo doc --open
   ```

### 还是没找到？

- **GitHub Issues**: [提问](https://github.com/YOUR_USERNAME/BrowerAI/issues/new)
- **Discussions**: [讨论区](https://github.com/YOUR_USERNAME/BrowerAI/discussions)
- **Email**: browerai@example.com

---

## 📊 文档统计

```
总文档数量: 50+
总文字量: 100,000+字
代码示例: 200+段
图表数量: 30+个
维护状态: ✅ 活跃更新

最后更新: 2026-02-17
文档版本: v1.0
项目版本: v1.0.0
```

---

## ✨ 推荐阅读组合

### 组合1：完整理解BrowerAI（3小时）
```
README.md → CORE_DESIGN_PHILOSOPHY.md → PROJECT_EVOLUTION_STORY.md
```

### 组合2：技术深度游（5小时）
```
LEARNING_PATH.md → TECHNICAL_IMPLEMENTATION.md → DESIGN_DECISIONS.md
```

### 组合3：快速上手开发（2小时）
```
GETTING_STARTED.md → examples/ → DEVELOPMENT_GUIDE.md
```

### 组合4：学术研究（1周）
```
CORE_DESIGN_PHILOSOPHY.md → TECHNICAL_IMPLEMENTATION.md → 
SOURCE CODE → 论文阅读 → 实验验证
```

---

**提示**: 所有文档路径均相对于项目根目录 `/home/stone/BrowerAI/`

**记住**: 学习没有固定路径，根据你的兴趣和需求自由探索！🚀
