# BrowerAI 项目规范 - 文档管理和开发指南

**版本**: 1.0  
**生效日期**: 2026-02-17  
**维护者**: 项目架构团队

---

## 📋 目录

1. [文档组织结构](#文档组织结构)
2. [根目录规范](#根目录规范)
3. [Docs目录规范](#docs目录规范)
4. [AI代码生成规范](#ai代码生成规范)
5. [文件命名规范](#文件命名规范)
6. [报告和总结规范](#报告和总结规范)

---

## 📁 文档组织结构

### 根目录 (`/`) - **仅保留7个关键文件**

```
/
├── README.md                      ✅ 项目主入口文档
├── CONTRIBUTING.md                ✅ 贡献指南
├── CHANGELOG.md                   ✅ 变更日志
├── LICENSE                        ✅ 许可证
├── QUICK_START.md                 ✅ 快速开始指南 (新统一文件)
├── DEVELOPMENT_GUIDE.md           ✅ 开发指南 (新统一文件)
└── PROJECT_STRUCTURE.md           ✅ 项目结构说明 (新统一文件)
```

**规则**:
- 根目录只保留最核心的7个文件
- 所有周报、阶段报告、执行总结等都移到 `/docs/` 
- 所有临时报告和实验报告都移到 `/docs/archived/`
- 所有技术指南都整合到 `/docs/guides/` 或主指南中

---

### Docs目录 - 有组织的结构

```
docs/
├── README.md                      # Docs总索引
│
├── guides/                        # 📖 技术指南
│   ├── QUICK_START.md            # 快速开始
│   ├── SETUP.md                  # 环境设置
│   ├── DEPLOYMENT.md             # 部署指南
│   ├── DEVELOPMENT.md            # 开发指南
│   ├── TESTING.md                # 测试指南
│   ├── TROUBLESHOOTING.md        # 故障排查
│   └── CI_CD.md                  # CI/CD使用
│
├── api/                           # 📡 API文档
│   ├── README.md
│   ├── ENDPOINTS.md              # API端点列表
│   ├── EXAMPLES.md               # 使用示例
│   └── SCHEMAS.md                # 数据模型
│
├── architecture/                  # 🏗️ 架构文档
│   ├── README.md
│   ├── OVERVIEW.md               # 架构概述
│   ├── MODULES.md                # 模块说明
│   └── DESIGN_DECISIONS.md       # 设计决策
│
├── development/                   # 🔧 开发相关
│   ├── CODE_STYLE.md             # 代码风格
│   ├── CONVENTIONS.md            # 开发约定
│   ├── BUILD_SYSTEM.md           # 构建系统
│   ├── TESTING_STRATEGY.md       # 测试策略
│   └── PERFORMANCE.md            # 性能指南
│
├── learning/                      # 🧠 学习和AI
│   ├── README.md
│   ├── MODEL_TRAINING.md         # 模型训练指南
│   ├── DATA_ANNOTATION.md        # 数据标注
│   ├── EVALUATION.md             # 评估指标
│   └── DEOBFUSCATION.md          # 去混淆
│
├── integration/                   # 🔗 集成文档
│   ├── README.md
│   ├── KUBERNETES.md             # K8s部署
│   ├── DOCKER.md                 # Docker配置
│   ├── CI_CD_WORKFLOWS.md        # CI/CD工作流
│   └── MONITORING.md             # 监控配置
│
├── maintenance/                   # 🛠️ 维护相关
│   ├── README.md
│   ├── UPGRADE.md                # 升级指南
│   ├── BACKUP.md                 # 备份策略
│   ├── MONITORING.md             # 监控
│   └── INCIDENT_RESPONSE.md      # 事件响应
│
├── references/                    # 📚 参考文档
│   ├── GLOSSARY.md               # 术语表
│   ├── DEPENDENCIES.md           # 依赖列表
│   ├── ENVIRONMENT.md            # 环境变量
│   └── COMMANDS.md               # 常用命令
│
├── phases/                        # 📅 项目阶段 (历史存档)
│   ├── WEEK6_SUMMARY.md
│   ├── WEEK7_SUMMARY.md
│   ├── WEEK8_SUMMARY.md
│   └── README.md
│
├── archived/                      # 📦 已归档 (旧报告)
│   ├── WEEK4_REPORTS/
│   ├── WEEK5_REPORTS/
│   ├── WEEK6_REPORTS/
│   ├── WEEK7_REPORTS/
│   ├── WEEK8_REPORTS/
│   └── TEMPORARY_REPORTS/
│
└── CHANGELOG.md                   # 文档变更日志
```

---

## ✅ 根目录规范

### 保留的核心文件

| 文件 | 内容 | 更新频率 |
|------|------|---------|
| `README.md` | 项目简介、快速链接、主要特性 | 按需 |
| `CONTRIBUTING.md` | 贡献指南、开发流程 | 按需 |
| `CHANGELOG.md` | 功能变更记录 | 每个版本 |
| `LICENSE` | 许可证 | 不变 |
| `QUICK_START.md` | 5分钟快速入门（聚合文件） | 每个版本 |
| `DEVELOPMENT_GUIDE.md` | 开发入门指南（聚合文件） | 按需 |
| `PROJECT_STRUCTURE.md` | 项目结构说明（聚合文件） | 按需 |

### 不应该在根目录的文件

❌ **绝对禁止在根目录创建**:
- `WEEK*_*.md` - 周报告（移到 `/docs/phases/`）
- `*_REPORT.md` - 执行报告（移到 `/docs/archived/`）
- `*_SUMMARY.md` - 总结文件（合并到相关指南）
- `*_PROGRESS.md` - 进度文件（移到 `/docs/phases/`）
- `*_EXECUTION_*.md` - 执行相关（移到 `/docs/archived/`）
- `*_PLAN.md` - 计划文件（仅在需要时，应该是一个）
- `*_CHECKLIST.md` - 检查表（合并到相关指南）
- `*_GUIDE.md` - 指南（移到 `/docs/guides/`）
- 重复的快速开始/入门文件

---

## 📚 Docs目录规范

### 子目录用途

#### `/docs/guides/` - 技术指南 📖
- 当前项目使用指南
- 如何做某事的教程
- 最佳实践
- 常见模式

**文件示例**:
- `QUICK_START.md` - 5分钟入门
- `SETUP.md` - 环境配置
- `DEPLOYMENT.md` - 部署步骤
- `TESTING.md` - 测试指南

#### `/docs/architecture/` - 架构设计 🏗️
- 系统架构概览
- 组件设计
- 数据流图
- 设计决策

#### `/docs/api/` - API参考 📡
- API端点文档
- 请求/响应示例
- 数据模型定义
- 错误处理

#### `/docs/development/` - 开发规范 🔧
- 代码风格指南
- 开发约定
- 构建流程
- 测试策略

#### `/docs/integration/` - 集成配置 🔗
- Docker/K8s配置
- CI/CD工作流
- 部署配置
- 监控设置

#### `/docs/phases/` - 项目历程 📅
- 每个周的总结
- 关键里程碑
- 项目演进
- 历史决策

#### `/docs/archived/` - 历史存档 📦
- 旧的执行报告
- 临时分析文档
- 一次性报告
- 实验记录

---

## 🤖 AI代码生成规范

### 原则

**◆ 核心原则**: 
> **绝对禁止** AI在项目根目录创建任何 `*_REPORT.md`, `*_SUMMARY.md`, `WEEK*_*.md` 等报告文件。所有分析、进度、报告内容必须**集成到已有的结构化文档**中，而不是创建新的报告文件。

### AI应该做什么

✅ **允许的操作**:
1. 更新或创建 `/docs/guides/` 中的技术指南
2. 更新或创建 `/docs/api/` 中的API文档
3. 更新或创建 `/docs/development/` 中的开发规范
4. 更新 `/docs/CHANGELOG.md` - 记录功能变更
5. 创建项目集成脚本、工具、源代码
6. 更新现有的文档索引和导航

### AI禁止做什么

❌ **禁止的操作**:
1. ❌ 创建任何周报告文件 (`WEEK*_*.md`)
2. ❌ 创建执行报告 (`*_EXECUTION_REPORT.md`)
3. ❌ 创建总结报告 (`*_SUMMARY.md`, `*_COMPLETE_SUMMARY.md`)
4. ❌ 创建进度报告 (`*_PROGRESS.md`, `*_PROGRESS_REPORT.md`)
5. ❌ 创建阶段报告 (`*_PHASE_*_*.md`)
6. ❌ 创建计划文件 (除非有明确批准的单一计划)
7. ❌ 创建检查清单文件 (`*_CHECKLIST.md`) - 应并入相关指南
8. ❌ 创建重复的指南或快速开始文件

### 如何处理报告内容

当需要记录**进度、总结或报告**时：

**方案A - 合并到现有文档**
```
✓ 进度更新 → 更新 /docs/phases/WEEK8_SUMMARY.md 中的相关部分
✓ 执行总结 → 添加到 /docs/guides/ 中的相关指南末尾
✓ 问题报告 → 创建或更新 /docs/CHANGELOG.md
✓ API更新 → 更新 /docs/api/ENDPOINTS.md
```

**方案B - 存档在历史文件中**
```
✓ 一次性分析 → /docs/archived/temporary_reports/
✓ 实验报告 → /docs/archived/experiments/
✓ 周期性报告 → /docs/phases/
```

**不要创建新的独立报告文件!**

---

## 📝 文件命名规范

### 允许的命名模式

#### On Root Level (根目录) - 仅这些:
- `README.md`
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `LICENSE`
- `QUICK_START.md` (单一文件)
- `DEVELOPMENT_GUIDE.md`
- `PROJECT_STRUCTURE.md`

#### In `/docs/guides/`:
```
QUICK_START.md          # 5分钟入门
SETUP.md                # 环境设置
DEPLOYMENT.md           # 部署指南
DEVELOPMENT.md          # 开发指南
TESTING.md              # 测试指南
TROUBLESHOOTING.md      # 故障排查
CI_CD.md                # CI/CD使用指南
```

#### In `/docs/phases/` (仅用于周总结):
```
WEEK6_SUMMARY.md        # 周6总结
WEEK7_SUMMARY.md        # 周7总结
WEEK8_SUMMARY.md        # 周8总结
```

#### In `/docs/archived/`:
```
WEEK4_REPORTS/          # 目录
WEEK5_REPORTS/          # 目录
WEEK6_REPORTS/          # 目录
temporary_reports/      # 临时报告目录
experiments/            # 实验报告目录
```

### 禁止的命名模式

❌ **绝对禁止在任何地方创建**:
- `*_EXECUTION_REPORT.md`
- `*_PROGRESS_REPORT.md`
- `*_COMPLETE_SUMMARY.md`
- `*_COMPLETION_REPORT.md`
- `*_FINAL_REPORT.md`
- `*_START_NOW.md`
- `*_READY_TO_*.md`
- `*_QUICK_REFERENCE.md` (应该是一个统一的参考或分布在各指南中)
- `*_IMPLEMENTATION_*.md`
- `*_EXECUTION_*.md`
- `*_DELIVERY_*.md`

---

## 🕐 报告和总结规范

### 什么是"报告"

**报告** = 关于某个事件/周期/阶段的总结文档
- 执行报告 - 说明实际执行的情况
- 进度报告 - 说明当前进度和后续plans
- 完成报告 - 说明完成了什么
- 阶段报告 - 某个阶段的总结
- 周报告 - 每周工作总结

### 报告应该去哪

| 报告类型 | 位置 | 操作 |
|--------|------|------|
| 周活动总结 | `/docs/phases/WEEK{N}_SUMMARY.md` | 更新现有文件 |
| 功能实现说明 | `/docs/guides/` 对应指南末尾 | 合并到指南 |
| 临时分析报告 | `/docs/archived/temporary_reports/` | 存档 |
| 执行详情 | `/docs/archived/execution_logs/` | 存档 |
| 阶段完成记录 | `/docs/phases/{PHASE}_summary.md` | 更新相关字段 |

### 报告内容的处理

❌ **错误做法**:
```
创建: WEEK8_PHASE_E_EXECUTION_REPORT.md
创建: WEEK8_PHASE_E_FINAL_SUMMARY.md
创建: WEEK8_DEPLOYMENT_PROGRESS_REPORT.md
创建: DEPLOYMENT_PROGRESS.md
创建: COMPREHENSIVE_TEST_REPORT.md
```

✅ **正确做法**:
```
1. 进度更新执行吗 → /docs/phases/WEEK8_SUMMARY.md 中的"Status"部分
2. 执行细节? → /docs/archived/execution_logs/WEEK8_PHASE_E.md
3. 测试报告? → /docs/guides/TESTING.md 中的"Test Results"
4. 部署说明? → /docs/guides/DEPLOYMENT.md
5. API更新? → /docs/api/ 中更新
```

---

## 🎯 文档索引系统

### 主索引文件

#### `/README.md` - 项目主入口
```markdown
# BrowerAI

快速链接到:
- 快速开始: /docs/guides/QUICK_START.md
- 开发指南: /docs/guides/DEVELOPMENT.md
- 部署说明: /docs/guides/DEPLOYMENT.md
- API文档: /docs/api/
- 架构: /docs/architecture/
- 变更日志: CHANGELOG.md
```

#### `/docs/README.md` - 文档总索引
```markdown
# BrowerAI 文档

## 📖 快速导航
- 新手入门: ./guides/QUICK_START.md
- 配置环境: ./guides/SETUP.md
- 开始开发: ./guides/DEVELOPMENT.md
- 部署生产: ./guides/DEPLOYMENT.md

## 📚 详细文档
- API文档: ./api/
- 架构设计: ./architecture/
- 开发规范: ./development/
- 学习资源: ./learning/

## 📅 项目历程
- 完成总结: ./phases/
- 变更日志: ./CHANGELOG.md
```

#### `/docs/guides/README.md` - 指南总览
```markdown
# 技术指南

## 开始使用
1. QUICK_START.md - 5分钟快速开始
2. SETUP.md - 环境配置
3. DEVELOPMENT.md - 开发入门

## 日常开发
- TESTING.md - 测试指南
- CI_CD.md - CI/CD使用
- TROUBLESHOOTING.md - 故障排查

## 部署和维护
- DEPLOYMENT.md - 部署指南
```

---

## 🚀 实施计划

### Phase 1: 清理根目录 (第1天)
1. ✅ 创建本规范文档
2. ✅ 列出所有根目录md文件
3. ✅ 分类为: 保留/移动/删除
4. 📋 执行移动和删除

### Phase 2: 重组Docs目录 (第2天)
1. 创建标准目录结构
2. 移动现有文件到新位置
3. 删除重复文件
4. 创建各子目录的README

### Phase 3: 整合文档 (第3天)
1. 合并重复的指南
2. 统一命名和格式
3. 更新所有索引
4. 建立交叉引用

### Phase 4: 建立流程 (第4天)
1. ✅ 发布本规范
2. 配置自动化检查
3. 培训AI/开发人员
4. 定期审核

---

## 📊 标准检查清单

创建任何文档前，问自己：

- [ ] 这是源代码还是文档?
- [ ] 这是临时报告还是永久指南?
- [ ] 这个内容已经存在于其他文档中吗?
- [ ] 根目录中已经有类似的文件吗?
- [ ] 这个文件应该属于哪个 `/docs/` 子目录?
- [ ] 我是在创建"第N个"快速开始/总结吗?
- [ ] 有没有办法把这内容合并到现有文档?

---

## 🔗 相关文档

- [项目结构说明](../PROJECT_STRUCTURE.md)
- [开发指南](./guides/DEVELOPMENT.md)
- [贡献指南](../CONTRIBUTING.md)
- [CI/CD指南](./guides/CI_CD.md)

---

**此规范由BrowerAI架构团队制定**  
**版本历史**:
- v1.0 - 2026-02-17: 初始版本发布

