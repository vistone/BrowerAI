# 🏗️ BrowerAI 项目结构说明

**版本**: 2026.02.17

---

## 📊 项目概览

```
BrowerAI/                   # 项目根目录
├── 文档                    # 7个核心根文件
├── crates/                 # 27个Rust模块
├── docs/                   # 完整文档库
├── tests/                  # 集成测试
├── training/               # AI模型训练
├── models/                 # 预训练模型
├── data/                   # 数据集
├── scripts/                # 工具脚本
├── config/                 # 配置文件
├── k8s/                    # Kubernetes清单
└── grafana/                # 监控配置
```

---

## 📁 目录详解

### 根目录 (7个关键文件)

| 文件 | 用途 |
|------|------|
| `README.md` | 项目主介绍 |
| `QUICK_START.md` | 5分钟快速开始 |
| `DEVELOPMENT_GUIDE.md` | 开发指南 |
| `PROJECT_STRUCTURE.md` | 本文件 |
| `CHANGELOG.md` | 版本变更记录 |
| `CONTRIBUTING.md` | 贡献指南 |
| `LICENSE` | MIT许可证 |

---

### crates/ - Rust模块库 (27个Crates)

#### 核心模块

```
crates/browerai/                      # 主库 (大部分业务逻辑)
├── src/
│   ├── lib.rs                        # 库入口
│   ├── ai/                           # AI集成
│   │   ├── inference.rs              # ONNX推理
│   │   ├── model_loader.rs           # 模型加载
│   │   ├── hot_reload.rs             # 热重载
│   │   └── integration.rs            # 集成接口
│   │
│   ├── parser/                       # 解析器
│   │   ├── html.rs                   # HTML5解析
│   │   ├── css.rs                    # CSS解析
│   │   ├── js.rs                     # JavaScript解析
│   │   └── js_analyzer/              # JS深度分析
│   │       ├── scope_analyzer.rs
│   │       ├── dataflow_analyzer.rs
│   │       ├── controlflow_analyzer.rs
│   │       ├── enhanced_call_graph.rs
│   │       └── analysis_pipeline.rs
│   │
│   ├── renderer/                     # 渲染引擎
│   │   ├── layout.rs                 # 布局计算
│   │   ├── paint.rs                  # 绘制
│   │   └── predictive.rs             # 预测优化
│   │
│   ├── dom/                          # DOM模型
│   │   ├── node.rs                   # 节点定义
│   │   └── sandbox.rs                # JS沙箱
│   │
│   ├── learning/                     # 学习系统
│   │   ├── feedback.rs               # 用户反馈
│   │   ├── online_learning.rs        # 在线学习
│   │   ├── code_generator.rs         # 代码生成
│   │   └── deobfuscator.rs           # 去混淆
│   │
│   ├── network/                      # 网络模块
│   │   ├── http.rs                   # HTTP客户端
│   │   ├── cache.rs                  # 缓存
│   │   └── crawling.rs               # 网页爬取
│   │
│   ├── devtools/                     # 开发工具
│   │   ├── inspector.rs              # DOM检查器
│   │   ├── profiler.rs               # 性能分析
│   │   └── network_monitor.rs        # 网络监控
│   │
│   ├── plugins/                      # 插件系统
│   │   └── plugin_system.rs          # 插件支持
│   │
│   └── testing/                      # 测试工具
│       └── benchmark.rs              # 性能测试
```

#### 辅助Crates (分离的模块)

```
browerai-api-server/                  # REST API服务器
├── src/
│   ├── main.rs                       # 入口点
│   ├── handlers/                     # HTTP处理器
│   │   ├── parse.rs                  # 解析端点
│   │   ├── render.rs                 # 渲染端点
│   │   └── health.rs                 # 健康检查
│   └── config.rs                     # 配置加载

browerai-html-parser/                 # HTML解析 (可选)
browerai-css-parser/                  # CSS解析 (可选)
browerai-js-parser/                   # JS解析 (可选)
browerai-js-analyzer/                 # JS分析 (可选)
browerai-renderer/                    # 渲染 (可选)
browerai-ai-core/                     # AI核心 (可选)
browerai-learning/                    # 学习 (可选)

... (共27个crates)
```

---

### docs/ - 文档库

```
docs/
├── README.md                         # 文档主索引
│
├── guides/                           # 技术指南
│   ├── QUICK_START.md
│   ├── SETUP.md
│   ├── DEVELOPMENT.md
│   ├── TESTING.md
│   ├── DEPLOYMENT.md
│   ├── CI_CD.md
│   ├── TROUBLESHOOTING.md
│   ├── PYTHON_QUICK_START.md
│   └── README.md                     # 指南总览
│
├── api/                              # API文档
│   ├── README.md
│   ├── ENDPOINTS.md                  # 端点列表
│   ├── EXAMPLES.md                   # 使用示例
│   ├── SCHEMAS.md                    # 数据模型
│   └── SPECIFICATIONS.md
│
├── architecture/                     # 架构文档
│   ├── README.md
│   ├── OVERVIEW.md                   # 系统架构
│   ├── MODULES.md                    # 模块说明
│   ├── DESIGN_DECISIONS.md           # 设计决策
│   ├── ANALYSIS.md                   # 架构分析
│   └── INTEGRATION_DESIGN.md
│
├── development/                      # 开发规范
│   ├── README.md
│   ├── CODE_STYLE.md                 # 代码风格
│   ├── CONVENTIONS.md                # 开发约定
│   ├── BUILD_SYSTEM.md               # 构建系统
│   ├── TESTING_STRATEGY.md           # 测试策略
│   ├── PERFORMANCE.md                # 性能指南
│   └── PYTHON_SETUP.md               # Python设置
│
├── learning/                         # AI学习
│   ├── README.md
│   ├── MODEL_TRAINING.md             # 模型训练
│   ├── DATA_ANNOTATION.md            # 数据标注
│   ├── EVALUATION.md                 # 评估指标
│   └── DEOBFUSCATION.md              # 去混淆
│
├── integration/                      # 集成部署
│   ├── README.md
│   ├── DOCKER.md                     # Docker配置
│   ├── KUBERNETES.md                 # K8s部署
│   ├── CI_CD_WORKFLOWS.md            # CI/CD工作流
│   └── MONITORING.md                 # 监控配置
│
├── maintenance/                      # 维护相关
│   ├── README.md
│   ├── UPGRADE.md                    # 升级指南
│   ├── BACKUP.md                     # 备份策略
│   └── INCIDENT_RESPONSE.md          # 事件响应
│
├── references/                       # 参考资料
│   ├── GLOSSARY.md                   # 术语表
│   ├── DEPENDENCIES.md               # 依赖列表
│   ├── ENVIRONMENT.md                # 环境变量
│   ├── COMMANDS.md                   # 常用命令
│   └── FAQ.md                        # 常见问题
│
├── phases/                           # 项目历程
│   ├── README.md
│   ├── WEEK6_SUMMARY.md
│   ├── WEEK7_SUMMARY.md
│   └── WEEK8_SUMMARY.md
│
├── archived/                         # 历史存档
│   ├── PROJECT_FINAL_STATUS.md
│   ├── TEST_SUBMISSION_REPORT.md
│   ├── WEEK4_REPORTS/
│   ├── WEEK5_REPORTS/
│   ├── WEEK6_REPORTS/
│   ├── WEEK7_REPORTS/
│   ├── WEEK8_REPORTS/
│   ├── WEEK8/
│   │   ├── PHASE_A_EXECUTION.md
│   │   ├── PHASE_B_EXECUTION.md
│   │   ├── PHASE_C_EXECUTION.md
│   │   ├── PHASE_D_EXECUTION.md
│   │   └── PHASE_E_EXECUTION.md
│   └── temporary_reports/
│
├── PROJECT_STANDARDS.md              # 项目规范 (重要!)
├── CLEANUP_PLAN.md                   # 清理计划
└── CHANGELOG.md                      # 文档变更记录
```

---

### tests/ - 集成测试

```
tests/
├── api_integration_tests.rs          # API测试
├── parser_integration_tests.rs       # 解析器测试
├── renderer_integration_tests.rs     # 渲染器测试
├── e2e_website_tests.rs              # 端到端测试
└── step4_rust_integration_tests.rs   # 跨模块集成测试
```

---

### training/ - 模型训练

```
training/
├── scripts/
│   ├── train_html_parser.py          # HTML解析模型训练
│   ├── train_css_parser.py           # CSS解析模型训练
│   ├── train_js_detector.py          # JS检测模型
│   └── convert_to_onnx.py            # ONNX转换
│
├── data/                             # 训练数据
│   ├── augmentation.py               # 数据增强
│   └── preprocessing.py              # 数据预处理
│
├── utilities/
│   ├── metrics.py                    # 评估指标
│   └── visualization.py              # 可视化
│
├── QUICKSTART.md                     # Python快速开始
└── requirements.txt                  # Python依赖
```

---

### models/ - 预训练模型

```
models/
├── local/                            # 本地模型
│   ├── html_parser_v1.onnx
│   ├── css_parser_v1.onnx
│   └── js_detector_v1.onnx
│
├── config/
│   └── model_config.toml             # 模型注册表
│
└── README.md                         # 模型文档
```

---

### data/ - 数据集

```
data/
├── week6_samples/                    # 周6数据
├── week6_training_results/           # 训练结果
├── week6_features/                   # 特征数据
├── week6_samples_production/         # 生产数据
└── README.md                         # 数据说明
```

---

### scripts/ - 工具脚本

```
scripts/
├── github_deploy_prepare.sh          # GitHub部署脚本
├── comprehensive_test.sh             # 综合测试脚本
├── build_release.sh                  # 发布构建
├── docker_build.sh                   # Docker构建
└── README.md                         # 脚本文档
```

---

### config/ - 配置文件

```
config/
├── docker-compose.yml                # Docker Compose配置
├── docker-compose.monitoring.yml     # 监控栈
├── docker-compose.api.yml            # API配置
├── prometheus.yml                    # Prometheus配置
├── alertmanager.yml                  # 告警规则
└── README.md
```

---

### k8s/ - Kubernetes清单

```
k8s/
├── namespace.yaml                    # 命名空间
├── deployment.yaml                   # 部署配置
├── service.yaml                      # 服务配置
├── ingress.yaml                      # 入口配置
├── configmap.yaml                    # 配置映射
├── monitoring.yaml                   # 监控配置
└── README.md
```

---

### grafana/ - 监控面板

```
grafana/
├── provisioning/
│   ├── datasources/
│   │   └── prometheus.yaml
│   └── dashboards/
│       └── main.json                 # 主面板
└── README.md
```

---

## 🔄 依赖关系

### Crates依赖图

```
browerai-webclient
  └── API客户端 → browerai-api-server

browerai-api-server
  └── browerai (主库)
      ├── browerai-html-parser
      ├── browerai-css-parser
      ├── browerai-js-parser
      │   └── browerai-js-analyzer
      ├── browerai-renderer
      ├── browerai-ai-core
      ├── browerai-learning
      ├── browerai-network
      ├── browerai-cache
      ├── browerai-dom
      └── ...等27个crates
```

---

## 📦 构建输出

```
target/
├── debug/
│   ├── browerai           # Debug二进制
│   ├── deps/              # 所有依赖
│   └── ...
│
├── release/
│   ├── browerai           # Release二进制 (优化)
│   └── ...
│
└── doc/                   # 生成的文档
    └── browerai/
```

---

## 🐳 Docker镜像

```
Docker镜像:
├── browerai-api:latest              # API服务器
├── browerai-webclient:latest        # React前端
└── browerai-full:latest             # 完整堆栈
```

---

## 📋 关键文件

### 配置文件

| 文件 | 用途 |
|------|------|
| `Cargo.toml` | Rust项目配置 |
| `Cargo.lock` | 依赖锁定 |
| `package.json` | Node.js配置 |
| `tsconfig.json` | TypeScript配置 |
| `.github/workflows/` | CI/CD工作流 |
| `Dockerfile` | Docker镜像 |
| `.dockerignore` | Docker忽略列表 |
| `.gitignore` | Git忽略列表 |

### 重要文件

| 文件 | 用途 |
|------|------|
| `src/lib.rs` | Rust库入口 |
| `crates/browerai-api-server/src/main.rs` | API服务器入口 |
| `frontend/src/App.tsx` | React主组件 |
| `src/main.rs` | 主应用 (如果需要CLI) |

---

## 🎯 典型工作流

### 添加新特性

```
1. 在 crates/browerai/ 中修改源代码
2. 添加单元测试
3. 可能创建新的子crate
4. 运行 cargo test 验证
5. 更新相关文档
6. 创建PR进行审查
```

### 修复Bug

```
1. 找到错误所在的模块
2. 添加失败的测试用例
3. 修复代码使测试通过
4. 运行完整测试套件
5. 更新CHANGELOG.md
```

### 发布新版本

```
1. 更新CHANGELOG.md
2. 创建git标签 (v1.0.0)
3. 推送标签触发CI/CD
4. GitHub Actions自动构建和发布
5. 检查Docker Hub和GitHub Release
```

---

## 📚 相关文档

- [项目规范](docs/PROJECT_STANDARDS.md) - 开发规范
- [清理计划](docs/CLEANUP_PLAN.md) - 文档清理
- [开发指南](DEVELOPMENT_GUIDE.md) - 详细开发说明
- [快速开始](QUICK_START.md) - 5分钟入门
- [文档库](docs/README.md) - 完整文档索引

---

**项目结构遵循** [项目规范](docs/PROJECT_STANDARDS.md)  
**最后更新**: 2026-02-17

