# AI-Core 文档索引

## 📚 文档导航

### 快速开始

| 文档 | 用途 | 阅读时间 |
|------|------|---------|
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | 常用API和示例速查 | 10分钟 |
| [README.md](./README.md) | 包概览和特性列表 | 5分钟 |

### 学习资源

| 文档 | 内容 | 用途 | 读者 |
|------|------|------|------|
| [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) | 详细使用指南和最佳实践 | 学习如何使用 | 开发者 |
| [ENHANCEMENT_PLAN.md](./ENHANCEMENT_PLAN.md) | 架构设计和规划 | 理解整体设计 | 架构师 |

### 参考资料

| 文档 | 内容 | 用途 |
|------|------|------|
| [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) | 详细的实现总结 | 了解具体实现 |
| [COMPLETION_REPORT.md](./COMPLETION_REPORT.md) | 完成情况和质量指标 | 验收和评估 |

### 代码文档

| 模块 | 说明 | 关键类型 |
|------|------|---------|
| `src/model_provider.rs` | 提供者trait和注册表 | `ModelProvider`, `ModelProviderRegistry` |
| `src/onnx_provider.rs` | ONNX运行时实现 | `OnnxModelProvider` |
| `src/advanced_metrics.rs` | 性能指标收集 | `MetricsAggregator`, `InferenceMetrics` |
| `src/resilience.rs` | 高可用性模式 | `CircuitBreaker`, `RetryPolicy` |
| `src/inference.rs` | 推理引擎核心 | `InferenceEngine` |
| `src/runtime.rs` | 统一运行时 | `AiRuntime` |
| `src/config.rs` | 配置管理 | `AiConfig`, `FallbackTracker` |
| `src/gpu_support.rs` | GPU加速 | `GpuConfig`, `GpuProvider` |

### 测试

| 文件 | 内容 | 覆盖 |
|------|------|------|
| `tests/integration_tests.rs` | 15+ 集成测试 | 核心流程 |
| `src/lib.rs` | 各模块单元测试 | 27+ 个测试 |

---

## 🎯 按用途查找

### 我想...

#### 快速上手
1. 阅读 [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) 的"常见用法模式"
2. 查看 [README.md](./README.md) 的"快速开始"部分
3. 运行 `cargo test --test integration_tests` 查看例子

#### 理解架构
1. 读 [ENHANCEMENT_PLAN.md](./ENHANCEMENT_PLAN.md) 的"全局增强规划"
2. 学 [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) 的"架构概览"
3. 研究 `src/model_provider.rs` 中的trait定义

#### 实现新功能
1. [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) - "扩展 AI-Core"部分
2. [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - "扩展点"部分
3. 查看 `src/onnx_provider.rs` 作为实现参考

#### 解决问题
1. [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) - "故障排除"部分
2. [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - "故障排除"表
3. 运行 `RUST_LOG=debug cargo test` 查看详细日志

#### 评估质量
1. 读 [COMPLETION_REPORT.md](./COMPLETION_REPORT.md) - "质量指标"
2. 查 [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) - "代码覆盖率"
3. 运行测试验证: `cargo test -p browerai-ai-core`

#### 优化性能
1. [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) - "性能优化指南"
2. [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - "性能建议"
3. 分析 `MetricsSnapshot` 数据

---

## 📖 阅读顺序建议

### 对于开发者

```
1. README.md (5分钟)
   └─ 了解包的功能
   
2. QUICK_REFERENCE.md (10分钟)
   └─ 学习基本API
   
3. DEVELOPER_GUIDE.md (30分钟)
   └─ 深入理解和最佳实践
   
4. src/*.rs 代码 (按需)
   └─ 研究实现细节
   
5. tests/integration_tests.rs (15分钟)
   └─ 看完整示例
```

### 对于架构师

```
1. README.md (5分钟)
   └─ 快速概览
   
2. ENHANCEMENT_PLAN.md (20分钟)
   └─ 理解设计决策
   
3. COMPLETION_REPORT.md (15分钟)
   └─ 了解实现情况
   
4. src/model_provider.rs (10分钟)
   └─ 研究核心抽象
```

### 对于测试/QA

```
1. README.md (5分钟)
   └─ 了解功能
   
2. COMPLETION_REPORT.md (15分钟)
   └─ 查看测试覆盖
   
3. tests/integration_tests.rs (20分钟)
   └─ 理解测试策略
   
4. QUICK_REFERENCE.md (10分钟)
   └─ 学习测试命令
```

### 对于用户

```
1. README.md (5分钟)
   └─ 了解功能
   
2. QUICK_REFERENCE.md (10分钟)
   └─ 学习基本用法
   
3. DEVELOPER_GUIDE.md - 快速开始部分 (15分钟)
   └─ 获得实际例子
```

---

## 🔗 相关资源

### 项目级文档

- [../../docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) - 整个BrowerAI的架构
- [../../README.md](../../README.md) - 项目总体说明

### 其他包

- [../browerai-core/](../browerai-core/) - 核心包
- [../browerai-html-parser/](../browerai-html-parser/) - HTML解析
- [../browerai-js-analyzer/](../browerai-js-analyzer/) - JS分析

### 外部资源

- [ONNX Runtime 文档](https://onnxruntime.ai/)
- [Hugging Face Candle](https://github.com/huggingface/candle)
- [Rust 官方文档](https://doc.rust-lang.org/)

---

## 📊 文档统计

| 文档 | 行数 | 字数 | 复杂度 |
|------|------|------|--------|
| QUICK_REFERENCE.md | 300+ | ~3K | ⭐ |
| README.md | 200+ | ~2K | ⭐⭐ |
| DEVELOPER_GUIDE.md | 400+ | ~4K | ⭐⭐⭐ |
| ENHANCEMENT_PLAN.md | 400+ | ~4K | ⭐⭐⭐ |
| IMPLEMENTATION_SUMMARY.md | 500+ | ~5K | ⭐⭐⭐⭐ |
| COMPLETION_REPORT.md | 500+ | ~5K | ⭐⭐⭐⭐ |
| **总计** | **2300+** | **23K+** | |

---

## ✅ 文档检查清单

- [x] 快速开始指南完整
- [x] API文档详细
- [x] 架构清晰
- [x] 示例可运行
- [x] 故障排除全面
- [x] 扩展点明确
- [x] 性能指南完整
- [x] 最佳实践列出
- [x] 相互引用正确
- [x] 索引易于导航

---

## 🎓 学习路径

### 新手路径 (1小时)

```
入门
  ├─ README.md (5分钟)
  ├─ QUICK_REFERENCE.md (10分钟)
  └─ 运行第一个例子 (10分钟)
      └─ cargo build && cargo test
          
理解
  ├─ DEVELOPER_GUIDE.md - 快速开始 (15分钟)
  ├─ 阅读一个模块代码 (15分钟)
  └─ 运行集成测试 (10分钟)
```

### 中级路径 (3小时)

```
掌握设计
  ├─ ENHANCEMENT_PLAN.md (30分钟)
  ├─ DEVELOPER_GUIDE.md 完整 (45分钟)
  └─ 研究核心模块 (30分钟)
      
实践
  ├─ 运行所有测试 (10分钟)
  ├─ 尝试自定义提供者 (45分钟)
  └─ 优化一个场景 (20分钟)
```

### 高级路径 (6小时)

```
深入研究
  ├─ COMPLETION_REPORT.md (30分钟)
  ├─ 全部代码审查 (2小时)
  ├─ 架构优化分析 (1小时)
  └─ 性能基准测试 (1小时)
      
贡献
  ├─ 实现新特性 (1小时)
  ├─ 添加测试 (30分钟)
  └─ 更新文档 (30分钟)
```

---

## 🚀 快速命令参考

```bash
# 构建
cargo build -p browerai-ai-core
cargo build -p browerai-ai-core --features onnx

# 测试
cargo test -p browerai-ai-core
cargo test --test integration_tests
cargo test test_circuit_breaker_resilience -- --nocapture

# 文档
cargo doc -p browerai-ai-core --open

# 检查
cargo check -p browerai-ai-core
cargo clippy -p browerai-ai-core

# 性能
cargo bench -p browerai-ai-core
```

---

## 📞 获取帮助

1. **API文档** → 查看源代码注释
2. **使用问题** → 读 DEVELOPER_GUIDE.md
3. **性能问题** → 查 QUICK_REFERENCE.md 的性能建议
4. **设计问题** → 阅读 ENHANCEMENT_PLAN.md
5. **bug报告** → 运行 `RUST_LOG=debug cargo test`

---

**最后更新**: 2026-01-07  
**版本**: 0.2.0  
**状态**: ✅ 完整
