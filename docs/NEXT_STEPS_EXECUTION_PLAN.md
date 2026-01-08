# 下一步执行计划 - 2026-01-07

## 📊 当前状态

✅ **Phase 3 完成**：混合 JS 编排器核心集成
- 150+ 个测试通过
- 三层清晰的架构（编排器 → 适配器 → 门面）
- 零循环依赖
- 详尽的文档

现在准备进入 **Phase 4：与实际管线集成**

## 🎯 可选的执行路径

### 路径 A：快速集成（推荐）
**时间**：1-2 天  
**目标**：让混合编排器在实际渲染/分析中工作

1. ✅ 准备工作（已完成）
   - 编写 Renderer 集成指南
   - 编写 Analyzer 集成指南
   - 编写集成路线图

2. **立即可做**：选择以下之一
   - Option A1：集成到 Renderer（脚本执行）
   - Option A2：集成到 Analyzer（混合分析）

### 路径 B：深度优化（可选）
**时间**：2-3 天  
**目标**：性能基准和缓存优化

1. 性能测试（三种策略对比）
2. 缓存层实现
3. 自适应策略选择

### 路径 C：完整验证（最全面）
**时间**：3-5 天  
**目标**：端到端集成和真实网页测试

1. 完成 A 路径
2. 完成 B 路径
3. 创建真实网页示例
4. 性能报告

## 🚀 立即可以执行的任务

### Task 1：在 Renderer 中集成 RenderingJsExecutor

**文件**：`crates/browerai-renderer-core/src/engine.rs`

**工作**：
```rust
// 添加字段
pub struct RenderEngine {
    layout_engine: LayoutEngine,
    paint_engine: PaintEngine,
    #[cfg(feature = "ai")]
    js_executor: Option<RenderingJsExecutor>,  // 新增
}

// 在 render() 中调用
pub fn render(&mut self, dom: &RcDom, styles: &[CssRule]) -> Result<RenderTree> {
    // ... 现有代码 ...
    
    // 新增：执行脚本
    self.execute_scripts(dom)?;
    
    // ... 继续 ...
}
```

**预期成果**：
- 能处理 HTML 中的 `<script>` 标签
- 脚本能修改 DOM
- 完整的错误处理

**文档参考**：[Renderer 集成指南](./docs/RENDERER_INTEGRATION_GUIDE.md)

---

### Task 2：在 Analyzer 中集成混合分析

**文件**：`crates/browerai-js-analyzer/src/hybrid_analyzer.rs` (新建)

**工作**：
```rust
pub struct HybridJsAnalyzer {
    scope_analyzer: ScopeAnalyzer,
    dataflow_analyzer: DataflowAnalyzer,
    callgraph_analyzer: CallGraphAnalyzer,
    ast_provider: AnalysisJsAstProvider,
    #[cfg(feature = "ai")]
    orchestrator: Option<HybridJsOrchestrator>,
}

impl HybridJsAnalyzer {
    pub fn analyze(&mut self, source: &str) -> Result<HybridAnalysisResult> {
        // 静态分析
        let static_result = self.analyze_static(source)?;
        
        // AST 特征检测
        let ast_info = self.ast_provider.parse_and_analyze(source)?;
        
        // 动态分析（可选）
        let dynamic_result = self.analyze_dynamic(source)?;
        
        // 融合结果
        Ok(self.combine_results(static_result, ast_info, dynamic_result)?)
    }
}
```

**预期成果**：
- 精准的 AST 信息
- 框架自动检测
- 融合的分析结果

**文档参考**：[Analyzer 集成指南](./docs/ANALYZER_INTEGRATION_GUIDE.md)

---

## 📋 推荐的执行顺序

### 第 1 天：Task 1（Renderer 集成）
- 修改 `RenderEngine` 添加 `RenderingJsExecutor`
- 实现 `execute_scripts()` 方法
- 添加脚本提取逻辑
- 编写单元测试（5-10 个）
- **预期**：3-4 小时

### 第 2 天：Task 2（Analyzer 集成）
- 创建 `hybrid_analyzer.rs`
- 实现 `HybridJsAnalyzer` 结构
- 实现混合分析逻辑
- 框架检测功能
- 编写测试（5-10 个）
- **预期**：4-5 小时

### 第 3 天：测试和优化
- 端到端集成测试
- 性能基准测试
- 文档和示例
- **预期**：3-4 小时

## 📊 预期工作量

| 任务 | 代码行数 | 测试行数 | 时间 |
|------|---------|---------|------|
| Task 1 | 400-500 | 200+ | 3-4h |
| Task 2 | 600-700 | 250+ | 4-5h |
| 集成测试 | 300-400 | 200+ | 2-3h |
| **总计** | **1,300-1,600** | **650+** | **9-12h** |

## ✅ 执行检查清单

### 开始前
- [ ] 确认所有 Phase 1 测试通过
- [ ] 查看集成指南理解架构
- [ ] 准备开发环境

### Task 1 完成
- [ ] RenderEngine 有 RenderingJsExecutor
- [ ] 脚本能被正确执行
- [ ] DOM 修改被应用
- [ ] 所有新代码有测试
- [ ] 编译成功，无警告

### Task 2 完成
- [ ] HybridJsAnalyzer 创建完成
- [ ] 静态分析集成
- [ ] 动态分析可选
- [ ] 框架检测工作
- [ ] 所有新代码有测试
- [ ] 编译成功，无警告

### 集成完成
- [ ] 端到端测试通过
- [ ] 性能满足基准
- [ ] 文档完整
- [ ] 示例可运行
- [ ] 所有测试通过

## 🔧 必要的依赖和工具

```bash
# 确保环境设置正确
cargo build --features ai,v8

# 运行现有测试
cargo test -p browerai-ai-integration
cargo test -p browerai-renderer-core
cargo test -p browerai-js-analyzer

# 检查代码质量
cargo clippy --all-targets --all-features
cargo fmt --check
```

## 📚 参考资源

### 核心文档
- [集成路线图](./INTEGRATION_ROADMAP.md)
- [Renderer 集成指南](./RENDERER_INTEGRATION_GUIDE.md)
- [Analyzer 集成指南](./ANALYZER_INTEGRATION_GUIDE.md)
- [快速参考](./HYBRID_JS_QUICK_REFERENCE.md)

### 源代码参考
- [HybridJsOrchestrator](../crates/browerai-ai-integration/src/js_orchestrator.rs)
- [RenderingJsExecutor](../crates/browerai-renderer-core/src/js_executor.rs)
- [AnalysisJsAstProvider](../crates/browerai-js-analyzer/src/ast_provider.rs)
- [UnifiedJsInterface](../crates/browerai/src/unified_js.rs)

### 测试参考
- [Orchestrator 测试](../crates/browerai-ai-integration/tests/js_orchestrator_tests.rs)
- [Executor 内嵌测试](../crates/browerai-renderer-core/src/js_executor.rs)
- [Provider 内嵌测试](../crates/browerai-js-analyzer/src/ast_provider.rs)

## 🎓 学习路径

如果还不熟悉混合编排器，建议按以下顺序了解：

1. **5 分钟**：快速参考的"快速开始"部分
2. **15 分钟**：集成路线图的架构部分
3. **30 分钟**：对应集成指南（Renderer 或 Analyzer）
4. **60 分钟**：查看源代码实现

## 🚨 常见陷阱和解决方案

### 陷阱 1：编译错误
**症状**：未找到 RenderingJsExecutor  
**原因**：没有启用 `ai` 特性  
**解决**：`cargo build --features ai`

### 陷阱 2：特性门禁混淆
**症状**：cfg(feature = "ai") 下的代码没有编译  
**原因**：条件编译错误  
**解决**：查看 js_executor.rs 中的模式

### 陷阱 3：测试失败
**症状**：新测试编译错误  
**原因**：缺少依赖导入或 mock  
**解决**：参考现有测试的模式

## 💡 建议和最佳实践

1. **增量开发**：先完成 Task 1，验证成功后再做 Task 2
2. **持续测试**：在每个小功能完成后立即运行 `cargo test`
3. **文档同步**：功能完成时同时更新文档和代码注释
4. **性能监控**：在集成过程中记录性能数据
5. **代码复用**：参考现有代码的模式，保持一致性

## 🎯 最终目标

在本执行计划完成后，BrowerAI 将具有：

✅ **完整的 JS 处理能力**
- 渲染中的脚本执行
- 分析中的混合静态/动态分析
- 自动框架检测

✅ **生产就绪的实现**
- 完整的测试覆盖
- 详尽的文档
- 性能优化

✅ **清晰的集成模式**
- 可复用的架构设计
- 明确的接口契约
- 环境变量控制

---

**准备好开始了吗？选择 Task 1 或 Task 2，开始集成吧！** 🚀
