# 混合 JS 编排器集成路线图

## 总体目标

将完成的混合 JS 编排器（V8 + SWC + Boa）集成到 BrowerAI 的核心渲染和分析管线中，实现完整的 JS 处理能力。

## 集成进度

### ✅ Phase 1：核心编排器（已完成）

**成果**：
- ✅ HybridJsOrchestrator 核心实现
- ✅ RenderingJsExecutor 包装器
- ✅ AnalysisJsAstProvider 适配器
- ✅ UnifiedJsInterface 顶层接口
- ✅ 150+ 个测试全部通过
- ✅ 零循环依赖

**文档**：
- ✅ HYBRID_JS_ORCHESTRATION_INTEGRATION.md
- ✅ HYBRID_JS_QUICK_REFERENCE.md
- ✅ PHASE3_EXECUTION_SUMMARY.md

---

### 📍 Phase 2：Renderer 管线集成（进行中）

**目标**：在 RenderEngine 中支持 JS 脚本执行

**任务清单**：
- [ ] **Task 2.1**：为 RenderEngine 添加 RenderingJsExecutor
  - 文件：`crates/browerai-renderer-core/src/engine.rs`
  - 工作量：100-150 行代码
  - 难度：低
  
- [ ] **Task 2.2**：实现脚本提取和执行
  - 实现 `extract_scripts()` 方法
  - 实现 `execute_scripts()` 方法
  - 工作量：150-200 行代码
  
- [ ] **Task 2.3**：DOM 修改跟踪
  - 设计 DomOperation 枚举
  - 实现修改捕获和应用
  - 工作量：200-250 行代码
  
- [ ] **Task 2.4**：错误处理和恢复
  - 脚本失败时的降级处理
  - 部分失败的继续渲染
  - 工作量：100 行代码
  
- [ ] **Task 2.5**：单元和集成测试
  - 测试基本 JS 执行
  - 测试脚本顺序
  - 测试 DOM 修改
  - 工作量：200+ 行代码

**预期成果**：
- 能够渲染包含 `<script>` 标签的完整 HTML
- 支持脚本对 DOM 的动态修改
- 完整的错误处理

**参考文档**：
- [Renderer 集成指南](./RENDERER_INTEGRATION_GUIDE.md)

---

### 📍 Phase 3：Analyzer 管线集成（进行中）

**目标**：在 JsAnalyzer 中支持混合静态/动态分析

**任务清单**：
- [ ] **Task 3.1**：创建 HybridJsAnalyzer
  - 文件：`crates/browerai-js-analyzer/src/hybrid_analyzer.rs`
  - 整合静态和动态分析
  - 工作量：300-400 行代码
  - 难度：中等
  
- [ ] **Task 3.2**：框架检测功能
  - React/Vue/Angular 检测
  - 依赖关系识别
  - 工作量：150-200 行代码
  
- [ ] **Task 3.3**：运行时属性提取
  - 执行代码并捕获全局对象
  - 构建属性映射
  - 工作量：100-150 行代码
  
- [ ] **Task 3.4**：分析缓存
  - 结果缓存实现
  - 增量分析支持
  - 工作量：150-200 行代码
  
- [ ] **Task 3.5**：测试覆盖
  - 混合分析测试
  - 框架检测测试
  - 性能基准测试
  - 工作量：200+ 行代码

**预期成果**：
- 精准的 AST 和运行时信息融合
- 自动框架检测
- 缓存优化

**参考文档**：
- [Analyzer 集成指南](./ANALYZER_INTEGRATION_GUIDE.md)

---

### ⏳ Phase 4：端到端测试和优化（计划中）

**目标**：验证集成的完整功能

**任务**：
- [ ] Task 4.1：创建端到端示例
  - React 应用渲染示例
  - Vue 应用渲染示例
  - 库代码分析示例
  
- [ ] Task 4.2：性能基准测试
  - 对比三种策略的性能
  - 测试缓存效果
  - 大规模代码处理
  
- [ ] Task 4.3：文档补充
  - 更新 README
  - 集成示例
  - 性能指南
  
- [ ] Task 4.4：优化和调优
  - 缓存策略优化
  - 超时处理完善
  - 错误恢复增强

---

## 时间线估计

| Phase | 任务数 | 代码行数 | 预计时间 |
|-------|--------|---------|---------|
| 1 (完成) | 5 | 1,500+ | ✅ 完成 |
| 2 (进行中) | 5 | 800-1,000 | 1-2 天 |
| 3 (进行中) | 5 | 1,000-1,200 | 2-3 天 |
| 4 (计划中) | 4 | 600-800 | 2 天 |
| **总计** | **19** | **3,900-4,500** | **~7-8 天** |

## 集成检查清单

### Phase 2：Renderer 集成

- [ ] RenderEngine 包含 RenderingJsExecutor 实例
- [ ] 脚本能被正确提取和执行
- [ ] DOM 修改被应用到渲染结果
- [ ] 脚本执行错误不会导致渲染失败
- [ ] 所有新代码有单元测试
- [ ] 集成测试通过（含真实 HTML）
- [ ] 文档已更新

### Phase 3：Analyzer 集成

- [ ] HybridJsAnalyzer 实现完成
- [ ] 框架检测功能正常工作
- [ ] 运行时属性提取有效
- [ ] 缓存系统工作正常
- [ ] 性能满足基准要求
- [ ] 所有新代码有单元测试
- [ ] 集成测试覆盖多个框架
- [ ] 文档完整清晰

### Phase 4：端到端

- [ ] 端到端示例可运行
- [ ] 性能基准数据完整
- [ ] README 和文档已更新
- [ ] 所有测试通过
- [ ] 性能达到预期

## 关键指标

### 代码质量
- 测试覆盖率：> 80%
- 编译警告：0
- Clippy 警告：0

### 性能目标
- 简单 JS 执行：< 10ms
- 大型代码分析：< 500ms
- 缓存命中加速：5-10 倍

### 功能完整性
- 支持所有三种策略：✅
- 支持主流框架检测：React, Vue, Angular
- 错误恢复机制：完整
- 文档完善度：详尽

## 相关资源

### 文档
- [Hybrid JS Orchestration Integration](./HYBRID_JS_ORCHESTRATION_INTEGRATION.md)
- [Renderer Integration Guide](./RENDERER_INTEGRATION_GUIDE.md)
- [Analyzer Integration Guide](./ANALYZER_INTEGRATION_GUIDE.md)
- [Quick Reference](./HYBRID_JS_QUICK_REFERENCE.md)

### 源代码
- [js_orchestrator.rs](../crates/browerai-ai-integration/src/js_orchestrator.rs)
- [js_executor.rs](../crates/browerai-renderer-core/src/js_executor.rs)
- [ast_provider.rs](../crates/browerai-js-analyzer/src/ast_provider.rs)
- [unified_js.rs](../crates/browerai/src/unified_js.rs)

### 测试
- [js_orchestrator_tests.rs](../crates/browerai-ai-integration/tests/js_orchestrator_tests.rs)
- [Renderer tests](../crates/browerai-renderer-core/src/js_executor.rs#tests)
- [Analyzer tests](../crates/browerai-js-analyzer/src/ast_provider.rs#tests)

## 下一步行动

### 立即（今天）
- [ ] 审视本路线图
- [ ] 选择 Phase 2 或 Phase 3 开始
- [ ] 创建 GitHub issue 或任务跟踪

### 短期（本周）
- [ ] 完成 Phase 2 或 Phase 3
- [ ] 编写集成测试
- [ ] 性能基准测试

### 中期（1-2 周）
- [ ] 完成所有集成
- [ ] 端到端测试
- [ ] 性能优化
- [ ] 文档完善

---

**更新时间**：2026-01-07  
**维护者**：BrowerAI 开发团队  
**状态**：🚀 活跃开发中
