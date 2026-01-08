# 混合 JS 编排器快速参考

## 快速开始

### 最简单的用法

```rust
use browerai::prelude::*;

let mut ujif = UnifiedJsInterface::new();
ujif.execute_for_render("console.log('Hello')")?;
```

## 三个核心接口

### 1. UnifiedJsInterface（推荐）

```rust
pub struct UnifiedJsInterface { }

impl UnifiedJsInterface {
    pub fn new() -> Self
    pub fn execute_for_render(&mut self, js: &str) -> Result<JsExecResult>
    pub fn parse_for_analysis(&mut self, js: &str) -> Result<JsParseResult>
    pub fn quick_validate(&mut self, js: &str) -> Result<bool>
}
```

**何时使用**：大多数情况下的首选
**特性需求**：无（默认可用）

### 2. RenderingJsExecutor（特定于渲染）

```rust
pub struct RenderingJsExecutor { }

impl RenderingJsExecutor {
    pub fn new() -> Self
    pub fn execute(&mut self, js: &str) -> Result<String>
    pub fn validate(&mut self, js: &str) -> Result<bool>
}
```

**何时使用**：需要细粒度的渲染 JS 控制
**环境变量**：`BROWERAI_RENDER_JS_POLICY`
**特性需求**：`ai`（可选）

### 3. AnalysisJsAstProvider（特定于分析）

```rust
pub struct AnalysisJsAstProvider { }

impl AnalysisJsAstProvider {
    pub fn new() -> Self
    pub fn parse_and_analyze(&self, js: &str) -> Result<JsAnalysisResult>
    pub fn validate(&self, js: &str) -> Result<bool>
}
```

**何时使用**：需要 JS 代码结构信息（模块类型等）
**环境变量**：`BROWERAI_ANALYSIS_JS_POLICY`
**特性需求**：无

## 策略速查表

| 策略 | AST | 执行 | 用途 |
|------|-----|------|------|
| `Performance` | SWC | V8 | 最高性能 |
| `Secure` | Boa | Boa | 安全沙箱 |
| `Balanced` | SWC→Boa | V8→Boa | 推荐默认 |

设置方式：
```bash
export BROWERAI_JS_POLICY=performance          # 全局
export BROWERAI_RENDER_JS_POLICY=secure        # 渲染专用
export BROWERAI_ANALYSIS_JS_POLICY=balanced    # 分析专用
```

## 常见模式

### 模式 1：渲染动态内容

```rust
let mut interface = UnifiedJsInterface::new();
let js = r#"
  document.body.innerHTML = 
    '<h1>Dynamic Content</h1>';
"#;
interface.execute_for_render(js)?;
```

### 模式 2：检测模块类型

```rust
let mut interface = UnifiedJsInterface::new();
let result = interface.parse_for_analysis(
    "import { x } from 'module';"
)?;
assert!(result.is_module);
```

### 模式 3：快速验证

```rust
let mut interface = UnifiedJsInterface::new();
let is_valid = interface.quick_validate(
    "const x = [1, 2, 3];"
)?;
assert!(is_valid);
```

### 模式 4：安全执行不可信代码

```rust
use browerai_ai_integration::{HybridJsOrchestrator, OrchestrationPolicy};

let mut orchestrator = HybridJsOrchestrator::with_policy(
    OrchestrationPolicy::Secure
);
let result = orchestrator.execute(untrusted_code)?;
```

## 编译选项

```bash
# 最小化编译（启发式实现）
cargo build

# 启用完整 AI 功能（三引擎）
cargo build --features ai

# 启用 V8 引擎（性能）
cargo build --features ai,v8

# 启用所有功能
cargo build --features ai,v8,ml,metrics
```

## 特性支持矩阵

| 特性 | V8 | SWC | Boa | 启发式 |
|------|----|----|-----|--------|
| 无特性 | ❌ | ❌ | ❌ | ✅ |
| `ai` | ✅ | ✅ | ✅ | ✅ |

## 错误处理

```rust
match interface.execute_for_render(js) {
    Ok(result) => println!("输出: {}", result.output),
    Err(e) => eprintln!("失败: {}", e),
}

// 或用 ? 传播
let result = interface.execute_for_render(js)?;
```

## 性能提示

1. **重用实例**：保留 orchestrator 以避免重复初始化
   ```rust
   let mut interface = UnifiedJsInterface::new();
   for code in codes {
       interface.execute_for_render(&code)?;
   }
   ```

2. **使用 quick_validate**：简单检查用 validate 比 parse 快
   ```rust
   interface.quick_validate(code)?;  // 快
   interface.parse_for_analysis(code)?;  // 慢
   ```

3. **批量处理**：在循环中保留编排器
   ```rust
   // ✓ 好
   let mut orch = HybridJsOrchestrator::new();
   for code in codes { orch.parse(&code)?; }
   
   // ✗ 差
   for code in codes { 
       HybridJsOrchestrator::new().parse(&code)?;
   }
   ```

## 导入速查

```rust
// 统一接口（推荐）
use browerai::prelude::UnifiedJsInterface;

// 渲染特定
use browerai::RenderingJsExecutor;

// 分析特定
use browerai_js_analyzer::AnalysisJsAstProvider;

// 低级编排器（高级用户）
use browerai_ai_integration::{
    HybridJsOrchestrator,
    OrchestrationPolicy,
    UnifiedAst,
    AstEngine,
    SourceKind,
};
```

## 故障排查

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| V8 不可用 | 总是用 Boa | `cargo build --features ai,v8` |
| 模块检测错误 | 脚本识别为模块 | 启用 `swc-full` 特性 |
| 内存泄漏 | OOM 错误 | 不在循环中创建新编排器 |

## 调试技巧

启用调试日志：
```bash
RUST_LOG=debug cargo run
```

查看引擎选择：
```bash
RUST_LOG=browerai_ai_integration=debug cargo run
```

## 参考文档

- [完整集成指南](./HYBRID_JS_ORCHESTRATION_INTEGRATION.md)
- [项目改进报告](./PHASE3_HYBRID_JS_INTEGRATION_REPORT.md)
- [源代码](../crates/)
