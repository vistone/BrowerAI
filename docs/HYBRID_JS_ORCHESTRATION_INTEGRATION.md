# 混合 JS 编排器集成指南

## 概述

BrowerAI 项目实现了一个混合 JS 编排器（HybridJsOrchestrator），可以根据策略动态选择 V8、SWC、Boa 三个引擎的最优组合。这个指南展示如何在渲染和分析管线中使用这个编排器。

## 架构概览

### 模块结构

```
browerai-ai-integration/
  └─ src/
     └─ js_orchestrator.rs       # 核心混合编排器

browerai-renderer-core/
  └─ src/
     └─ js_executor.rs           # 渲染管线中的 JS 执行器

browerai-js-analyzer/
  └─ src/
     └─ ast_provider.rs          # 分析管线中的 AST 提供器

browerai/
  └─ src/
     └─ unified_js.rs            # 统一接口（顶层 facade）
```

### 依赖关系

```
ai-integration ─→ js-analyzer (获取 SwcAstExtractor)
                           ↓
renderer-core (使用 ai-integration，ai 特性可选)
js-analyzer (纯启发式，避免循环依赖)
       ↓
    unified_js (组合两个适配器，ai 特性可选)
```

**关键设计**：为了避免循环依赖，js-analyzer 的 ast_provider 只提供启发式实现。真正的混合编排器集成在上层（renderer-core、browerai）进行。

## 三层集成

### 第一层：核心编排器（ai-integration）

```rust
use browerai_ai_integration::{HybridJsOrchestrator, OrchestrationPolicy};

let mut orchestrator = HybridJsOrchestrator::with_policy(OrchestrationPolicy::Balanced);

// 解析 AST（自动选择 SWC > Boa）
let ast = orchestrator.parse("import x from 'y';")?;
println!("模块: {}", matches!(ast.source_kind, SourceKind::Module));
println!("使用的引擎: {:?}", ast.engine);

// 执行代码（自动选择 V8 > Boa）
let output = orchestrator.execute("console.log('Hello')")?;

// 验证语法
let is_valid = orchestrator.validate("const x = 1;")?;
```

**策略选择**：
- `Performance`：优先 V8 + SWC（快速但需要 V8 模块）
- `Secure`：优先 Boa（安全的纯 Rust 沙箱）
- `Balanced`：智能选择，默认推荐

### 第二层：适配器（renderer 和 analyzer）

#### 在渲染管线中使用

```rust
use browerai::RenderingJsExecutor;

// 创建执行器（如果启用 ai 特性，使用混合编排器；否则回退）
let mut executor = RenderingJsExecutor::new();

// 执行渲染相关 JS
let result = executor.execute("
  const container = document.getElementById('app');
  container.innerHTML = '<h1>Hello World</h1>';
")?;

println!("执行输出: {}", result);
```

环境变量控制：`BROWERAI_RENDER_JS_POLICY`
- 默认: `balanced`
- 可选: `performance`, `secure`

#### 在分析管线中使用

```rust
use browerai_js_analyzer::AnalysisJsAstProvider;

// 创建 AST 提供器（纯启发式实现）
let provider = AnalysisJsAstProvider::new();

// 解析并分析 JS
let result = provider.parse_and_analyze("
  import React from 'react';
  export const App = () => <div>Hello</div>;
")?;

println!("是否模块: {}", result.is_module);
println!("是否 TS/JSX: {}", result.is_typescript_jsx);
```

### 第三层：统一接口（顶层 facade）

```rust
use browerai::prelude::UnifiedJsInterface;

let mut interface = UnifiedJsInterface::new();

// 执行渲染 JS
let render_result = interface.execute_for_render("
  document.querySelector('.container').textContent = 'Updated!';
")?;

// 分析模块 JS
let analysis_result = interface.parse_for_analysis("
  import { useState } from 'react';
  export function Counter() { ... }
")?;

// 快速验证语法
let is_valid = interface.quick_validate("const x = 1 + 2;")?;
```

## 策略选择指南

### 何时使用 Performance（性能优先）

- 处理大量 JS 代码需要快速解析
- 分析性能敏感的应用
- V8 引擎可用且已启用

```rust
use browerai_ai_integration::OrchestrationPolicy;

let orchestrator = HybridJsOrchestrator::with_policy(
    OrchestrationPolicy::Performance
);
// 会优先选择 V8（执行）和 SWC（解析）
```

### 何时使用 Secure（安全优先）

- 处理不可信的第三方脚本
- 需要完全 Rust 沙箱隔离
- 不需要高性能

```rust
let orchestrator = HybridJsOrchestrator::with_policy(
    OrchestrationPolicy::Secure
);
// 会优先选择 Boa 引擎的纯 Rust 实现
```

### 何时使用 Balanced（平衡）

- 默认选择，适合大多数场景
- 自动根据代码类型选择最优引擎
- 性能和安全性的好衡

```rust
let orchestrator = HybridJsOrchestrator::with_policy(
    OrchestrationPolicy::Balanced
);
// 智能选择：模块代码 → SWC，脚本 → Boa，执行 → V8(可用) → Boa
```

## 环境变量控制

### 全局策略

- `BROWERAI_JS_POLICY`: 默认编排策略
  - 值: `performance` / `secure` / `balanced`
  - 默认: `balanced`

### 管线特定策略

- `BROWERAI_RENDER_JS_POLICY`: 渲染执行策略（默认 balanced）
- `BROWERAI_ANALYSIS_JS_POLICY`: 分析解析策略（默认 balanced）

```bash
# 使用性能优先策略处理渲染
export BROWERAI_RENDER_JS_POLICY=performance

# 使用安全策略处理分析
export BROWERAI_ANALYSIS_JS_POLICY=secure

cargo run my_app
```

## 特性标志

### 启用完整 AI 功能

```bash
cargo build --features ai
```

此时所有三个引擎都可用（需要 V8 和 ONNX 运行时依赖）。

### 禁用 AI 功能（默认）

```bash
cargo build
```

编排器自动回退到启发式方法，只使用 Boa（无额外依赖）。

## 实际使用场景

### 场景 1：渲染动态网页

```rust
use browerai::prelude::*;

async fn render_page(html_content: &str) -> anyhow::Result<String> {
    // 创建统一接口
    let mut unified = UnifiedJsInterface::new();
    
    // 在模板中执行 JS 以生成最终 HTML
    let render_js = r#"
        const { renderToString } = require('react-dom/server');
        const App = require('./app').default;
        renderToString(<App />);
    "#;
    
    let result = unified.execute_for_render(render_js)?;
    Ok(result.output)
}
```

### 场景 2：分析第三方库

```rust
fn analyze_library(source_code: &str) -> anyhow::Result<LibraryAnalysis> {
    let mut unified = UnifiedJsInterface::new();
    
    // 确定库的类型（ESM、UMD、CommonJS 等）
    let analysis = unified.parse_for_analysis(source_code)?;
    
    Ok(LibraryAnalysis {
        is_module: analysis.is_module,
        has_typescript: analysis.is_typescript_jsx,
        statement_count: analysis.statement_count,
    })
}
```

### 场景 3：安全的脚本沙箱

```rust
use browerai_ai_integration::OrchestrationPolicy;

fn execute_untrusted_script(script: &str) -> anyhow::Result<String> {
    // 使用安全策略执行不可信脚本
    let mut orchestrator = HybridJsOrchestrator::with_policy(
        OrchestrationPolicy::Secure
    );
    
    orchestrator.execute(script)
}
```

## 性能考虑

### 模型加载

- 第一次使用 V8/SWC 时，模型会被加载到内存
- 后续调用重用相同的引擎实例
- Boa 因为是纯 Rust 编译，启动时间最短

### 缓存策略

```rust
// 对于频繁调用的代码，保留 orchestrator 实例
let mut orchestrator = HybridJsOrchestrator::new();

for code in code_batch {
    let result = orchestrator.parse(&code)?;
    // 重用同一实例，避免重复初始化
}
```

### 并发执行

- 编排器使用 Arc<Session> 实现线程安全
- 支持 tokio 异步执行

```rust
use tokio::task;

async fn parallel_analysis(scripts: Vec<&str>) -> anyhow::Result<Vec<_>> {
    let mut handles = vec![];
    
    for script in scripts {
        let handle = task::spawn_blocking(move || {
            let mut unified = UnifiedJsInterface::new();
            unified.parse_for_analysis(script)
        });
        handles.push(handle);
    }
    
    let results: Vec<_> = futures::future::try_join_all(handles)
        .await?
        .into_iter()
        .collect::<Result<_, _>>()?;
    Ok(results)
}
```

## 故障排查

### 问题：V8 引擎不可用

**症状**：`execute()` 总是回退到 Boa，性能不如预期

**解决**：
1. 检查 V8 module (`browerai-js-v8`) 是否编译成功
2. 确认 `browerai-ai-integration` 的依赖正确包含了 V8
3. 验证特性标志: `cargo build --features ai,v8`

### 问题：SWC 模块检测不准确

**症状**：错误地将脚本识别为模块（或反之）

**解决**：
1. 启用 `swc-full` 特性使用完整 SWC 解析器而非启发式
2. 检查代码中是否包含检测规则的边界情况
3. 提交 issue 或调整启发式规则

### 问题：内存溢出

**症状**：处理大量文件时内存持续增长

**解决**：
1. 不在循环中反复创建编排器实例
2. 在处理完每个批次后显式 drop orchestrator
3. 使用 `quick_validate()` 代替 `parse()` 用于简单的语法检查

## 最佳实践

1. **使用 UnifiedJsInterface**：大多数情况下，顶层 facade 已足够
2. **根据场景选择策略**：渲染用 balanced，分析用 balanced，不可信代码用 secure
3. **避免重复初始化**：保留编排器实例供重用
4. **启用日志**：设置 `RUST_LOG=debug` 查看引擎选择过程
5. **测试回退路径**：确保在所有引擎都不可用时也能工作

## 参考资源

- [HybridJsOrchestrator 源码](../crates/browerai-ai-integration/src/js_orchestrator.rs)
- [RenderingJsExecutor 源码](../crates/browerai-renderer-core/src/js_executor.rs)
- [UnifiedJsInterface 源码](../crates/browerai/src/unified_js.rs)
- [测试用例](../tests/)
