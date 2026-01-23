# Renderer 管线集成指南

## 概述

本文档说明如何在 BrowerAI 的渲染管线中集成混合 JS 编排器（HybridJsOrchestrator），以支持动态 JS 执行和 DOM 操作。

## 集成点

### 1. 脚本执行（Script Execution）

在渲染过程中，某些 HTML `<script>` 标签需要被执行以修改 DOM：

```
HTML Parse → DOM Construction → Script Execution → Layout → Paint
                                     ↑
                      使用 RenderingJsExecutor
```

**当前状态**：renderer-core 已有 RenderingJsExecutor 包装器，但尚未集成到实际管线。

**集成方式**：
1. 在 RenderEngine 中添加 RenderingJsExecutor 实例
2. 在 render() 方法中调用脚本执行

### 2. 属性计算（Computed Properties）

CSS 中的某些值（如 `calc()`, JavaScript-computed 值）需要执行 JS：

```css
width: calc(100% - 20px);
content: attr(data-js-value);  /* 可能需要 JS 计算 */
```

### 3. 框架支持（Framework Support）

React、Vue、Angular 等框架的初始化需要：
- 解析框架代码
- 执行初始化脚本
- 捕获生成的 DOM

## 实现步骤

### Step 1：为 RenderEngine 添加 JS 执行器

```rust
use super::js_executor::RenderingJsExecutor;

pub struct RenderEngine {
    layout_engine: LayoutEngine,
    paint_engine: PaintEngine,
    #[cfg(feature = "ai")]
    js_executor: Option<RenderingJsExecutor>,  // 新增
}

impl RenderEngine {
    pub fn new() -> Self {
        Self::with_viewport(800.0, 600.0)
    }

    pub fn with_viewport(width: f32, height: f32) -> Self {
        #[cfg(feature = "ai")]
        let js_executor = Some(RenderingJsExecutor::new());

        #[cfg(not(feature = "ai"))]
        let js_executor = None;

        Self {
            layout_engine: LayoutEngine::new(width, height),
            paint_engine: PaintEngine::with_viewport(width, height),
            #[cfg(feature = "ai")]
            js_executor,
        }
    }
}
```

### Step 2：在渲染过程中执行脚本

```rust
pub fn render(&mut self, dom: &RcDom, styles: &[CssRule]) -> Result<RenderTree> {
    log::info!("Starting render process");

    // ... existing layout code ...

    // 新增：执行页面脚本
    self.execute_scripts(dom)?;

    // ... existing paint code ...

    Ok(render_tree)
}

/// 执行 DOM 中的脚本
fn execute_scripts(&mut self, dom: &RcDom) -> Result<()> {
    #[cfg(feature = "ai")]
    {
        if let Some(executor) = &mut self.js_executor {
            // 查找所有 <script> 标签
            let scripts = self.extract_scripts(dom)?;
            
            for script in scripts {
                log::debug!("Executing script: {} bytes", script.len());
                executor.execute(&script)?;
            }
        }
    }

    Ok(())
}

/// 从 DOM 中提取脚本内容
fn extract_scripts(&self, dom: &RcDom) -> Result<Vec<String>> {
    let mut scripts = Vec::new();
    
    // 遍历 DOM，查找 <script> 节点
    // 这里需要使用 markup5ever 的 DOM 遍历 API
    
    Ok(scripts)
}
```

### Step 3：处理 DOM 修改

当 JS 执行时，可能会修改 DOM（通过 `document.write()`、`innerHTML` 等）。需要捕获这些修改：

```rust
/// 带 JS 环境的 DOM 操作上下文
pub struct JsRenderContext {
    dom: RcDom,
    execution_log: Vec<DomOperation>,
}

#[derive(Debug, Clone)]
pub enum DomOperation {
    SetInnerHTML { selector: String, html: String },
    SetProperty { selector: String, prop: String, value: String },
    InsertElement { parent: String, html: String, position: usize },
    RemoveElement { selector: String },
}

impl JsRenderContext {
    /// 记录 JS 执行导致的 DOM 变化
    pub fn apply_operations(&mut self, ops: Vec<DomOperation>) -> Result<()> {
        for op in ops {
            match op {
                DomOperation::SetInnerHTML { selector, html } => {
                    // 使用 selector 查找元素，设置 innerHTML
                }
                DomOperation::SetProperty { selector, prop, value } => {
                    // 设置元素属性
                }
                // ... 其他操作 ...
            }
        }
        Ok(())
    }
}
```

## 配置和控制

### 环境变量

```bash
# 渲染管线的 JS 执行策略
export BROWERAI_RENDER_JS_POLICY=balanced  # 默认
export BROWERAI_RENDER_JS_POLICY=performance  # 性能优先
export BROWERAI_RENDER_JS_POLICY=secure  # 安全优先
```

### 特性标志

```bash
# 启用完整 JS 执行支持
cargo build --features ai

# 启用 V8 以获得最高性能
cargo build --features ai,v8
```

## 性能考虑

### 1. 脚本执行顺序

HTML 中的脚本应该按照出现顺序执行，以保持依赖关系正确：

```html
<script src="jquery.js"></script>
<script src="app.js"></script>  <!-- 依赖 jquery -->
```

### 2. 缓存和优化

对于重复出现的脚本，可以使用缓存避免重新执行：

```rust
use std::collections::HashMap;

pub struct ScriptCache {
    cache: HashMap<String, String>,  // script_hash -> execution_result
}

impl ScriptCache {
    pub fn get_or_execute(&mut self, script: &str, executor: &mut RenderingJsExecutor) -> Result<String> {
        let hash = format!("{:x}", md5::compute(script.as_bytes()));
        
        if let Some(result) = self.cache.get(&hash) {
            return Ok(result.clone());
        }
        
        let result = executor.execute(script)?;
        self.cache.insert(hash, result.clone());
        Ok(result)
    }
}
```

### 3. 超时控制

对于可能耗时的脚本，设置执行超时：

```rust
pub fn execute_with_timeout(&mut self, script: &str, timeout_ms: u64) -> Result<String> {
    use std::time::Duration;
    use std::thread;
    
    let timeout = Duration::from_millis(timeout_ms);
    
    // 在独立线程执行
    let executor = self.js_executor.take();
    let script = script.to_string();
    
    let result = thread::spawn(move || {
        if let Some(mut exec) = executor {
            exec.execute(&script)
        } else {
            Err(anyhow::anyhow!("No executor available"))
        }
    });
    
    match result.join() {
        Ok(exec_result) => {
            self.js_executor = /* 恢复 executor */;
            exec_result
        }
        Err(_) => {
            Err(anyhow::anyhow!("Script execution timeout"))
        }
    }
}
```

## 错误处理

### 脚本执行失败

如果脚本执行失败，应该：
1. 记录错误日志
2. 继续渲染（使用修改前的 DOM）
3. 标记页面为部分失败

```rust
pub fn execute_scripts_safe(&mut self, dom: &RcDom) -> Result<()> {
    let scripts = self.extract_scripts(dom)?;
    
    for (idx, script) in scripts.iter().enumerate() {
        match self.execute_script_internal(script) {
            Ok(_) => {
                log::debug!("Script {} executed successfully", idx);
            }
            Err(e) => {
                log::warn!("Script {} execution failed: {}", idx, e);
                // 继续处理其他脚本
            }
        }
    }
    
    Ok(())
}
```

## 测试

### 单元测试

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_with_script() {
        let mut engine = RenderEngine::new();
        let html = r#"
            <html>
                <body>
                    <div id="app"></div>
                    <script>
                        document.getElementById('app').innerHTML = 'Hello from JS';
                    </script>
                </body>
            </html>
        "#;
        
        // 解析 HTML
        let dom = parse_html(html);
        let styles = vec![];
        
        // 渲染
        let result = engine.render(&dom, &styles);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_script_execution_order() {
        // 测试脚本按照声明顺序执行
        // 确保依赖关系正确
    }
}
```

### 集成测试

```rust
#[test]
fn test_render_react_app() {
    // 使用真实的 React 应用进行测试
    let html = include_str!("../fixtures/react_app.html");
    
    let mut engine = RenderEngine::new();
    let dom = parse_html(html);
    let styles = extract_styles(&dom);
    
    let result = engine.render(&dom, &styles).expect("render failed");
    
    // 验证结果包含 React 生成的内容
    assert!(result.to_string().contains("expected-content"));
}
```

## 相关文档

- [混合 JS 编排器集成指南](./HYBRID_JS_ORCHESTRATION_INTEGRATION.md)
- [RenderingJsExecutor API](../crates/browerai-renderer-core/src/js_executor.rs)
- [快速参考](./HYBRID_JS_QUICK_REFERENCE.md)

## 后续工作

1. **DOM 修改跟踪**：完整实现 DOM 操作的捕获和应用
2. **框架支持**：添加 React、Vue 等框架的初始化脚本
3. **性能优化**：实现脚本缓存和执行优化
4. **错误恢复**：完善脚本失败的处理机制
5. **调试工具**：添加脚本执行的调试能力
