# Phase 2-5 执行和验证指南

**创建日期**: 2026-01-07  
**状态**: ✅ 所有文件已创建，编译通过  
**总新增测试**: 67 个 (Phase 2-4) + 7 个单元测试 (Phase 5)

---

## 快速启动

### 1. 验证编译

```bash
cd /home/stone/BrowerAI
cargo build --tests
```

### 2. 查看新增文件

```bash
# 查看所有新增的测试文件
ls -lh tests/deobfuscation_*.rs tests/real_world_*.rs

# 查看新增的模块
ls -lh crates/browerai-learning/src/{website_deobfuscator,execution_validator}.rs
```

### 3. 查看核心文档

```bash
cat docs/PHASE_2_5_IMPLEMENTATION_GUIDE.md
cat docs/PHASE_2_5_COMPLETION_SUMMARY.md
```

---

## Phase 2: 控制流和死代码测试

### 位置

`tests/deobfuscation_controlflow_tests.rs`

### 运行

```bash
cargo test deobfuscation_controlflow
cargo test deobfuscation_controlflow -- --nocapture
```

### 覆盖点

- if/switch/三元运算符
- 死代码移除、未使用变量
- for/while/do-while、状态机、深度嵌套

---

## Phase 3: 变量和函数处理测试

### 位置

`tests/deobfuscation_variables_functions_tests.rs`

### 运行

```bash
cargo test deobfuscation_variables_functions
cargo test deobfuscation_variables_functions -- --nocapture
```

### 覆盖点

- 变量重命名（单字母、数字后缀、无意义）
- 函数检测、参数重命名、匿名/箭头/异步函数
- 闭包与作用域、类与继承、高阶函数、递归、IIFE

---

## Phase 4: 真实网站反混淆验证

### 位置

`tests/real_world_deobfuscation_tests.rs`（含 1 个联网测试，默认 #[ignore]）

### 新增模块

`crates/browerai-learning/src/website_deobfuscator.rs`（使用 reqwest 拉取真实 JS）

### 运行（离线基准）

```bash
cargo test real_world_deobfuscation
cargo test real_world_deobfuscation -- --nocapture
```

### 运行（真实线上库，需要网络）

```bash
cargo test test_real_world_minified_libraries_execution_validation -- --ignored --nocapture
```

覆盖 React 18 UMD 与 Day.js 1.11.10 的最小化代码，流程：拉取 → 反混淆 → `ExecutionValidator` 校验可执行性。

### 自定义 URL 示例

```rust
let mut verifier = WebsiteDeobfuscationVerifier::new();
let result = verifier.verify_website("https://example.com/app.js", None)?;
let exec = ExecutionValidator::new().validate_execution(&result.original_code, &result.deobfuscated_code);
```

⚠️ 注意：
- 需要外网访问（可设置 `HTTP_PROXY/HTTPS_PROXY`）。
- 联网测试默认 `#[ignore]`，避免 CI 无网失败。
- `WebsiteDeobfuscationResult` 包含 `original_code` 与 `deobfuscated_code` 便于二次验证。

---

## Phase 5: 代码执行验证框架

### 位置

`crates/browerai-learning/src/execution_validator.rs`

### 运行

```bash
cargo test --lib execution_validator
cargo test execution_validator::tests
```

### 使用示例

```rust
let validator = ExecutionValidator::new();
let result = validator.validate_execution("var a = 42;", "var answer = 42;");
assert!(result.is_valid_syntax);
let safety = validator.report_safety(&result);
println!("Safety Score: {:.1}", safety.safety_score);
```

### 检测能力

- 语法验证（JS 解析器优先，括号匹配回退）
- 风险检测：eval、document.write 等
- 特性检测：async/await、类、箭头函数
- 安全评分：0-100 分

---

## 统一测试运行

```bash
cargo test deobfuscation
cargo test real_world
cargo test deobfuscation -- --nocapture 2>&1 | tail -100
```

### 测试统计

```bash
grep "^#\[test\]" tests/deobfuscation_controlflow_tests.rs | wc -l
grep "^#\[test\]" tests/deobfuscation_variables_functions_tests.rs | wc -l
grep "^#\[test\]" tests/real_world_deobfuscation_tests.rs | wc -l
grep "^    #\[test\]" crates/browerai-learning/src/execution_validator.rs | wc -l
```

---

## 覆盖率

```bash
cargo llvm-cov -p browerai-learning --html
open target/llvm-cov/html/index.html
```

预期：Phase 5 后整体覆盖率目标 85%+。

---

## 常见问题 (FAQ)

### 测试编译失败，提示 unresolved module？

```bash
cargo build -p browerai-learning
cargo build --tests
```

### 找不到 JsDeobfuscator？

确保在 `browerai/src/lib.rs` 已导出：

```rust
pub use crate::learning::{JsDeobfuscator, DeobfuscationStrategy};
```

### 某些测试失败？

```bash
cargo test deobfuscation -- --nocapture 2>&1 | grep "FAILED"
```

### 只运行特定测试？

```bash
cargo test closure
cargo test real_world
```

---

## 最后检查清单

- [x] 所有文件成功创建并可编译
- [x] 导入语句正确
- [x] 测试命名清晰且包含断言
- [ ] 已运行全部测试
- [ ] 已检查覆盖率
- [ ] 如有需要已做性能优化
