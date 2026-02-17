# Rust 模块注册说明

## 添加到 lib.rs

在 `crates/browerai-ai-integration/src/lib.rs` 中的 `pub mod` 列表下添加：

```rust
pub mod framework_detector;
```

并在 pub use 部分添加：

```rust
pub use framework_detector::FrameworkDetectorIntegration;
```

## 完整示例

文件: `crates/browerai-ai-integration/src/lib.rs`

```rust
// ... 现有模块 ...

pub mod decoder;
pub mod integration;
pub mod framework_detector;  // ← 新增

// ... 现有 pub use ...

pub use framework_detector::FrameworkDetectorIntegration;  // ← 新增
```

完成后可以使用：

```rust
use browerai_ai_integration::FrameworkDetectorIntegration;
```
