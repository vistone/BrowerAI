# BrowerAI 快速参考

## 🚀 快速命令

```bash
# 演示 AI 集成（使用内置示例）
cargo run

# 查看 AI 系统状态
cargo run -- --ai-report

# 访问单个网站学习
cargo run -- --learn https://example.com

# 批量访问多个网站
cargo run -- --learn https://example.com https://httpbin.org/html https://www.w3.org

# 导出反馈数据（学习模式会自动导出）
cargo run -- --export-feedback ./custom_path.json

# 详细日志
RUST_LOG=debug cargo run -- --learn https://example.com
```

## 📂 关键文件

| 文件 | 用途 |
|------|------|
| [src/ai/runtime.rs](../../src/ai/runtime.rs) | AI 运行时核心 |
| [src/ai/feedback_pipeline.rs](../../src/ai/feedback_pipeline.rs) | 反馈事件收集 |
| [src/ai/reporter.rs](../../src/ai/reporter.rs) | AI 状态报告 |
| [src/learning/website_learner.rs](../../src/learning/website_learner.rs) | 网站访问学习器 |
| [src/main.rs](../../src/main.rs) | CLI 入口（4 种模式）|
| [models/model_config.toml](../../models/model_config.toml) | 模型配置 |
| `training/data/feedback_*.json` | 反馈数据（自动生成）|

## 🔧 调整参数

### 网络超时（30 秒 → 60 秒）
```rust
.timeout(Duration::from_secs(60))  // 改这里
```

### 访问延迟（1 秒 → 3 秒）
```rust
std::thread::sleep(Duration::from_secs(3));  // 改这里
```

## 📊 反馈数据格式

```json
[
  {
    "type": "html_parsing",
    "timestamp": "2026-01-04T10:38:39Z",
    "success": true,
    "ai_used": true,
    "complexity": 0.5
  }
]
```

### 事件类型
- `html_parsing`: HTML 解析
- `css_parsing`: CSS 解析
- `js_parsing`: JS 解析
- `rendering_performance`: 渲染性能
- `model_inference`: 模型推理统计

## 🛠️ 常用操作

### 查看反馈数据
```bash
# 查看最新的反馈文件
ls -lt training/data/feedback_*.json | head -1

# 格式化查看
cat training/data/feedback_*.json | jq '.'
```

## 🔍 调试

### 启用详细日志
```bash
# 所有调试日志
RUST_LOG=debug cargo run

# 特定模块
RUST_LOG=browerai::ai=debug cargo run
```

## 📚 相关文档

- [完整 README](README.md)
- [英文文档](../en/README.md)
- [训练指南](../../training/README.md)
