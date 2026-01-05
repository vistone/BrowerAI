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
| [src/ai/runtime.rs](src/ai/runtime.rs) | AI 运行时核心 |
| [src/ai/feedback_pipeline.rs](src/ai/feedback_pipeline.rs) | 反馈事件收集 |
| [src/ai/reporter.rs](src/ai/reporter.rs) | AI 状态报告 |
| [src/learning/website_learner.rs](src/learning/website_learner.rs) | 网站访问学习器 |
| [src/main.rs](src/main.rs) | CLI 入口（4 种模式）|
| [models/model_config.toml](models/model_config.toml) | 模型配置 |
| `training/data/feedback_*.json` | 反馈数据（自动生成）|

## 🔧 调整参数

### 网络超时（30 秒 → 60 秒）
[src/learning/website_learner.rs:30](src/learning/website_learner.rs#L30)
```rust
.timeout(Duration::from_secs(60))  // 改这里
```

### 访问延迟（1 秒 → 3 秒）
[src/learning/website_learner.rs:104](src/learning/website_learner.rs#L104)
```rust
std::thread::sleep(Duration::from_secs(3));  // 改这里
```

### 反馈容量（10000 → 50000）
[src/ai/feedback_pipeline.rs:104](src/ai/feedback_pipeline.rs#L104)
```rust
events: Vec::with_capacity(50000),  // 改这里
```

## 📊 反馈数据格式

```json
[
  {
    "type": "html_parsing",
    "timestamp": "2026-01-04T10:38:39Z",
    "success": true,
    "ai_used": true,
    "complexity": 0.5,
    "error": null
  },
  {
    "type": "css_parsing",
    "timestamp": "2026-01-04T10:38:39Z",
    "success": true,
    "ai_used": true,
    "rule_count": 7,
    "error": null
  }
]
```

### 事件类型
- `html_parsing`: HTML 解析（complexity: 0.0-1.0）
- `css_parsing`: CSS 解析（rule_count: 规则数）
- `js_parsing`: JS 解析（statement_count: 语句数）
- `js_compatibility_violation`: JS 兼容性问题
- `rendering_performance`: 渲染性能
- `layout_performance`: 布局性能
- `model_inference`: 模型推理统计

## 🛠️ 常用操作

### 查看反馈数据
```bash
# 查看最新的反馈文件
ls -lt training/data/feedback_*.json | head -1

# 格式化查看
cat training/data/feedback_*.json | jq '.'

# 统计事件类型
jq '[.[] | .type] | group_by(.) | map({type: .[0], count: length})' \
  training/data/feedback_*.json

# 计算平均复杂度
jq '[.[] | select(.type == "html_parsing") | .complexity] | add / length' \
  training/data/feedback_*.json
```

### 批量访问网站
```bash
# 从文件读取 URL 列表
while read url; do
  cargo run --bin browerai -- --learn "$url"
  sleep 5  # 礼貌延迟
done < websites.txt
```

### 合并反馈数据
```bash
# 合并所有反馈文件到一个
jq -s 'add' training/data/feedback_*.json > training/data/merged_feedback.json
```

## 🎯 学习工作流

```
1. 收集数据
   ↓
   cargo run -- --learn <urls>
   
2. 查看反馈
   ↓
   cat training/data/feedback_*.json | jq '.'
   
3. 训练模型
   ↓
   cd training && python scripts/train_html_parser_v2.py
   
4. 部署模型
   ↓
   cp training/models/*.onnx models/local/
   
5. 更新配置
   ↓
   vim models/model_config.toml
   
6. 重新编译
   ↓
   cargo build --release --features ai
   
7. 测试新模型
   ↓
   cargo run -- --ai-report
```

## 📈 监控指标

### 解析性能
- HTML 解析 < 1ms（优秀）
- CSS 规则提取数量
- JS 语句解析数量

### 网络性能
- 平均获取时间
- 成功率 > 95%
- 超时/错误数量

### AI 效果
- AI 使用率（ai_used: true）
- 复杂度分布
- 优化建议数量

## 🐛 调试

### 启用详细日志
```bash
RUST_LOG=trace cargo run -- --learn https://example.com
```

### 仅看特定模块
```bash
RUST_LOG=browerai::learning=debug cargo run -- --learn https://example.com
```

### 测试单个组件
```rust
// examples/test_learner.rs
use browerai::learning::website_learner::WebsiteLearner;
use browerai::ai::AiRuntime;

fn main() {
    let learner = WebsiteLearner::new();
    let mut runtime = AiRuntime::with_stub();
    let report = learner.visit_and_learn("https://example.com", &mut runtime).unwrap();
    println!("{}", report.format());
}
```

## 📚 完整文档

- [LEARNING_GUIDE.md](LEARNING_GUIDE.md) - 学习与调优详细指南
- [AI_LEARNING_IMPLEMENTATION.md](AI_LEARNING_IMPLEMENTATION.md) - 实现技术报告
- [GETTING_STARTED.md](GETTING_STARTED.md) - 项目入门
- [training/QUICKSTART.md](training/QUICKSTART.md) - 模型训练快速开始

## ⚡ 常见问题

**Q: 为什么复杂度都是 0.5？**  
A: 当前使用模拟 AI（stub mode）。需要训练真实 ONNX 模型后启用 `--features ai`。

**Q: 如何处理 HTTPS 错误？**  
A: 仅测试时在 `create_client()` 添加 `.danger_accept_invalid_certs(true)`。

**Q: 批量访问被封了？**  
A: 增加延迟、添加 User-Agent、使用代理、遵守 robots.txt。

**Q: 反馈数据太大？**  
A: 增加容量、定期导出并清空、实现分文件存储。

---

💡 提示：首次使用建议先运行 `cargo run` 查看演示，然后用 `cargo run -- --learn https://example.com` 测试真实访问。
