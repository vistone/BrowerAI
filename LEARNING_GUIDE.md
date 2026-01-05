# BrowerAI 学习与调优指南

本指南介绍如何使用 BrowerAI 的自主学习功能访问真实网站、收集反馈数据并调整参数。

## 快速开始

### 1. 访问单个网站学习

```bash
cargo run --bin browerai -- --learn https://example.com
```

### 2. 批量访问多个网站

```bash
cargo run --bin browerai -- --learn \
  https://example.com \
  https://httpbin.org/html \
  https://www.w3.org
```

### 3. 查看 AI 系统状态

```bash
cargo run --bin browerai -- --ai-report
```

## 学习流程

```
访问网站 → 解析 HTML/CSS/JS → 渲染 → 收集反馈 → 导出训练数据
```

每次学习会自动：
- 📥 下载 HTML 内容
- 🔍 使用 AI 增强的解析器处理
- 🎨 提取 CSS 规则和样式
- ⚙️ 分析 JavaScript 代码
- 🖼️ 渲染页面生成节点树
- 📊 记录性能指标和错误
- 💾 导出 JSON 格式反馈数据

## 反馈数据结构

导出的 JSON 文件位于 `training/data/feedback_YYYYMMDD_HHMMSS.json`：

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

**事件类型**：
- `html_parsing`: HTML 解析事件（complexity 表示复杂度 0.0-1.0）
- `css_parsing`: CSS 解析事件（rule_count 表示规则数量）
- `js_parsing`: JavaScript 解析事件
- `js_compatibility_violation`: JS 兼容性问题
- `rendering_performance`: 渲染性能数据
- `layout_performance`: 布局计算性能
- `model_inference`: 模型推理统计

## 调整参数

### 1. 网络请求超时

编辑 [src/learning/website_learner.rs](src/learning/website_learner.rs#L30)：

```rust
fn create_client() -> Result<Client> {
    Ok(Client::builder()
        .timeout(Duration::from_secs(30))  // 修改这里：30秒超时
        .build()?)
}
```

**建议值**：
- 快速网站（CDN）：10-15 秒
- 普通网站：30 秒（默认）
- 慢速网站：60 秒

### 2. 批量访问延迟

编辑 [src/learning/website_learner.rs](src/learning/website_learner.rs#L104)：

```rust
pub fn batch_visit(&self, urls: Vec<String>, runtime: &mut AiRuntime) -> Vec<VisitReport> {
    // ...
    std::thread::sleep(Duration::from_secs(1));  // 修改这里：延迟时间
}
```

**建议值**：
- 本地测试：0 秒
- 正常爬取：1-2 秒（默认 1 秒）
- 礼貌爬取：3-5 秒
- 谨慎爬取：10+ 秒

### 3. 反馈事件容量

编辑 [src/ai/feedback_pipeline.rs](src/ai/feedback_pipeline.rs#L104)：

```rust
pub fn new() -> Self {
    Self {
        events: Vec::with_capacity(10000),  // 修改这里：事件容量
    }
}
```

**建议值**：
- 小型测试：1,000
- 中型学习：10,000（默认）
- 大规模收集：100,000

### 4. AI 复杂度阈值

编辑 [src/parser/html.rs](src/parser/html.rs#L86)：

```rust
fn analyze_with_ai(&self, _dom: &RcDom) -> (bool, f32) {
    // 模拟 AI 验证
    let complexity = 0.5;  // 修改这里：复杂度基准
    (true, complexity)
}
```

**建议值**：
- 简单页面：0.2-0.4
- 普通页面：0.5（默认）
- 复杂页面：0.7-0.9

### 5. CSS 优化规则数

编辑 [src/parser/css.rs](src/parser/css.rs#L82)：

```rust
fn generate_optimizations(&self, _rules: &[CssRule]) -> Vec<CssRule> {
    // 模拟 AI 优化建议
    let optimization_count = original_count + 3;  // 修改这里：建议数量
    // ...
}
```

**建议值**：
- 保守优化：+1 到 +2
- 平衡优化：+3（默认）
- 激进优化：+5 到 +10

## 实验建议

### 阶段 1：基准测试（1-2 天）

访问 10-20 个知名网站建立基准：

```bash
cargo run --bin browerai -- --learn \
  https://example.com \
  https://www.wikipedia.org \
  https://github.com \
  https://www.rust-lang.org
```

记录：
- 平均获取时间
- 平均渲染节点数
- CSS 规则分布
- 成功率

### 阶段 2：参数调优（3-5 天）

根据阶段 1 数据调整：

1. **超时太短**（很多失败）→ 增加到 60 秒
2. **复杂度偏离**（都是 0.5）→ 改用真实计算
3. **反馈过多**（接近 10000）→ 增加容量到 50000

### 阶段 3：真实模型训练（1-2 周）

1. 收集 1000+ 网站的反馈数据：
```bash
# 运行自动化脚本
for url in $(cat websites.txt); do
  cargo run --bin browerai -- --learn $url
  sleep 5
done
```

2. 使用训练脚本（参考 [training/QUICKSTART.md](training/QUICKSTART.md)）：
```bash
cd training
python scripts/train_html_parser_v2.py --data ../training/data/*.json
python scripts/train_css_parser.py --data ../training/data/*.json
```

3. 部署训练好的模型：
```bash
cp training/models/*.onnx models/local/
```

4. 更新配置文件 `models/model_config.toml`：
```toml
[[models]]
name = "html_parser_v2"
model_type = "HtmlParser"
path = "html_parser_v2.onnx"
version = "2.0.0"
enabled = true
```

5. 用 `--features ai` 重新编译：
```bash
cargo build --release --features ai
```

### 阶段 4：A/B 测试（持续）

比较模型版本：
```bash
# 版本 1
cargo run --bin browerai -- --ai-report

# 切换到版本 2（修改 model_config.toml）
cargo run --bin browerai -- --ai-report

# 对比性能数据
```

## 监控指标

### 解析性能
- HTML 解析耗时（< 1ms 为优秀）
- CSS 规则提取数量
- JS 语句解析数量

### 渲染性能
- 渲染节点总数
- 布局计算耗时
- 绘制操作数量

### AI 增强效果
- AI 使用率（ai_used: true 的占比）
- 复杂度分布（0.0-1.0 范围）
- 优化建议采纳率

### 网络性能
- 平均获取时间
- 成功率（应 > 95%）
- 超时/错误数量

## 调试技巧

### 查看详细日志

```bash
RUST_LOG=debug cargo run --bin browerai -- --learn https://example.com
```

### 单独测试组件

```rust
// 在 examples/ 目录创建测试文件
use browerai::learning::website_learner::WebsiteLearner;

fn main() {
    let learner = WebsiteLearner::new();
    let report = learner.visit_and_learn("https://example.com", &mut runtime).unwrap();
    println!("{}", report.format());
}
```

### 验证反馈数据

```bash
# 检查 JSON 格式
jq '.' training/data/feedback_*.json

# 统计事件类型
jq '[.[] | .type] | group_by(.) | map({type: .[0], count: length})' \
  training/data/feedback_*.json

# 计算平均复杂度
jq '[.[] | select(.type == "html_parsing") | .complexity] | add / length' \
  training/data/feedback_*.json
```

## 常见问题

### Q: 为什么所有网站的复杂度都是 0.5？
A: 当前使用模拟 AI（stub mode）。需要训练并部署真实 ONNX 模型后才有动态复杂度。

### Q: 如何处理 HTTPS 证书错误？
A: 在 `create_client()` 中添加：
```rust
Client::builder()
    .danger_accept_invalid_certs(true)  // 仅用于测试！
    .build()
```

### Q: 批量访问时如何避免被封？
A: 
1. 增加延迟到 3-5 秒
2. 添加随机 User-Agent
3. 使用代理池轮换 IP
4. 遵守 robots.txt

### Q: 反馈数据太大怎么办？
A: 
1. 增加 `events` 容量
2. 定期调用 `export_training_samples()` 并清空
3. 实现分文件存储策略

## 下一步

1. ✅ 访问真实网站收集数据
2. ⏳ 调整参数优化性能
3. ⏳ 训练第一个 ONNX 模型
4. ⏳ A/B 测试模型版本
5. ⏳ 实现在线学习闭环

查看 [ROADMAP.md](ROADMAP.md) 了解完整路线图。
