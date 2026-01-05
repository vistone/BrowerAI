# BrowerAI Quick Reference

## 🚀 Quick Commands

```bash
# Demo AI integration (using built-in examples)
cargo run

# View AI system status
cargo run -- --ai-report

# Learn from a single website
cargo run -- --learn https://example.com

# Batch visit multiple websites
cargo run -- --learn https://example.com https://httpbin.org/html https://www.w3.org

# Export feedback data (learning mode exports automatically)
cargo run -- --export-feedback ./custom_path.json

# Detailed logging
RUST_LOG=debug cargo run -- --learn https://example.com
```

## 📂 Key Files

| File | Purpose |
|------|---------|
| [src/ai/runtime.rs](../../src/ai/runtime.rs) | AI runtime core |
| [src/ai/feedback_pipeline.rs](../../src/ai/feedback_pipeline.rs) | Feedback event collection |
| [src/ai/reporter.rs](../../src/ai/reporter.rs) | AI status reporting |
| [src/learning/website_learner.rs](../../src/learning/website_learner.rs) | Website visiting learner |
| [src/main.rs](../../src/main.rs) | CLI entry (4 modes) |
| [models/model_config.toml](../../models/model_config.toml) | Model configuration |
| `training/data/feedback_*.json` | Feedback data (auto-generated) |

## 🔧 Adjusting Parameters

### Network Timeout (30s → 60s)
[src/learning/website_learner.rs:30](../../src/learning/website_learner.rs#L30)
```rust
.timeout(Duration::from_secs(60))  // Change here
```

### Visit Delay (1s → 3s)
[src/learning/website_learner.rs:104](../../src/learning/website_learner.rs#L104)
```rust
std::thread::sleep(Duration::from_secs(3));  // Change here
```

### Feedback Capacity (10000 → 50000)
[src/ai/feedback_pipeline.rs:104](../../src/ai/feedback_pipeline.rs#L104)
```rust
events: Vec::with_capacity(50000),  // Change here
```

## 📊 Feedback Data Format

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

### Event Types
- `html_parsing`: HTML parsing (complexity: 0.0-1.0)
- `css_parsing`: CSS parsing (rule_count: number of rules)
- `js_parsing`: JS parsing (statement_count: number of statements)
- `js_compatibility_violation`: JS compatibility issues
- `rendering_performance`: Rendering performance
- `layout_performance`: Layout performance
- `model_inference`: Model inference statistics

## 🛠️ Common Operations

### View Feedback Data
```bash
# View the latest feedback file
ls -lt training/data/feedback_*.json | head -1

# Format and view
cat training/data/feedback_*.json | jq '.'

# Count event types
jq '[.[] | .type] | group_by(.) | map({type: .[0], count: length})' \
  training/data/feedback_*.json
```

### Clean Old Data
```bash
# Keep only the last 10 feedback files
ls -t training/data/feedback_*.json | tail -n +11 | xargs rm
```

## 🔍 Debugging

### Enable Detailed Logging
```bash
# All debug logs
RUST_LOG=debug cargo run

# Specific module
RUST_LOG=browerai::ai=debug cargo run

# Multiple modules
RUST_LOG=browerai::ai=debug,browerai::parser=debug cargo run
```

### Check Model Loading
```bash
# List models
ls -lh models/local/

# Verify model config
cat models/model_config.toml
```

## 📈 Performance Tuning

### Batch Processing
Adjust batch size in `scripts/collect_sites.sh`:
```bash
BATCH_SIZE=10 START=1 STOP=10 bash scripts/collect_sites.sh
```

### Parallel Processing
Set RUST_LOG level appropriately to reduce overhead:
```bash
RUST_LOG=info cargo run --release  # Production
RUST_LOG=debug cargo run           # Development
```

## 🆘 Troubleshooting

### Common Issues

**Issue**: Models not loading
- Check `models/local/` directory exists
- Verify ONNX files are present
- Check `model_config.toml` syntax

**Issue**: Feedback data not saving
- Ensure `training/data/` directory exists
- Check write permissions
- Verify disk space

**Issue**: Network timeout
- Increase timeout in website_learner.rs
- Check internet connection
- Try different websites

## 📚 Related Documentation

- [Full README](README.md)
- [Getting Started](GETTING_STARTED.md)
- [Learning Guide](LEARNING_GUIDE.md)
- [Training Guide](../../training/README.md)
