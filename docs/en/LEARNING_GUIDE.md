# BrowerAI Learning and Tuning Guide

This guide explains how to use BrowerAI's autonomous learning features to visit real websites, collect feedback data, and adjust parameters.

## Quick Start

### 1. Learn from a Single Website

```bash
cargo run -- --learn https://example.com
```

### 2. Batch Visit Multiple Websites

```bash
cargo run -- --learn \
  https://example.com \
  https://httpbin.org/html \
  https://www.w3.org
```

### 3. View AI System Status

```bash
cargo run -- --ai-report
```

## Learning Workflow

```
Visit Website → Parse HTML/CSS/JS → Render → Collect Feedback → Export Training Data
```

Each learning session automatically:
- 📥 Downloads HTML content
- 🔍 Processes with AI-enhanced parsers
- 🎨 Extracts CSS rules and styles
- ⚙️ Analyzes JavaScript code
- 🖼️ Renders page and generates node tree
- 📊 Records performance metrics and errors
- 💾 Exports feedback data in JSON format

## Feedback Data Structure

Exported JSON files are located in `training/data/feedback_YYYYMMDD_HHMMSS.json`:

```json
[
  {
    "type": "html_parsing",
    "timestamp": "2026-01-04T10:38:39Z",
    "success": true,
    "ai_used": true,
    "complexity": 0.5,
    "content": "<html>...</html>",
    "size": 1024
  },
  {
    "type": "css_parsing",
    "timestamp": "2026-01-04T10:38:39Z",
    "success": true,
    "ai_used": true,
    "rule_count": 7,
    "content": "body { ... }"
  }
]
```

## Event Types

The feedback pipeline collects the following event types:

1. **html_parsing**: HTML parsing events with complexity metrics
2. **css_parsing**: CSS parsing events with rule counts
3. **js_parsing**: JavaScript parsing events with statement counts
4. **js_compatibility_violation**: JavaScript compatibility issues
5. **rendering_performance**: Rendering performance metrics
6. **layout_performance**: Layout calculation metrics
7. **model_inference**: AI model inference statistics

## Tuning Parameters

### Network Timeout

Adjust in `src/learning/website_learner.rs`:
```rust
let client = Client::builder()
    .timeout(Duration::from_secs(60))  // Increase for slow websites
    .build()?;
```

### Visit Delay

Adjust in `src/learning/website_learner.rs`:
```rust
std::thread::sleep(Duration::from_secs(3));  // Delay between visits
```

### Feedback Capacity

Adjust in `src/ai/feedback_pipeline.rs`:
```rust
impl Default for FeedbackPipeline {
    fn default() -> Self {
        Self::new(10000)  // Maximum events to keep
    }
}
```

## Training Workflow

After collecting feedback data, train new models:

```bash
# 1. Collect data from multiple websites
cargo run -- --learn https://example.com https://www.mozilla.org

# 2. Verify feedback data was saved
ls -lh training/data/feedback_*.json

# 3. Train models (see training/QUICKSTART.md)
cd training && python scripts/train_html_parser_v2.py

# 4. Deploy trained models
cp training/models/*.onnx models/local/

# 5. Test with new models
cargo run -- --ai-report
```

## Performance Monitoring

Monitor learning effectiveness:

```bash
# View detailed AI system report
cargo run -- --ai-report

# Check model health
cat models/model_config.toml

# Analyze feedback statistics
jq '[.[] | .type] | group_by(.) | map({type: .[0], count: length})' \
  training/data/feedback_*.json
```

## Best Practices

1. **Diverse Website Selection**: Visit websites with different structures and technologies
2. **Incremental Learning**: Start with simple websites, gradually increase complexity
3. **Regular Validation**: Run `--ai-report` to monitor model health
4. **Data Cleanup**: Remove old feedback files to save disk space
5. **Model Versioning**: Use semantic versioning for trained models

## Troubleshooting

### Issue: Network Timeout
- Increase timeout in `website_learner.rs`
- Check internet connection
- Try different websites

### Issue: Feedback Not Saving
- Ensure `training/data/` directory exists
- Check write permissions
- Verify disk space

### Issue: Models Not Loading
- Check `models/local/` directory exists
- Verify ONNX files are present
- Validate `model_config.toml` syntax

## Advanced Features

### Custom Learning Scripts

Create custom scripts for specific learning scenarios:

```bash
# Learn from a list of URLs
cat urls.txt | while read url; do
  cargo run -- --learn "$url"
  sleep 5
done
```

### Batch Processing

Use the batch collection script:

```bash
BROWSER_BIN=./target/release/browerai \
  BATCH_SIZE=10 START=1 STOP=10 \
  bash scripts/collect_sites.sh
```

## See Also

- [Quick Reference](QUICKREF.md) - Command reference
- [Training Guide](../../training/QUICKSTART.md) - Model training
- [Implementation Guide](IMPLEMENTATION_GUIDE.md) - Technical details
