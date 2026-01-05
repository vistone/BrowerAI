# BrowerAI - AI Coding Agent Instructions

## Project Overview
BrowerAI is an experimental AI-powered browser that uses machine learning models to autonomously parse and render HTML/CSS/JS. Unlike traditional browsers with hard-coded parsers, this project integrates ONNX models into the parsing and rendering pipeline.

## Architecture

**Dual Pipeline System**: Rust core + Python training pipeline
- `src/`: Rust browser engine with AI integration
- `training/`: Python scripts for model training and ONNX export

**Core Module Hierarchy** (see [src/lib.rs](src/lib.rs)):
```
ai/          → Model management, ONNX inference, hot reload
parser/      → HTML (html5ever), CSS (cssparser), JS (boa_parser)
renderer/    → Layout engine, paint, predictive rendering
learning/    → Feedback, A/B testing, online learning, personalization
dom/         → Document model, JS sandbox
network/     → HTTP client, caching
plugins/     → Plugin system for extensibility
```

**AI Integration Pattern**: Each parser has optional AI enhancement:
```rust
// Pattern used in src/parser/html.rs, css.rs, js.rs
pub struct Parser {
    inference_engine: Option<InferenceEngine>,
    enable_ai: bool,
}
// with_ai() constructor enables ML-powered features
// Fallback to standard parsing when models unavailable
```

## Critical Development Workflows

### Building & Running
```bash
cargo build --release           # Rust core
cargo run                       # Run example in src/main.rs
cargo test                      # Run Rust tests
cargo test -- --nocapture       # See test output
```

### Model Training Pipeline
```bash
cd training
python scripts/train_html_parser.py --epochs 10
cp models/*.onnx ../models/local/   # Deploy to Rust app
```

**Model Registration**: Update `models/model_config.toml` after training:
```toml
[[models]]
name = "html_parser_v1"
model_type = "HtmlParser"
path = "html_parser_v1.onnx"
version = "1.0.0"
```

### Testing Strategy
- Unit tests: `#[cfg(test)] mod tests` in each module
- Integration tests: `tests/ai_integration_tests.rs`
- No mocking - direct integration with optional AI features

## Project-Specific Conventions

### Error Handling
Use `anyhow::Result` with context:
```rust
use anyhow::{Context, Result};
std::fs::read_to_string(path).context("Failed to read config")?;
```

### Logging
Standard `log` crate with `env_logger`:
```rust
log::info!("Operation succeeded");
log::debug!("AI enhancement enabled");
log::warn!("Model file not found, using fallback");
```

### Feature Flags
ONNX Runtime is optional: `cargo build --features ai`
- Default build: parsers work without AI
- With `ai` feature: enables model inference via `ort` crate

### Public API Design
- All major components re-exported from `src/lib.rs` for ergonomic imports
- Use `#[allow(dead_code)]` for future-ready methods not yet used
- Parsers implement `Default` trait pointing to `new()`

## Integration Points

### ONNX Runtime (ort crate)
- Version: `2.0.0-rc.10` (pinned - breaking changes between versions)
- Located in: `src/ai/inference.rs`, `src/ai/model_loader.rs`
- Models loaded from: `models/local/` directory
- Thread-safe: Uses `Arc<Session>` for shared inference

### Parser Dependencies
- **HTML**: `html5ever` + `markup5ever_rcdom` for DOM tree
- **CSS**: `cssparser` + `selectors` for rules/matching
- **JS**: `boa_parser` (pure Rust, no V8 dependency)

### Learning System
Phase 5 features in `src/learning/`:
- `feedback.rs`: User feedback collection
- `online_learning.rs`: Model fine-tuning
- `versioning.rs`: Semantic versioning for models
- `ab_testing.rs`: Experiment framework
- `personalization.rs`: Privacy-preserving user preferences

## Common Patterns

### Adding a New Model Type
1. Add variant to `ModelType` enum in `src/ai/model_manager.rs`
2. Create integration module in `src/ai/integration.rs`
3. Add training script in `training/scripts/train_*.py`
4. Export to ONNX with matching input/output shapes

### Extending Parsers
Follow the pattern in `src/parser/html.rs`:
- Constructor: `new()` for basic, `with_ai()` for enhanced
- Main method returns `Result<T>` from `anyhow`
- AI enhancement is optional and gracefully degraded

### Cross-Module Communication
- Parsers produce intermediate representations (DOM, CSS rules, AST)
- Renderer consumes these structures via `render()` method
- Learning modules observe via callbacks, don't block main pipeline

## Performance Considerations
- Model inference is CPU-bound (no GPU requirement)
- Use `log::debug!()` for AI operations - disabled in release mode
- Models lazy-loaded on first use via `ModelManager`
- Hot reload supported via `HotReloadManager` in `src/ai/hot_reload.rs`

## Documentation Standards
See examples in `GETTING_STARTED.md`, `training/QUICKSTART.md`:
- Include code examples with imports
- Show both CLI commands and Rust API usage
- Reference specific files using relative paths
