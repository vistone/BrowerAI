# BrowerAI Architecture

## System Overview

BrowerAI is a Rust-based AI-powered browser engine that combines traditional parsing with machine learning models for intelligent HTML/CSS/JS processing.

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                  Browser Engine (Rust)                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Parser     │  │  Renderer    │  │  Learning    │ │
│  │              │  │              │  │              │ │
│  │ • HTML       │  │ • Layout     │  │ • Feedback   │ │
│  │ • CSS        │  │ • Paint      │  │ • Analysis   │ │
│  │ • JS         │  │ • Predictive │  │ • Metrics    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│         │                 │                 │          │
│         └─────────────────┬─────────────────┘          │
│                           │                            │
│  ┌────────────────────────────────────────────────┐   │
│  │         AI Integration Layer (ONNX)            │   │
│  │  • Model Management • Inference • Hot Reload  │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │         Support Systems                        │   │
│  │  • Network • DOM • Testing • Plugins          │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Module Organization

### `src/parser/`
- **html.rs** - HTML5 parsing with html5ever
  - AI enhancement for structure prediction
  - DOM tree generation
  
- **css.rs** - CSS parsing with cssparser
  - Selector compilation
  - Style computation
  
- **js_analyzer/** - JavaScript analysis
  - AST parsing with boa_parser
  - Scope analysis
  - Data flow analysis
  - Deobfuscation

### `src/renderer/`
- **layout.rs** - Layout engine
  - Box model computation
  - Flex/Grid support
  
- **paint.rs** - Rasterization
  - Shape rendering
  - Color/opacity
  
- **predictive.rs** - Predictive rendering
  - Queue management
  - Partial rendering
  
- **ai_regeneration.rs** - AI-powered code simplification
  - Website simplification
  - CSS/HTML optimization

### `src/ai/`
- **model_manager.rs** - Model lifecycle
- **inference.rs** - ONNX runtime wrapper
- **hot_reload.rs** - Dynamic model updates
- **runtime.rs** - Execution monitoring

### `src/learning/`
- **feedback.rs** - User feedback collection
- **online_learning.rs** - Model fine-tuning
- **versioning.rs** - Model versioning

## Data Flow

```
HTML/CSS/JS Input
        │
        ▼
    Parsing Layer
    ├─ HTML Parser
    ├─ CSS Parser
    └─ JS Analyzer
        │
        ▼
    Feature Extraction
    ├─ Structure
    ├─ Styles
    └─ Scripts
        │
        ▼
    AI Enhancement (ONNX)
    ├─ Prediction
    ├─ Optimization
    └─ Generation
        │
        ▼
    Rendering
    ├─ Layout
    ├─ Paint
    └─ Display
        │
        ▼
    Learning Feedback
    ├─ Metrics
    ├─ User Feedback
    └─ Model Updates
```

## Key Technologies

- **Rust** 2021 Edition - Type safety, performance
- **ONNX Runtime** - Cross-platform ML inference
- **html5ever** - W3C-compliant HTML parsing
- **cssparser** - CSS parsing and computation
- **boa_parser** - Pure Rust JavaScript parsing
- **serde** - Serialization/deserialization

## JavaScript Processing Pipeline

### Scope Analysis
- Variable scope tracking (global, function, block)
- Parameter analysis
- Closure variable capture

### Data Flow Analysis
- Definition-use chains
- Unused variable detection
- Constant propagation
- Type inference

### Deobfuscation
- Name recovery
- Dead code elimination
- Control flow simplification
- Pattern-based transformations

## AI Integration Points

### 1. HTML Parsing Enhancement
- Structural prediction for malformed HTML
- DOM optimization hints
- Element importance scoring

### 2. CSS Processing
- Rule optimization
- Selector simplification
- Unused style detection

### 3. JavaScript Analysis
- Variable renaming suggestions
- Dead code identification
- Optimization opportunities

### 4. Rendering Optimization
- Predictive rendering priority
- Layout optimization hints
- Paint operation optimization

## Performance Characteristics

| Operation | Typical Time |
|-----------|-------------|
| Small HTML parse | <1ms |
| CSS computation | <5ms |
| JS analysis | <10ms |
| ONNX inference | 45ms |
| Layout | <20ms |
| Full render | <100ms |

## Testing Strategy

1. **Unit Tests** - Component-level testing
2. **Integration Tests** - Cross-module validation
3. **Benchmark Tests** - Performance verification
4. **E2E Tests** - Full workflow testing

See [COMPREHENSIVE_TESTING.md](COMPREHENSIVE_TESTING.md) for details.

## Model Management

### ONNX Models
- Location: `models/local/`
- Format: `.onnx`
- Configuration: `models/model_config.toml`

### Available Models
- website_learner_v1 - HTML simplification
- (Additional models in development)

See [../training/](../training/) for model training pipeline.

## Quality Metrics

- **Test Coverage**: 100% (459+ tests)
- **Code Quality**: Production-ready
- **Performance**: All targets met
- **Compatibility**: Rust 1.70+

## Workspace Crate Reference

### Core Modules (Always Available)

| Crate | Purpose | Integration Status |
|-------|---------|-------------------|
| `browerai-core` | Core types, errors, traits | ✅ Integrated |
| `browerai-dom` | DOM implementation | ✅ Integrated |
| `browerai-html-parser` | HTML5 parsing | ✅ Integrated |
| `browerai-css-parser` | CSS3 parsing | ✅ Integrated |
| `browerai-js-parser` | JavaScript parsing | ✅ Integrated |
| `browerai-js-analyzer` | JS analysis (7-phase pipeline) | ✅ Integrated |
| `browerai-renderer-core` | Rendering engine | ✅ Integrated |
| `browerai-renderer` | Full rendering pipeline | ✅ Integrated |
| `browerai-renderer-predictive` | Predictive rendering | ✅ Integrated |
| `browerai-devtools` | Developer tools | ✅ Integrated |
| `browerai-network` | HTTP client, crawler | ✅ Integrated |
| `browerai-plugins` | Plugin system | ✅ Integrated |
| `browerai-testing` | Test framework | ✅ Integrated |

### AI Modules (Feature-gated)

| Crate | Feature Flag | Purpose |
|-------|-------------|---------|
| `browerai-ai-core` | `ai` | ONNX model management |
| `browerai-ai-integration` | - | AI integration layer |
| `browerai-intelligent-rendering` | - | AI-powered rendering |
| `browerai-learning` | - | Learning system |
| `browerai-deobfuscation` | - | Code deobfuscation |
| `browerai-ml` | `ml` | ML toolkit |

### Storage & Cache Modules

| Crate | Feature Flag | Purpose |
|-------|-------------|---------|
| `browerai-cache` | - | Basic caching |
| `browerai-multilayer-cache` | - | L1/L2/L3 caching |
| `browerai-persistent-layer` | - | Persistent storage |
| `browerai-db` | `db` | Database support |
| `browerai-redis-integration` | `redis` | Redis caching |

### Optional Modules

| Crate | Feature Flag | Purpose |
|-------|-------------|---------|
| `browerai-js-v8` | `v8` | V8 JavaScript engine |
| `browerai-metrics` | `metrics` | Prometheus metrics |
| `browerai-api-server` | `api-server` | HTTP API server |

### Usage Example

```rust
use browerai::prelude::*;

// Core parsers (always available)
let html_parser = HtmlParser::new();
let css_parser = CssParser::new();
let js_parser = JsParser::new();

// With AI feature enabled
#[cfg(feature = "ai")]
{
    use browerai::ai::AiCore;
    let ai = AiCore::new()?;
}
```

### Feature Flags

Enable optional functionality at compile time:

```toml
[dependencies]
browerai = { version = "0.2.0", features = ["ai", "v8", "db"] }
```

Available features:
- `ai` - ONNX-based AI inference
- `ai-candle` - Candle-based LLM support
- `v8` - V8 JavaScript engine
- `ml` - PyTorch-based ML toolkit
- `metrics` - Prometheus metrics
- `db` - Database support
- `redis` - Redis integration
- `api-server` - HTTP API server

---

**Last Updated**: February 17, 2026
