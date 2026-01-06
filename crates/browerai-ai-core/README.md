# browerai-ai-core

AI Core for BrowerAI

## Overview

Core AI infrastructure including model management, ONNX inference engine, hot reload, GPU support, and performance monitoring.

## Features

- Model lifecycle management (loading, unloading, health checks)
- ONNX Runtime integration (optional via 'onnx' feature)
- Hot reload for model updates
- GPU acceleration support
- Performance monitoring and metrics
- Fallback strategies when models unavailable

## Usage

```rust
use browerai_ai_core::*;

// Example usage
```

## Tests

Run tests with:

```bash
cargo test -p browerai-ai-core
```

## Documentation

See the main [BrowerAI documentation](../../docs/ARCHITECTURE.md) for architecture overview.

## Dependencies

See [Cargo.toml](./Cargo.toml) for full dependency list.
