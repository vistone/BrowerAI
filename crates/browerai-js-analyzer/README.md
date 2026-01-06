# browerai-js-analyzer

JavaScript Deep Analyzer for BrowerAI

## Overview

Advanced JavaScript analysis including scope analysis, dataflow tracking, control flow analysis, call graphs, and deobfuscation. Uses a phase-based architecture.

## Features

- **Phase 1**: Scope analyzer - Lexical scope tracking
- **Phase 2**: SWC extractor - TypeScript/JSX support
- **Phase 3 Week 1**: Dataflow analyzer - Variable flow tracking  
- **Phase 3 Week 2**: Control flow analyzer - Branch/loop analysis
- **Phase 3 Week 3**: Enhanced call graph, loop analyzer, performance optimizer
- **Phase 3 Week 3 Task 4**: Unified analysis pipeline

## Usage

```rust
use browerai_js_analyzer::*;

// Example usage
```

## Tests

Run tests with:

```bash
cargo test -p browerai-js-analyzer
```

## Documentation

See the main [BrowerAI documentation](../../docs/ARCHITECTURE.md) for architecture overview.

## Dependencies

See [Cargo.toml](./Cargo.toml) for full dependency list.
