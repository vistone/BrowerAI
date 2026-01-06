# browerai-core

Core types, traits, and utilities for the BrowerAI browser engine.

## Overview

This crate provides the foundational abstractions used across all BrowerAI components:

- **Common types**: Error types, result types, configuration structures
- **Core traits**: Interfaces for parsers, renderers, and AI integration
- **Utilities**: Logging helpers, serialization utilities

## Features

- Shared error handling with `anyhow::Result`
- Common traits for extensibility
- No heavy dependencies (no HTML/CSS/JS parsing)

## Usage

```rust
use browerai_core::{Result, BrowserError};

fn example() -> Result<()> {
    // Your code here
    Ok(())
}
```

## Dependencies

This crate is the foundation - it has minimal dependencies and is used by all other BrowerAI crates.

## Documentation

See the main [BrowerAI documentation](../../docs/ARCHITECTURE.md) for architecture overview.
