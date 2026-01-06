# browerai-dom

Document Object Model (DOM) implementation for BrowerAI.

## Overview

This crate provides:

- **DOM tree structures**: `Document`, `Element`, `Node` types
- **DOM manipulation API**: Query, traverse, and modify the DOM
- **JavaScript sandbox**: Safe execution environment for JS code
- **Event handling**: DOM event system

## Features

- Standards-compliant DOM API
- JavaScript integration via Boa engine
- Memory-efficient tree representation
- Thread-safe DOM access

## Usage

```rust
use browerai_dom::{Document, DomElement};

let mut doc = Document::new();
let element = doc.create_element("div");
element.set_attribute("id", "container");
```

## Architecture

```
Document
 ├─ DomNode (tree structure)
 ├─ JsSandbox (JS execution)
 └─ DomApiExtensions (convenience methods)
```

## Dependencies

- `browerai-core`: Core types and traits
- `boa_engine`: JavaScript execution engine

## Tests

Run tests with:

```bash
cargo test -p browerai-dom
```

## Documentation

See the main [BrowerAI documentation](../../docs/ARCHITECTURE.md) for architecture overview.
