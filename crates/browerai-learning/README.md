# browerai-learning

Learning System for BrowerAI

## Overview

Continuous learning system with feedback collection, code generation, and model improvement. Now includes a comprehensive **Framework Knowledge Base** for detecting and analyzing JavaScript frameworks worldwide.

## Features

### 🆕 Framework Knowledge Base
Comprehensive detection system for **30+ JavaScript frameworks** worldwide:
- **Global Frameworks**: React, Vue, Angular, Next.js, Nuxt.js, Webpack, Vite, Redux, MobX, Material-UI, Svelte, Preact, Solid.js, Alpine.js, Lit, Gatsby, SvelteKit, Zustand, Pinia
- **Chinese Frameworks**: Taro (京东), Uni-app (DCloud), Rax (阿里), San (百度), Omi (腾讯), Qiankun (阿里), Element UI (饿了么), Vant (有赞)
- **Obfuscators**: javascript-obfuscator, Terser/UglifyJS

**Coverage**:
- 40 detection signatures with pattern matching
- 20 obfuscation patterns with before/after examples
- 17 deobfuscation strategies with success rates
- 7 categories: Bundler, Frontend, Meta, Mobile, State, UI, Obfuscator

See [FRAMEWORK_KNOWLEDGE.md](./FRAMEWORK_KNOWLEDGE.md) for detailed documentation.

### JavaScript Deobfuscation
- User feedback collection
- Online learning and model fine-tuning
- Code generator (HTML/CSS/JS with constraints)
- Multi-strategy JavaScript deobfuscation
- Model versioning (semantic versioning)
- Metrics dashboard for monitoring
- Privacy-preserving personalization

## Usage

```rust
use browerai_learning::{FrameworkKnowledgeBase, FrameworkCategory};

// Initialize framework knowledge base
let kb = FrameworkKnowledgeBase::new();

// Analyze JavaScript code
let code = r#"
    import React from 'react';
    function App() {
        return React.createElement("div", null, "Hello");
    }
"#;

let results = kb.analyze_code(code)?;
for result in results {
    println!("Detected: {} (confidence: {:.1}%)", 
        result.framework_name, 
        result.confidence * 100.0
    );
}

// Get statistics
let stats = kb.get_statistics();
println!("Total frameworks: {}", stats.total_frameworks);
```

## Examples

Run the framework knowledge demo:
```bash
cargo run --package browerai --example framework_knowledge_demo
```

## Tests

Run tests with:

```bash
cargo test -p browerai-learning
```

All **131 tests** passing, including 8 new tests for framework knowledge.

## Documentation

- [FRAMEWORK_KNOWLEDGE.md](./FRAMEWORK_KNOWLEDGE.md) - Detailed framework detection documentation
- [BrowerAI Architecture](../../docs/ARCHITECTURE.md) - Overall architecture

## Dependencies

See [Cargo.toml](./Cargo.toml) for full dependency list.

