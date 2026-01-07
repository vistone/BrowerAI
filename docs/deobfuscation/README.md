# Enhanced JavaScript Deobfuscation

This directory contains enhanced JavaScript deobfuscation capabilities learned from top GitHub projects.

## Quick Start

```rust
use browerai::learning::EnhancedDeobfuscator;

let mut deobfuscator = EnhancedDeobfuscator::new();
let result = deobfuscator.deobfuscate(obfuscated_code)?;

println!("Deobfuscated: {}", result.code);
println!("Stats: {:?}", result.stats);
```

## Run Demo

```bash
cd crates/browerai
cargo run --example enhanced_js_deobfuscation_demo
```

## Documentation

- **[JS_DEOBFUSCATION_TECHNIQUES.md](../JS_DEOBFUSCATION_TECHNIQUES.md)** - Technical reference
- **[IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md)** - Implementation overview

## Features

✅ String array unpacking
✅ Proxy function removal
✅ Self-defending code removal
✅ Control flow simplification
✅ Constant folding
✅ Dead code elimination
✅ Multi-pass deobfuscation

## Test Coverage

28 unit tests covering all major techniques (all passing)

```bash
cargo test --package browerai-learning deobfus
```

## Research Sources

- [webcrack](https://github.com/j4k0xb/webcrack) (2.3k ⭐)
- [synchrony](https://github.com/relative/synchrony) (500+ ⭐)
- [decode-js](https://github.com/echo094/decode-js)
- [javascript-deobfuscator](https://github.com/ben-sb/javascript-deobfuscator)
