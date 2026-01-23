# BrowerAI Testing Strategy

## Overview
This document outlines the comprehensive testing strategy for BrowerAI, inspired by best practices from leading Rust projects.

## Test Categories

### 1. Unit Tests
**Location**: Inline with code (`#[cfg(test)] mod tests`)

**Coverage Target**: 80%+

**Examples**:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_parser_basic() {
        let parser = HtmlParser::new();
        let result = parser.parse("<html><body>test</body></html>");
        assert!(result.is_ok());
    }
}
```

### 2. Integration Tests
**Location**: `tests/` directory

**Purpose**: Test interactions between crates

**Example Structure**:
```
tests/
├── html_integration_tests.rs
├── css_integration_tests.rs
├── js_integration_tests.rs
├── v8_integration_tests.rs
└── e2e/
    ├── website_rendering.rs
    └── full_pipeline.rs
```

### 3. Property-Based Tests
**Tool**: proptest or quickcheck

**Purpose**: Test properties that should hold for all inputs

**Example**:
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn parse_doesnt_crash(html in ".*") {
        let parser = HtmlParser::new();
        let _ = parser.parse(&html);
        // Should not crash
    }
    
    #[test]
    fn parse_is_deterministic(html in ".*") {
        let parser = HtmlParser::new();
        let result1 = parser.parse(&html);
        let result2 = parser.parse(&html);
        prop_assert_eq!(result1.is_ok(), result2.is_ok());
    }
}
```

### 4. Fuzz Testing
**Tool**: cargo-fuzz

**Purpose**: Find crashes and panics with random inputs

**Setup**:
```bash
# Install cargo-fuzz
cargo install cargo-fuzz

# Create fuzz targets
cargo fuzz init
cargo fuzz add html_parser
cargo fuzz add css_parser
cargo fuzz add js_parser

# Run fuzzing
cargo fuzz run html_parser
```

### 5. Benchmark Tests
**Tool**: criterion

**Purpose**: Track performance over time

**Example**:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_html_parsing(c: &mut Criterion) {
    let html = include_str!("../test_data/large.html");
    let parser = HtmlParser::new();
    
    c.bench_function("parse large html", |b| {
        b.iter(|| parser.parse(black_box(html)))
    });
}

criterion_group!(benches, benchmark_html_parsing);
criterion_main!(benches);
```

### 6. Doc Tests
**Location**: Documentation comments

**Purpose**: Ensure examples in docs work

**Example**:
```rust
/// Parse HTML and return a DOM tree
///
/// # Example
///
/// ```
/// use browerai::HtmlParser;
///
/// let parser = HtmlParser::new();
/// let dom = parser.parse("<html><body>Hello</body></html>")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn parse(&self, html: &str) -> Result<Dom> {
    // Implementation
}
```

## Test Data Organization

```
test_data/
├── html/
│   ├── simple.html
│   ├── complex.html
│   ├── large.html
│   └── malformed.html
├── css/
│   ├── basic.css
│   ├── complex.css
│   └── invalid.css
└── js/
    ├── simple.js
    ├── modern.js
    ├── obfuscated.js
    └── invalid.js
```

## CI Integration

### Test Matrix
- **OS**: Linux, macOS, Windows
- **Rust**: stable, beta, nightly
- **Features**: default, all-features, minimal

### Coverage Requirements
- **Overall**: 80%+
- **Critical paths**: 90%+
- **New code**: 85%+

## Performance Testing

### Benchmark Suite
```toml
[[bench]]
name = "parser_benchmarks"
harness = false

[[bench]]
name = "renderer_benchmarks"
harness = false

[[bench]]
name = "js_execution_benchmarks"
harness = false
```

### Performance Budgets
- HTML parsing: < 1ms for 10KB
- CSS parsing: < 0.5ms for 5KB
- JS execution: < 10ms for simple scripts
- DOM traversal: < 0.1ms for 1000 nodes

## Test Commands

```bash
# Run all tests
just test

# Run with coverage
just test-coverage

# Run benchmarks
just bench

# Run specific crate
just test-crate browerai-html-parser

# Run with output
just test-verbose

# Watch and test
just watch
```

## Continuous Improvement

### Monthly Reviews
- Review test coverage reports
- Identify gaps in testing
- Add tests for edge cases
- Update performance benchmarks

### Quality Gates
- All PRs must pass tests
- Coverage must not decrease
- Benchmarks must not regress
- No clippy warnings

## Resources

- [Rust Testing Book](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Proptest Documentation](https://docs.rs/proptest/)
- [Criterion User Guide](https://bheisler.github.io/criterion.rs/book/)
- [cargo-fuzz Tutorial](https://rust-fuzz.github.io/book/)
