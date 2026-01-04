# Contributing to BrowerAI

Thank you for your interest in contributing to BrowerAI! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/BrowerAI.git`
3. Add upstream remote: `git remote add upstream https://github.com/vistone/BrowerAI.git`
4. Create a branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- Rust 1.70 or later
- Git
- A code editor (VS Code, IntelliJ IDEA, or similar)

### Building the Project

```bash
# Standard build
cargo build

# Release build
cargo build --release

# Build with AI features
cargo build --features ai
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with output
cargo test -- --nocapture
```

### Code Formatting

```bash
# Format your code
cargo fmt

# Check formatting
cargo fmt -- --check
```

### Linting

```bash
# Run clippy
cargo clippy

# Fix clippy warnings automatically where possible
cargo clippy --fix
```

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported
2. Create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs. actual behavior
   - System information (OS, Rust version)
   - Code samples if applicable

### Suggesting Features

1. Check if the feature has been suggested
2. Open an issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternative solutions considered
   - Any implementation ideas

### Code Contributions

1. Pick an issue or propose a new feature
2. Discuss your approach in the issue
3. Write your code
4. Add tests
5. Update documentation
6. Submit a pull request

## Coding Standards

### Rust Style

- Follow the [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/)
- Use `cargo fmt` to format code
- Address all `cargo clippy` warnings

### Documentation

- Document all public APIs with doc comments
- Include examples in documentation
- Update README.md if adding major features

### Code Organization

```rust
// 1. Imports
use std::path::Path;
use anyhow::Result;

// 2. Type definitions
pub struct MyStruct {
    field: String,
}

// 3. Implementation
impl MyStruct {
    /// Create a new instance
    pub fn new(field: String) -> Self {
        Self { field }
    }
    
    /// Get the field value
    pub fn field(&self) -> &str {
        &self.field
    }
}

// 4. Tests
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new() {
        let s = MyStruct::new("test".to_string());
        assert_eq!(s.field(), "test");
    }
}
```

### Error Handling

- Use `Result<T, Error>` for fallible functions
- Use `anyhow::Result` for application code
- Use `thiserror` for library errors
- Provide context with `.context()`

```rust
use anyhow::{Context, Result};

pub fn read_config(path: &Path) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .context("Failed to read config file")?;
    
    toml::from_str(&content)
        .context("Failed to parse config")
}
```

## Testing

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        let result = function_to_test();
        assert_eq!(result, expected_value);
    }

    #[test]
    fn test_error_case() {
        let result = function_that_fails();
        assert!(result.is_err());
    }
}
```

### Test Coverage

- Write tests for all new functionality
- Include edge cases
- Test error conditions
- Aim for comprehensive coverage

### Running Specific Tests

```bash
# Run tests for a module
cargo test parser::html

# Run a specific test
cargo test test_parse_simple_html

# Run tests with logging
RUST_LOG=debug cargo test
```

## Pull Request Process

### Before Submitting

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Code is formatted (`cargo fmt`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Documentation is updated
- [ ] Commit messages are clear

### PR Description

Include:
- What the PR does
- Why the change is needed
- How it was implemented
- Any breaking changes
- Related issues

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address review comments
4. Squash commits if requested
5. Maintainer will merge when approved

## Areas for Contribution

### High Priority

1. **Model Training**: Create ONNX models for parsers
2. **Parser Improvements**: Enhance HTML/CSS/JS parsing
3. **Performance**: Optimize critical paths
4. **Documentation**: Examples and guides

### Medium Priority

1. **Testing**: Increase test coverage
2. **Error Messages**: Improve error reporting
3. **Logging**: Better debugging information
4. **Examples**: More usage examples

### Future Work

1. **Rendering**: Complete rendering engine
2. **Layout**: CSS layout algorithms
3. **JavaScript**: Full JS execution
4. **Networking**: HTTP client integration

## Specific Contribution Guidelines

### Adding a New Parser Feature

1. Identify the HTML/CSS/JS feature to support
2. Add implementation in the appropriate parser file
3. Write comprehensive tests
4. Document the feature
5. Update examples if relevant

Example:
```rust
// src/parser/html.rs

/// Extract all links from the document
pub fn extract_links(&self, dom: &RcDom) -> Vec<String> {
    let mut links = Vec::new();
    self.collect_links(&dom.document, &mut links);
    links
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_extract_links() {
        let parser = HtmlParser::new();
        let html = r#"<a href="http://example.com">Link</a>"#;
        let dom = parser.parse(html).unwrap();
        let links = parser.extract_links(&dom);
        assert_eq!(links.len(), 1);
    }
}
```

### Contributing Models

1. Train your model using PyTorch/TensorFlow
2. Export to ONNX format
3. Test with ONNX Runtime
4. Create model config entry
5. Document model purpose and usage
6. Share training code (optional but appreciated)

### Improving Documentation

- Fix typos and grammar
- Add clarifying examples
- Expand on technical details
- Create tutorials
- Update outdated information

## Community

- Be patient with reviewers
- Help others in issues and discussions
- Share your models and findings
- Participate in architectural discussions

## Questions?

- Open a discussion on GitHub
- Ask in pull request comments
- Check existing issues and documentation

Thank you for contributing to BrowerAI! 🎉
