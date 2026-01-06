# Contributing to BrowerAI

Thank you for your interest in contributing to BrowerAI! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Prerequisites

- Rust 1.92.0 or later
- Cargo
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/vistone/BrowerAI.git
cd BrowerAI

# Build the project (without ML dependencies)
cargo build

# Run tests
cargo test --workspace --exclude browerai-ml

# Format code
cargo fmt --all

# Check code quality
cargo clippy --workspace
```

## 📋 Development Workflow

### 1. Before You Start

- Check existing issues and pull requests
- Discuss major changes in an issue first
- Fork the repository and create a feature branch

### 2. Making Changes

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ...

# Format code
cargo fmt --all

# Run tests
cargo test --workspace --exclude browerai-ml

# Check for issues
cargo clippy --workspace
```

### 3. Code Quality Standards

#### Required Checks

- ✅ All tests must pass
- ✅ Code must be formatted with `cargo fmt`
- ✅ No clippy errors
- ✅ Documentation for public APIs
- ✅ Tests for new functionality

#### Code Style

- Use `anyhow::Result` for error handling
- Prefer explicit types over `impl Trait` in public APIs
- Add doc comments for public functions and structs
- Keep functions small and focused
- Use descriptive variable names

#### Example

```rust
/// Parse HTML content and return a DOM tree
///
/// # Arguments
///
/// * `html` - The HTML content as a string
///
/// # Returns
///
/// Returns a `Result` containing the parsed DOM or an error
///
/// # Example
///
/// ```
/// use browerai::HtmlParser;
///
/// let parser = HtmlParser::new();
/// let dom = parser.parse("<html><body>Hello</body></html>")?;
/// ```
pub fn parse(&self, html: &str) -> Result<Dom> {
    // Implementation
}
```

### 4. Testing

```bash
# Run all tests
cargo test --workspace --exclude browerai-ml

# Run specific crate tests
cargo test -p browerai-js-analyzer

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name
```

#### Test Guidelines

- Write unit tests for new functions
- Add integration tests for cross-crate features
- Test error cases
- Use descriptive test names
- Keep tests focused and independent

### 5. Documentation

- Document all public APIs
- Update README.md if adding features
- Add examples for complex features
- Keep docs up-to-date with code changes

## 🏗️ Project Structure

```
BrowerAI/
├── crates/               # Workspace crates
│   ├── browerai-core/    # Core types
│   ├── browerai-dom/     # DOM implementation
│   ├── browerai-*-parser/  # Parsers
│   ├── browerai-ai-*/    # AI components (optional)
│   ├── browerai-ml/      # ML toolkit (optional)
│   └── ...
├── docs/                 # Documentation
├── examples/             # Example programs
├── tests/                # Integration tests
└── training/             # ML training pipeline
```

## 🔍 Common Tasks

### Adding a New Parser Feature

1. Update the parser in `crates/browerai-*-parser/src/lib.rs`
2. Add tests in the same file or `tests/` directory
3. Update documentation
4. Add examples if needed

### Adding a New Crate

1. Create directory in `crates/`
2. Add `Cargo.toml` with workspace dependencies
3. Update root `Cargo.toml` workspace members
4. Implement functionality
5. Add tests and documentation

### Fixing a Bug

1. Write a failing test that reproduces the bug
2. Fix the bug
3. Verify the test passes
4. Check for similar issues in other crates

## 📦 Workspace Features

The project uses Cargo features for optional functionality:

- `ai` - ONNX-based AI features
- `ai-candle` - Candle-based LLM support
- `ml` - PyTorch-based ML toolkit

When adding dependencies:
- Use workspace dependencies when possible
- Make heavy dependencies optional
- Document feature requirements

## 🐛 Reporting Issues

### Bug Reports

Include:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Rust version)
- Code samples if applicable

### Feature Requests

Include:
- Clear description of the feature
- Use cases and motivation
- Potential implementation approach
- Alternatives considered

## 📝 Pull Request Process

1. **Create PR**: Open a pull request with clear description
2. **CI Checks**: Ensure all CI checks pass
3. **Code Review**: Address review feedback
4. **Merge**: Maintainers will merge when approved

### PR Title Format

```
feat: Add new feature
fix: Fix bug in parser
docs: Update documentation
test: Add tests for feature
refactor: Improve code structure
perf: Optimize performance
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing performed

## Checklist
- [ ] Code formatted with `cargo fmt`
- [ ] Clippy warnings addressed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🤝 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Keep discussions professional

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 💡 Questions?

- Open an issue for questions
- Check existing documentation
- Review similar code in the project

---

**Thank you for contributing to BrowerAI!** 🎉
