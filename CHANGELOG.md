# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- SECURITY.md with security policy and best practices
- CONTRIBUTING.md with comprehensive contribution guidelines
- .gitattributes for better cross-platform Git handling
- Workspace architecture with 18 specialized crates
- Feature flags for optional dependencies (ai, ai-candle, ml)
- Comprehensive test suite (459+ tests)
- Code quality improvements with clippy
- Formatted codebase with rustfmt

### Changed
- Made browerai-ml crate optional (requires `ml` feature flag)
- Updated README.md with detailed project information
- Updated TODO.md to reflect 95% completion status
- Improved error handling patterns throughout codebase
- Enhanced documentation structure

### Fixed
- Critical build issue with torch-sys TLS certificate download
- Test compilation errors across multiple crates
- CssParser missing methods (validate, is_ai_enabled)
- Test imports in DOM, renderer, and JS analyzer crates
- Candle example compilation with proper feature gating
- All clippy warnings and errors
- Code formatting inconsistencies

### Security
- No unsafe code in the codebase
- All TLS operations use rustls instead of OpenSSL
- JavaScript execution properly sandboxed
- Input validation in all parsers

## [0.1.0] - 2026-01-06

### Added
- Initial workspace structure
- Core browser engine components
- HTML, CSS, and JavaScript parsers
- AI integration layer (optional)
- ML toolkit integration (optional)
- Rendering engines
- Learning system
- Network utilities and crawler
- Developer tools
- Plugin system
- Comprehensive testing infrastructure

### Infrastructure
- Modular workspace with 18 crates
- CI/CD ready structure
- Documentation system
- Example programs
- Training pipeline for ML models

[Unreleased]: https://github.com/vistone/BrowerAI/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vistone/BrowerAI/releases/tag/v0.1.0
