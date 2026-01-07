# Security Policy

## Supported Versions

Currently supported versions for security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of BrowerAI seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

1. **DO NOT** open a public issue
2. Email security concerns to the repository maintainers
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Response Time**: We aim to respond within 48 hours
- **Updates**: You'll receive updates on the progress every 5-7 days
- **Disclosure**: We follow coordinated disclosure practices

## Security Best Practices

### Dependencies

- All dependencies use `rustls` instead of OpenSSL for TLS
- Regular dependency audits recommended
- Use `cargo audit` for vulnerability scanning

### Build Configuration

```bash
# Secure build without ML dependencies
cargo build --release

# With AI features (ONNX only, no network dependencies)
cargo build --release --features ai

# With ML features (requires LibTorch, additional security considerations)
cargo build --release --features ml
```

### Code Safety

- **No unsafe code** in the codebase (0 occurrences)
- All parsers use safe Rust implementations
- JavaScript execution in sandboxed environment
- Network requests use secure TLS by default

### Known Considerations

1. **JavaScript Sandbox**: The JS sandbox uses `boa_engine` which is experimental. Review executed code carefully.
2. **Network Requests**: HTTP client validates TLS certificates by default
3. **File Operations**: All file I/O uses safe Rust APIs
4. **ML Models**: ONNX models should be from trusted sources only

## Security Features

- ✅ Memory-safe Rust implementation
- ✅ TLS certificate validation
- ✅ JavaScript sandboxing
- ✅ No unsafe blocks
- ✅ Input validation in parsers
- ✅ Error handling with `anyhow::Result`

## Vulnerability Disclosure Timeline

1. **Day 0**: Vulnerability reported
2. **Day 1-2**: Initial assessment and response
3. **Day 3-7**: Investigation and patch development
4. **Day 8-14**: Testing and validation
5. **Day 15+**: Public disclosure (coordinated with reporter)

## Security Checklist for Contributors

- [ ] Use `cargo clippy` to check for common issues
- [ ] Run `cargo audit` to check dependencies
- [ ] Avoid `unwrap()` in production code paths
- [ ] Validate all user inputs
- [ ] Use `anyhow::Result` for error handling
- [ ] Document security-sensitive functions
- [ ] Add tests for security-related features

## Contact

For security issues, please contact the repository maintainers through GitHub.

---

**Last Updated**: January 6, 2026
