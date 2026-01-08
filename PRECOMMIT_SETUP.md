# BrowerAI Pre-commit Scripts Usage Guide

## Quick Start

```bash
# 1. Setup git hooks (one-time setup)
git config core.hooksPath .githooks

# 2. Make scripts executable
chmod +x scripts/pre-commit.sh scripts/pre-commit-quick.sh

# 3. Now git will automatically run pre-commit checks before committing
git commit -m "Your commit message"
```

## Available Scripts

### 1. Full Pre-commit Checks
**File:** `scripts/pre-commit.sh`

Runs **all** comprehensive checks in sequence:
- ✅ Format validation (rustfmt)
- ✅ Code quality (clippy, 2 passes)
- ✅ Dependencies & licenses (cargo-deny)
- ✅ Build validation (default & all features)
- ✅ Testing (unit, integration, doc tests)
- ✅ Documentation generation
- ✅ Code coverage reporting
- ✅ Security auditing (cargo-audit)

**When to use:**
- Before pushing to GitHub
- Final validation before merge
- CI/CD pipeline

**Execution time:** 90-160 minutes

**Usage:**
```bash
bash scripts/pre-commit.sh
```

### 2. Quick Pre-commit Checks
**File:** `scripts/pre-commit-quick.sh`

Runs **lightweight** checks for rapid iteration:
- ✅ Format validation (rustfmt)
- ✅ Quick linting (clippy on lib/bins only)
- ✅ Dependencies & licenses (cargo-deny)
- ✅ Parse check (cargo check)
- ✅ Security audit (critical/high only)

**When to use:**
- During active development
- Before local commits
- Rapid iteration cycles

**Execution time:** 2-5 minutes

**Usage:**
```bash
bash scripts/pre-commit-quick.sh
```

## Configuration

### Git Hook Automatic Execution

Once git hooks are configured, pre-commit checks run **automatically** before each commit:

```bash
# Setup (one-time)
git config core.hooksPath .githooks

# Now this will automatically run pre-commit.sh
git commit -m "My changes"
```

### Environment Variables

**Skip all checks (emergency only):**
```bash
SKIP_PRECOMMIT=1 git commit -m "Emergency fix"
```

**Skip security audit (not recommended):**
```bash
SKIP_AUDIT=1 bash scripts/pre-commit.sh
```

## What Each Check Does

### Format Check
**Tool:** `cargo fmt`

Validates Rust code is properly formatted.

**Fixes:** `cargo fmt --all`

### Clippy Linting
**Tool:** `cargo clippy`

Catches common Rust programming mistakes and suggests improvements.

**Fixes:** Review clippy suggestions and apply

### Dependency & License Checks
**Tool:** `cargo deny`

Validates:
- No security vulnerabilities in dependencies
- All dependencies have approved licenses
- No duplicate versions
- Trusted sources only

**Config:** `deny.toml`

**Fixes:** Update `deny.toml` or upgrade dependencies

### Build Validation
**Tool:** `cargo build`

Ensures code compiles for:
- Default features
- All features enabled

**Fixes:** Fix compilation errors

### Testing
**Tool:** `cargo test`

Runs:
- Unit tests
- Integration tests
- Documentation tests
- All feature combinations

**Expected:** ~456 tests passing

**Fixes:** Fix test failures

### Documentation
**Tool:** `cargo doc`

Generates HTML documentation and validates:
- No documentation warnings
- Code examples compile and run

**Fixes:** Add doc comments or fix examples

### Code Coverage
**Tool:** `cargo llvm-cov`

Generates code coverage metrics in `codecov.json` format.

**Output:** Ready for codecov.io upload

### Security Audit
**Tool:** `cargo audit`

Checks for known vulnerabilities in all dependencies.

**Config:** `deny.toml` (advisories section)

**Fixes:** Run `cargo audit` for details and upgrade dependencies

## Troubleshooting

### Issue: "Permission denied" running scripts
```bash
# Make scripts executable
chmod +x scripts/pre-commit.sh scripts/pre-commit-quick.sh
```

### Issue: Pre-commit doesn't run automatically
```bash
# Verify git hooks are configured
git config core.hooksPath
# Should output: .githooks

# If not configured:
git config core.hooksPath .githooks

# Verify hook file is executable
ls -la .githooks/pre-commit
# Should show: -rwxr-xr-x
chmod +x .githooks/pre-commit
```

### Issue: "cargo-llvm-cov not found"
```bash
# Install it
cargo install cargo-llvm-cov
```

### Issue: "cargo deny check all" is slow
```bash
# Use offline mode if databases are cached
cargo deny check all --offline
```

### Issue: Tests fail with "not enough parallelism"
```bash
# Run tests serially
cargo test --workspace -- --test-threads=1
```

## Continuous Integration

The full pre-commit script is also run in GitHub Actions CI to ensure consistency between local and remote validation. See `.github/workflows/` for CI configuration.

## Best Practices

1. **Run quick checks during development**
   ```bash
   bash scripts/pre-commit-quick.sh
   ```

2. **Run full checks before pushing**
   ```bash
   bash scripts/pre-commit.sh
   ```

3. **Fix issues incrementally**
   - Fix format: `cargo fmt --all`
   - Fix linting: `cargo clippy --all-features --fix`
   - Fix tests: `cargo test --workspace`

4. **Use git hooks to prevent bad commits**
   ```bash
   git config core.hooksPath .githooks
   ```

5. **Review security vulnerabilities**
   ```bash
   cargo audit
   cargo deny check advisories
   ```

## Documentation

For detailed information about each check, see:
- [PRE_COMMIT_CHECKS.md](./docs/PRE_COMMIT_CHECKS.md) - Comprehensive check documentation

## Support

- **Rust Issues:** Check `cargo build` or `cargo test` output
- **Linting Issues:** See clippy suggestions
- **Dependency Issues:** Run `cargo deny check all` for details
- **Security Issues:** Run `cargo audit` for remediation steps

---

**Last Updated:** 2026-01-08
**Rust Edition:** 2021
**Total Tests:** 456+
**Code Coverage Tool:** cargo-llvm-cov (LLVM-based)
