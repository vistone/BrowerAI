# Pre-commit Checks Documentation

## Overview

BrowerAI project uses comprehensive pre-commit checks to ensure code quality, security, and maintainability. There are two levels of checks available:

- **Full Checks** (`scripts/pre-commit.sh`): Comprehensive validation before pushing
- **Quick Checks** (`scripts/pre-commit-quick.sh`): Fast iteration during development

## Full Pre-commit Checks (`scripts/pre-commit.sh`)

Runs **all** necessary validations in the following order:

### 1. Toolchain Setup ✓
- Ensures `rustfmt` is installed
- Ensures `clippy` is installed
- Installs required cargo tools: `cargo-llvm-cov`, `cargo-audit`, `cargo-deny`

### 2. Format Check (1-2 min)
```bash
cargo fmt --all -- --check
```
Validates Rust code formatting according to rustfmt standards.

**What it catches:**
- Inconsistent indentation
- Incorrect spacing around brackets
- Trailing whitespace

**Fix command:** `cargo fmt --all`

---

### 3. Lint & Code Quality (8-15 min)

#### 3a. Clippy (all features) - D warnings
```bash
cargo clippy --workspace --all-features -- -D warnings
```
Lints all crates with all feature combinations enabled.

#### 3b. Clippy (default features) - D warnings
```bash
cargo clippy --workspace -- -D warnings
```
Lints with default features only.

**What it catches:**
- Unused variables
- Incorrect error handling
- Inefficient code patterns
- Logic errors
- API misuse

**Fix:** Review clippy suggestions and apply fixes

---

### 4. Dependency & License Checks (2-5 min)
```bash
cargo deny check all
```
Validates:
- **Advisories**: Known security vulnerabilities in dependencies
- **Licenses**: Ensures all dependencies have approved licenses
- **Bans**: Prevents multiple versions of the same crate
- **Sources**: Validates dependency sources (registries, git repos)

**Configured in:** `deny.toml`

**What it catches:**
- Critical security vulnerabilities
- License incompatibilities
- Deprecated or yanked crates
- Untrusted sources

**Fix:** Update `deny.toml` or upgrade vulnerable dependencies

---

### 5. Build Checks (10-20 min)

#### 5a. Build with default features
```bash
cargo build --workspace --exclude browerai-ml --exclude browerai-js-v8
```

#### 5b. Build with all features
```bash
cargo build --workspace --exclude browerai-ml --exclude browerai-js-v8 --all-features
```

**What it catches:**
- Compilation errors
- Feature gating issues
- Platform-specific code problems
- Linking errors

**Note:** Excludes `browerai-ml` and `browerai-js-v8` due to heavy dependencies

---

### 6. Testing (20-40 min)

#### 6a. Unit & Integration tests
```bash
cargo test --workspace --exclude browerai-ml --exclude browerai-js-v8
```

#### 6b. Documentation tests
```bash
cargo test --doc --workspace --exclude browerai-ml --exclude browerai-js-v8
```

#### 6c. All features test
```bash
cargo test --workspace --exclude browerai-ml --exclude browerai-js-v8 --all-features
```

**What it validates:**
- All unit tests pass
- All integration tests pass
- Documentation examples compile and run
- Code examples in doc comments work correctly

**Total test count:** ~456 tests

---

### 7. Documentation Build (5-10 min)
```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features --workspace --exclude browerai-ml --exclude browerai-js-v8
```

Generates documentation and treats warnings as errors.

**What it catches:**
- Missing documentation on public items
- Broken documentation links
- Invalid markdown in doc comments
- Incorrect code examples in documentation

**Fix:** Add/fix documentation or doc comments

---

### 8. Code Coverage Report (15-30 min)
```bash
cargo llvm-cov --all-features --workspace --exclude browerai-ml --exclude browerai-js-v8 --codecov --output-path codecov.json
```

Generates code coverage metrics in Codecov format.

**Output:** `codecov.json` (ready for CI/CD upload)

**Metrics provided:**
- Line coverage percentage
- Branch coverage percentage
- Function coverage percentage

---

### 9. Security Audit (1-2 min)
```bash
cargo audit --json
```

Scans all dependencies for known security vulnerabilities using the RustSec advisory database.

**Configuration:** `deny.toml` (advisories section)

**What it catches:**
- CVEs in dependencies
- Uncontrolled recursion issues
- Cryptographic weaknesses
- Unsafe code patterns

**Skip mechanism:** `SKIP_AUDIT=1 cargo commit` (not recommended)

---

## Quick Pre-commit Checks (`scripts/pre-commit-quick.sh`)

Optimized for rapid iteration during development. Skips time-consuming compilations.

### Checks included:
1. ✓ Format check (`cargo fmt --check`)
2. ✓ Quick linting (`cargo clippy` - lib/bins only)
3. ✓ Dependency checks (`cargo deny`)
4. ✓ Quick parse check (`cargo check` - lib/bins only)
5. ✓ Security audit (`cargo audit` - critical/high only)

**Execution time:** ~2-5 minutes

**When to use:**
- During active development
- For rapid iteration on code changes
- Before local commits

**Before pushing to GitHub:**
- Always run the **full** `scripts/pre-commit.sh`

---

## Usage

### Setup Git Hooks
```bash
# Make scripts executable
chmod +x scripts/pre-commit.sh scripts/pre-commit-quick.sh

# Configure git to use hooks directory
git config core.hooksPath .githooks

# Create the hook wrapper if it doesn't exist
mkdir -p .githooks
echo '#!/bin/bash
bash scripts/pre-commit.sh' > .githooks/pre-commit
chmod +x .githooks/pre-commit
```

### Manual Execution

**Full validation (before pushing):**
```bash
bash scripts/pre-commit.sh
```

**Quick iteration (during development):**
```bash
bash scripts/pre-commit-quick.sh
```

**Skip all checks (emergency only):**
```bash
SKIP_PRECOMMIT=1 git commit -m "..."
```

**Skip only security audit:**
```bash
SKIP_AUDIT=1 bash scripts/pre-commit.sh
```

---

## Typical Execution Times

| Check | Time | Excluded Crates |
|-------|------|-----------------|
| Format | 1-2 min | None |
| Clippy (all features) | 8-15 min | browerai-ml, browerai-js-v8 |
| Clippy (default) | 3-5 min | browerai-ml, browerai-js-v8 |
| Deny | 2-5 min | None |
| Build (default) | 5-10 min | browerai-ml, browerai-js-v8 |
| Build (all features) | 10-15 min | browerai-ml, browerai-js-v8 |
| Tests | 20-40 min | browerai-ml, browerai-js-v8 |
| Doc tests | 5-10 min | browerai-ml, browerai-js-v8 |
| Documentation | 5-10 min | browerai-ml, browerai-js-v8 |
| Code coverage | 15-30 min | browerai-ml, browerai-js-v8 |
| Audit | 1-2 min | None |
| **TOTAL** | **~90-160 min** | **See above** |

---

## Common Issues & Solutions

### Issue: Clippy warnings fail the check
```
error: ... (warnings used as errors)
```

**Solution:**
```bash
# Review and fix the issue
cargo clippy --workspace --all-features
# Or suppress if it's a known issue:
#[allow(clippy::rule_name)]
```

### Issue: Cargo-deny fails with license violations
```
error: ...unresolved license ...
```

**Solution:** Update `deny.toml` to approve the license or upgrade the dependency

### Issue: Documentation tests fail
```
error: code example does not compile
```

**Solution:** Fix the code example in your doc comments or mark it as ignored:
```rust
/// ```ignore
/// // This example doesn't compile due to ...
/// ```
```

### Issue: Code coverage incomplete
The coverage report may show lower percentages if test cases don't exercise all code paths.

**Solution:** Add tests to improve coverage

### Issue: Security vulnerability found
```
error: Found 1 critical/high severity vulnerability
```

**Solution:**
1. Run `cargo audit` to see details
2. Update the vulnerable dependency
3. If no fix available, update `deny.toml` with the advisory ID

---

## Configuration Files

### `deny.toml`
Controls what `cargo deny check all` validates:
- Ignored advisories
- Allowed licenses
- Version conflict policies
- Source restrictions

### Rust Edition & Features
All checks use Rust Edition 2021 with workspace-wide dependency resolution (resolver = "2")

### Excluded Crates
Some crates are excluded from certain checks due to heavy dependencies:
- `browerai-ml`: Heavy PyTorch bindings
- `browerai-js-v8`: Heavy V8 engine bindings

These are included in CI/CD but excluded from local pre-commit for faster iteration.

---

## CI/CD Integration

The GitHub Actions CI pipeline runs the **full** pre-commit checks for all pull requests and pushes. The scripts ensure local and CI behavior matches.

See `.github/workflows/` for CI configuration.

---

## Maintenance

### Adding New Checks
1. Add the check function to `scripts/pre-commit.sh`
2. Document the check in this file
3. Test with both `pre-commit.sh` and `pre-commit-quick.sh`
4. Update timing estimates

### Updating Tool Versions
When updating cargo tools:
```bash
cargo install --force cargo-audit
cargo install --force cargo-deny
cargo install --force cargo-llvm-cov
```

---

## References

- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [Clippy Lints](https://doc.rust-lang.org/clippy/)
- [cargo-deny](https://embarkstudios.github.io/cargo-deny/)
- [cargo-audit](https://docs.rs/cargo-audit/)
- [RustSec Advisory Database](https://rustsec.org/)
