# BrowerAI Developer Commands
# Install Just: cargo install just
# Usage: just <command>

# Default recipe (show help)
default:
    @just --list

# Run all pre-commit checks
check-all: fmt clippy test
    @echo "✅ All checks passed!"

# Format all code
fmt:
    @echo "🎨 Formatting code..."
    cargo fmt --all

# Check formatting without making changes
fmt-check:
    @echo "🔍 Checking code formatting..."
    cargo fmt --all -- --check

# Run clippy linter
clippy:
    @echo "📎 Running clippy..."
    cargo clippy --all-features --workspace --exclude browerai-ml --exclude browerai-js-v8 -- -D warnings

# Run all tests
test:
    @echo "🧪 Running tests..."
    cargo test --workspace --exclude browerai-ml --exclude browerai-js-v8

# Run tests with output
test-verbose:
    @echo "🧪 Running tests with output..."
    cargo test --workspace --exclude browerai-ml --exclude browerai-js-v8 -- --nocapture

# Run specific crate tests
test-crate crate:
    @echo "🧪 Testing {{crate}}..."
    cargo test -p {{crate}}

# Run tests with coverage
test-coverage:
    @echo "📊 Running tests with coverage..."
    cargo llvm-cov --all-features --workspace --exclude browerai-ml --exclude browerai-js-v8 --html
    @echo "📊 Coverage report generated in target/llvm-cov/html/index.html"

# Run benchmarks
bench:
    @echo "⚡ Running benchmarks..."
    cargo bench --workspace --exclude browerai-ml --exclude browerai-js-v8

# Build everything
build:
    @echo "🔨 Building project..."
    cargo build --workspace --exclude browerai-ml --exclude browerai-js-v8

# Build in release mode
build-release:
    @echo "🔨 Building release..."
    cargo build --release --workspace --exclude browerai-ml --exclude browerai-js-v8

# Build with V8 feature
build-v8:
    @echo "🔨 Building with V8..."
    cargo build --features v8 --workspace --exclude browerai-ml

# Build documentation
docs:
    @echo "📚 Building documentation..."
    cargo doc --no-deps --all-features --workspace --exclude browerai-ml --exclude browerai-js-v8 --open

# Check documentation
docs-check:
    @echo "📚 Checking documentation..."
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features --workspace --exclude browerai-ml --exclude browerai-js-v8

# Clean build artifacts
clean:
    @echo "🧹 Cleaning..."
    cargo clean
    rm -rf target/

# Update dependencies
update:
    @echo "🔄 Updating dependencies..."
    cargo update

# Check for outdated dependencies
outdated:
    @echo "🔍 Checking for outdated dependencies..."
    cargo outdated

# Run security audit
audit:
    @echo "🔒 Running security audit..."
    cargo audit

# Fix clippy warnings automatically
fix:
    @echo "🔧 Fixing clippy warnings..."
    cargo clippy --fix --allow-dirty --allow-staged --workspace --exclude browerai-ml --exclude browerai-js-v8

# Run examples
example name:
    @echo "🎯 Running example {{name}}..."
    cargo run --example {{name}}

# Run V8 examples
example-v8 name:
    @echo "🎯 Running V8 example {{name}}..."
    cargo run --example {{name}} --features v8

# Install development dependencies
install-dev:
    @echo "📦 Installing development tools..."
    cargo install cargo-llvm-cov cargo-audit cargo-outdated just

# Quick development cycle
dev: fmt clippy test
    @echo "✅ Development cycle complete!"

# Full CI simulation
ci: fmt-check clippy test docs-check
    @echo "✅ CI checks passed!"

# Watch for changes and run tests
watch:
    @echo "👀 Watching for changes..."
    cargo watch -x "test --workspace --exclude browerai-ml --exclude browerai-js-v8"

# Generate coverage report and open in browser
coverage-report: test-coverage
    @echo "🌐 Opening coverage report..."
    open target/llvm-cov/html/index.html || xdg-open target/llvm-cov/html/index.html

# Count lines of code
loc:
    @echo "📊 Counting lines of code..."
    @find crates -name "*.rs" -type f | xargs wc -l | tail -1

# Show project statistics
stats:
    @echo "📊 Project Statistics:"
    @echo "  Crates: $(find crates -name Cargo.toml | wc -l)"
    @echo "  Rust files: $(find crates -name '*.rs' -type f | wc -l)"
    @echo "  Lines of code:"
    @find crates -name "*.rs" -type f | xargs wc -l | tail -1
    @echo "  Tests:"
    @rg -c "#\[test\]" crates --type rust | awk -F: '{sum+=$2} END {print "    " sum " test functions"}'
