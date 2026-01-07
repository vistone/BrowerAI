#!/usr/bin/env bash
# Pre-commit gate for BrowerAI
# Blocks commits unless all local checks pass.
# Usage: git config core.hooksPath .githooks
#        (the hook wrapper will call this script)
# Skip with SKIP_PRECOMMIT=1 if absolutely necessary.

set -euo pipefail

if [[ "${SKIP_PRECOMMIT:-0}" == "1" ]]; then
  echo "[pre-commit] SKIP_PRECOMMIT=1 set; skipping checks (not recommended)."
  exit 0
fi

# Move to repo root
ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

# Nice, readable headers
section() { echo -e "\n\033[1;34m[pre-commit]\033[0m $*"; }

# Ensure required toolchain components are installed
section "Ensuring Rust toolchain components are installed"
if ! rustup component list | grep -q "rustfmt .* (installed)"; then
  rustup component add rustfmt
fi
if ! rustup component list | grep -q "clippy .* (installed)"; then
  rustup component add clippy
fi

# 1) Format check
section "Running cargo fmt --check"
cargo fmt --all -- --check

# 2) Clippy (all features) with warnings as errors
section "Running cargo clippy (workspace, all-features, -D warnings)"
cargo clippy --workspace --all-features -- -D warnings

# 3) Build without heavy crates (mirror CI)
section "Building (workspace, exclude browerai-ml and browerai-js-v8)"
cargo build --workspace --exclude browerai-ml --exclude browerai-js-v8

# 4) Build with all features for the rest (mirror CI)
section "Building with all features (exclude browerai-ml and browerai-js-v8)"
cargo build --workspace --exclude browerai-ml --exclude browerai-js-v8 --all-features

# 5) Tests (exclude heavy crates as per CI)
section "Running cargo test (workspace, exclude browerai-ml and browerai-js-v8)"
cargo test --workspace --exclude browerai-ml --exclude browerai-js-v8

# 6) Docs with warnings denied (same as CI docs job)
section "Building docs with warnings denied"
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features --workspace --exclude browerai-ml --exclude browerai-js-v8

section "All checks passed. Ready to commit ✅"
exit 0
