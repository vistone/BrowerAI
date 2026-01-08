#!/usr/bin/env bash
# Pre-commit gate for BrowerAI
# Comprehensive checks: formatting, linting, security, documentation, testing, and code coverage
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

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color

# Nice, readable headers
section() { echo -e "\n${BLUE}[pre-commit]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }

# Track failed checks
FAILED_CHECKS=()

# Function to run a check and track failures
run_check() {
  local check_name="$1"
  local check_cmd="$2"
  
  section "$check_name"
  if eval "$check_cmd"; then
    success "$check_name passed"
  else
    error "$check_name failed"
    FAILED_CHECKS+=("$check_name")
    return 1
  fi
}

# Ensure required toolchain components are installed
section "Ensuring Rust toolchain components are installed"
if ! rustup component list | grep -q "rustfmt .* (installed)"; then
  rustup component add rustfmt
fi
if ! rustup component list | grep -q "clippy .* (installed)"; then
  rustup component add clippy
fi

# Install required cargo tools
for tool in cargo-llvm-cov cargo-audit cargo-deny; do
  if ! command -v "$tool" &> /dev/null; then
    echo "[pre-commit] Installing $tool..."
    cargo install "$tool"
  fi
done

# Common exclusions for heavy crates
EXCLUDE_FLAGS="--exclude browerai-ml --exclude browerai-js-v8"

# =============================================================================
# 1) FORMAT CHECK
# =============================================================================
run_check "Format check (cargo fmt)" "cargo fmt --all -- --check"

# =============================================================================
# 2) LINT & CODE QUALITY CHECKS
# =============================================================================

# 2a) Clippy (all features) with warnings as errors
run_check "Clippy linting (all-features)" "cargo clippy --workspace --all-features -- -D warnings"

# 2b) Clippy (default features)
run_check "Clippy linting (default)" "cargo clippy --workspace -- -D warnings"

# =============================================================================
# 3) DEPENDENCY & LICENSE CHECKS
# =============================================================================

# 3a) Deny - check licenses, advisories, and supply chain security
run_check "Dependency & license check (cargo-deny)" "cargo deny check all"

# =============================================================================
# 4) BUILD CHECKS
# =============================================================================

# 4a) Build without heavy crates (default features)
run_check "Build (default features)" "cargo build --workspace $EXCLUDE_FLAGS"

# 4b) Build with all features
run_check "Build (all features)" "cargo build --workspace $EXCLUDE_FLAGS --all-features"

# 4c) Check (faster than build, still validates)
run_check "Check (workspace, all-features)" "cargo check --workspace --all-features $EXCLUDE_FLAGS"

# =============================================================================
# 5) TESTING
# =============================================================================

# 5a) Unit and integration tests
run_check "Unit & integration tests" "cargo test --workspace $EXCLUDE_FLAGS"

# 5b) Doc tests
run_check "Documentation tests" "cargo test --doc --workspace $EXCLUDE_FLAGS"

# 5c) Test with all features
run_check "Tests (all-features)" "cargo test --workspace $EXCLUDE_FLAGS --all-features"

# =============================================================================
# 6) DOCUMENTATION
# =============================================================================

run_check "Documentation build" "RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features --workspace $EXCLUDE_FLAGS"

# =============================================================================
# 7) CODE COVERAGE
# =============================================================================

section "Generating code coverage report"
if cargo llvm-cov --all-features --workspace $EXCLUDE_FLAGS --codecov --output-path codecov.json > /tmp/coverage.log 2>&1; then
  success "Code coverage report generated: codecov.json"
else
  warning "Code coverage generation had issues (non-fatal):"
  tail -20 /tmp/coverage.log || true
fi
rm -f /tmp/coverage.log

# =============================================================================
# 8) SECURITY AUDITS
# =============================================================================

# 8a) Cargo audit
section "Running security audit (cargo-audit)"
if cargo audit --json > /tmp/audit_report.json 2>&1; then
  success "No vulnerabilities found (cargo-audit)"
else
  # Parse and display vulnerabilities
  if command -v jq &> /dev/null; then
    WARNING_COUNT=$(jq '.vulnerabilities.list | length' /tmp/audit_report.json 2>/dev/null || echo "0")
    if [[ "$WARNING_COUNT" -gt 0 ]]; then
      echo ""
      warning "Found $WARNING_COUNT security issues:"
      jq -r '.vulnerabilities.list[] | "  - [\(.advisory.id)] \(.advisory.title) (severity: \(.advisory.severity))"' /tmp/audit_report.json
    fi
  else
    warning "jq not found; showing raw audit output:"
    cat /tmp/audit_report.json
  fi
  
  # Check for critical/high vulnerabilities
  CRITICAL_COUNT=$(jq '[.vulnerabilities.list[] | select(.advisory.severity == "critical" or .advisory.severity == "high")] | length' /tmp/audit_report.json 2>/dev/null || echo "0")
  
  if [[ "$CRITICAL_COUNT" -gt 0 ]]; then
    echo ""
    error "Found $CRITICAL_COUNT critical/high severity vulnerability/vulnerabilities"
    echo "Run 'cargo audit' for details and remediation steps"
    echo "To skip this check (not recommended), set: SKIP_AUDIT=1"
    
    if [[ "${SKIP_AUDIT:-0}" != "1" ]]; then
      rm -f /tmp/audit_report.json
      FAILED_CHECKS+=("Security audit (critical/high vulnerabilities)")
    else
      warning "SKIP_AUDIT=1 set; continuing despite critical vulnerabilities (⚠️  NOT RECOMMENDED)"
    fi
  else
    success "No critical/high severity vulnerabilities found"
  fi
fi
rm -f /tmp/audit_report.json

# =============================================================================
# SUMMARY & FINAL STATUS
# =============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ ${#FAILED_CHECKS[@]} -eq 0 ]]; then
  echo -e "${GREEN}✅ All checks passed! Ready to commit.${NC}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 0
else
  echo -e "${RED}❌ The following checks failed:${NC}"
  for check in "${FAILED_CHECKS[@]}"; do
    echo -e "  ${RED}✗${NC} $check"
  done
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 1
fi
