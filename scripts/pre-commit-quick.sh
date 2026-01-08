#!/usr/bin/env bash
# Quick pre-commit checks (skips time-consuming compilations)
# Useful for rapid iteration during development
# Full checks should still be run before final commit with scripts/pre-commit.sh

set -euo pipefail

if [[ "${SKIP_PRECOMMIT:-0}" == "1" ]]; then
  echo "[pre-commit-quick] SKIP_PRECOMMIT=1 set; skipping checks."
  exit 0
fi

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'

section() { echo -e "\n${BLUE}[pre-commit-quick]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }

FAILED_CHECKS=()

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

# =============================================================================
# QUICK CHECKS (fast path for development)
# =============================================================================

# 1) Format check
run_check "Format check" "cargo fmt --all -- --check" || true

# 2) Quick linting (skip all-features to save time)
run_check "Clippy (default features)" "cargo clippy --lib --bins -- -D warnings 2>/dev/null" || true

# 3) Dependency checks
run_check "Dependency & license check" "cargo deny check all --offline || cargo deny check all" || true

# 4) Quick check (doesn't do full compilation)
run_check "Quick check (parsing)" "cargo check --lib --bins 2>/dev/null" || true

# 5) Security audit (always run)
section "Security audit"
if cargo audit --json > /tmp/quick_audit.json 2>&1; then
  success "No vulnerabilities found"
else
  CRITICAL=$(jq '[.vulnerabilities.list[] | select(.advisory.severity == "critical" or .advisory.severity == "high")] | length' /tmp/quick_audit.json 2>/dev/null || echo "0")
  if [[ "$CRITICAL" -gt 0 ]]; then
    error "Found $CRITICAL critical/high severity vulnerabilities"
    jq -r '.vulnerabilities.list[] | select(.advisory.severity == "critical" or .advisory.severity == "high") | "  - [\(.advisory.id)] \(.advisory.title)"' /tmp/quick_audit.json 2>/dev/null || true
    FAILED_CHECKS+=("Security audit (critical/high)")
  else
    success "No critical/high vulnerabilities found"
  fi
fi
rm -f /tmp/quick_audit.json

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ ${#FAILED_CHECKS[@]} -eq 0 ]]; then
  echo -e "${GREEN}✅ Quick checks passed!${NC}"
  echo "For full pre-commit validation before pushing, run:"
  echo "  bash scripts/pre-commit.sh"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 0
else
  echo -e "${RED}❌ The following checks failed:${NC}"
  for check in "${FAILED_CHECKS[@]}"; do
    echo -e "  ${RED}✗${NC} $check"
  done
  echo ""
  echo -e "${YELLOW}Note: Some checks failed. Review errors above.${NC}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 1
fi
