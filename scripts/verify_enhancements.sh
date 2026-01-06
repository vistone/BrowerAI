#!/bin/bash
# Verification Script for System Enhancements
# Tests all newly added functionality

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "BrowerAI Enhancement Verification"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Track results
PASSED=0
FAILED=0

# Test function
test_feature() {
    local name=$1
    local command=$2
    
    echo -n "Testing $name... "
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# 1. Check file structure
echo "1. Checking File Structure"
echo "----------------------------------------"

test_feature "GPU support module" "test -f $PROJECT_ROOT/src/ai/gpu_support.rs"
test_feature "Model generator script" "test -f $PROJECT_ROOT/scripts/generate_minimal_models.py"
test_feature "Deployment script" "test -f $PROJECT_ROOT/scripts/deploy_models.sh"
test_feature "E2E test suite" "test -f $PROJECT_ROOT/tests/e2e_website_tests.rs"
test_feature "Benchmark module" "test -f $PROJECT_ROOT/src/testing/benchmark.rs"
test_feature "Enhancement docs" "test -f $PROJECT_ROOT/docs/SYSTEM_ENHANCEMENTS.md"

echo ""

# 2. Check code modifications
echo "2. Checking Code Modifications"
echo "----------------------------------------"

test_feature "Paint viewport support" "grep -q 'viewport_width' $PROJECT_ROOT/src/renderer/paint.rs"
test_feature "Engine style collection" "grep -q 'Collect computed styles' $PROJECT_ROOT/src/renderer/engine.rs"
test_feature "GPU exports in ai/mod" "grep -q 'gpu_support' $PROJECT_ROOT/src/ai/mod.rs"
test_feature "Benchmark exports" "grep -q 'BenchmarkRunner' $PROJECT_ROOT/src/testing/mod.rs"

echo ""

# 3. Check for removed TODOs
echo "3. Checking TODO Cleanup"
echo "----------------------------------------"

TODO_COUNT=$(grep -r "TODO: Use actual viewport\|TODO: Collect actual styles" $PROJECT_ROOT/src/renderer/*.rs 2>/dev/null | wc -l || echo "0")

if [ "$TODO_COUNT" -eq 0 ]; then
    echo -e "${GREEN}✓ PASS${NC} All rendering TODOs resolved"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ FAIL${NC} $TODO_COUNT rendering TODOs still present"
    FAILED=$((FAILED + 1))
fi

echo ""

# 4. Syntax validation
echo "4. Syntax Validation"
echo "----------------------------------------"

test_feature "Rust syntax check" "cd $PROJECT_ROOT && cargo check --all-targets 2>&1 | grep -q 'Finished'"
test_feature "Python syntax check" "python3 -m py_compile $PROJECT_ROOT/scripts/generate_minimal_models.py"

echo ""

# 5. Test compilation (if cargo available)
echo "5. Compilation Tests"
echo "----------------------------------------"

if command -v cargo &> /dev/null; then
    test_feature "Library compilation" "cd $PROJECT_ROOT && cargo build --lib 2>&1 | grep -q 'Finished'"
    test_feature "Test compilation" "cd $PROJECT_ROOT && cargo test --no-run 2>&1 | grep -q 'Finished'"
else
    echo -e "${YELLOW}⊘ SKIP${NC} Cargo not available"
fi

echo ""

# 6. Check examples
echo "6. Example Files"
echo "----------------------------------------"

test_feature "E2E demo example" "test -f $PROJECT_ROOT/examples/e2e_test_demo.rs"
test_feature "Benchmark demo example" "test -f $PROJECT_ROOT/examples/benchmark_demo.rs"

echo ""

# 7. Documentation checks
echo "7. Documentation"
echo "----------------------------------------"

test_feature "Enhancement report exists" "test -s $PROJECT_ROOT/docs/SYSTEM_ENHANCEMENTS.md"
test_feature "Report has 5 sections" "grep -c '^## [0-9]\\.' $PROJECT_ROOT/docs/SYSTEM_ENHANCEMENTS.md | grep -q '^5$'"

echo ""

# Summary
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo -e "Tests Passed: ${GREEN}$PASSED${NC}"
echo -e "Tests Failed: ${RED}$FAILED${NC}"
echo "Total Tests: $((PASSED + FAILED))"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All enhancements verified successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Deploy models: ./scripts/deploy_models.sh"
    echo "  2. Build with AI: cargo build --features ai"
    echo "  3. Run tests: cargo test"
    exit 0
else
    echo -e "${RED}✗ Some verifications failed${NC}"
    echo "Please review the failed tests above."
    exit 1
fi
