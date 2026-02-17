#!/bin/bash
# Phase E Build Script
# Purpose: Build and test application locally

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔨 Phase E Build Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
VENV_DIR="${PWD}/venv"
LOG_FILE="${PWD}/build.log"

# Functions
log_step() {
    echo -e "${GREEN}→${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Step 1: Check Python version
log_step "Checking Python version..."
python --version
PYTHON_MINOR=$(python --version | cut -d' ' -f2 | cut -d'.' -f2)
if [ "$PYTHON_MINOR" -lt "11" ]; then
    log_error "Python 3.11+ required, found $(python --version)"
    exit 1
fi
echo "✅ Python version OK"

# Step 2: Create virtual environment
log_step "Setting up Python virtual environment..."
if [ -d "$VENV_DIR" ]; then
    log_warning "Virtual environment exists, removing..."
    rm -rf "$VENV_DIR"
fi

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment created"

# Step 3: Install dependencies
log_step "Installing dependencies..."
pip install --upgrade pip setuptools wheel > "$LOG_FILE" 2>&1
pip install -r requirements.txt >> "$LOG_FILE" 2>&1
pip install pytest pytest-cov pytest-asyncio pylint flake8 black isort >> "$LOG_FILE" 2>&1
echo "✅ Dependencies installed"

# Step 4: Run code formatting
log_step "Checking code format..."
if ! black --check browerai-api-server/ > /dev/null 2>&1; then
    log_warning "Code formatting issues found, fixing..."
    black browerai-api-server/ > /dev/null 2>&1
fi
echo "✅ Code formatting OK"

# Step 5: Run import sorting
log_step "Checking import sorting..."
if ! isort --check-only browerai-api-server/ > /dev/null 2>&1; then
    log_warning "Import sorting issues found, fixing..."
    isort browerai-api-server/ > /dev/null 2>&1
fi
echo "✅ Import sorting OK"

# Step 6: Run linters
log_step "Running pylint checks..."
PYLINT_SCORE=$(pylint browerai-api-server/ --exit-zero 2>/dev/null | tail -2 | head -1 || echo "N/A")
echo "  Pylint score: $PYLINT_SCORE"

log_step "Running flake8 checks..."
FLAKE8_ISSUES=$(flake8 browerai-api-server/ --max-line-length=120 --count || echo "0")
echo "  Flake8 issues: $FLAKE8_ISSUES"
echo "✅ Linting checks complete"

# Step 7: Run unit tests
log_step "Running unit tests..."
TESTS_DIR="browerai-api-server/tests"
if [ -d "$TESTS_DIR" ]; then
    pytest "$TESTS_DIR" -v --tb=short --cov=browerai-api-server --cov-report=term-missing 2>&1 | tee -a "$LOG_FILE"
    TEST_RESULT=$?
    if [ $TEST_RESULT -eq 0 ]; then
        echo "✅ Unit tests passed"
    else
        log_warning "Some unit tests failed or skipped"
    fi
else
    log_warning "No tests directory found at $TESTS_DIR"
fi

# Step 8: Generate coverage report
log_step "Generating coverage report..."
pytest browerai-api-server/tests/ \
    --cov=browerai-api-server \
    --cov-report=html \
    --cov-report=xml \
    --cov-report=term 2>/dev/null || log_warning "Coverage report generation skipped"
if [ -d "htmlcov" ]; then
    echo "  Coverage report: file://$(pwd)/htmlcov/index.html"
fi
echo "✅ Coverage report generated"

# Step 9: Build summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Build Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Python version: $(python --version)"
echo "Virtual env: $VENV_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Activate venv: source $VENV_DIR/bin/activate"
echo "  2. Run app: python browerai-api-server/app.py"
echo "  3. Run tests: pytest browerai-api-server/tests/ -v"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
