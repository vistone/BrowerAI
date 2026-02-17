#!/bin/bash
# Week 8 Phase A Startup Script
# Real HTTP Communication Integration Tests

set -e

WORKSPACE_ROOT="/home/stone/BrowerAI"
VENV_PATH="${WORKSPACE_ROOT}/venv_test"
PYTHON="${VENV_PATH}/bin/python"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║              Week 8 Phase A - Real HTTP Communication Setup                ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

# Activate venv
echo "📦 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Install/upgrade dependencies
echo "📥 Ensuring all dependencies are installed..."
pip install --quiet flask flask-cors numpy requests psutil pydantic werkzeug 2>/dev/null || true

echo ""
echo "📊 Phase A Deliverables:"
echo "  ✓ Real HTTP Client Module:       training/http_client.py"
echo "  ✓ Integration Test Runner:       training/real_http_integration_tests.py"
echo "  ✓ Phase A Plan:                  WEEK8_PHASE_A_PLAN.md"
echo ""

echo "🚀 How to proceed:"
echo ""
echo "  1. Start the API server (in another terminal):"
echo "     $ cd $WORKSPACE_ROOT/training"
echo "     $ source ../venv_test/bin/activate"
echo "     $ python api_server.py"
echo ""
echo "  2. Run real HTTP integration tests:"
echo "     $ cd $WORKSPACE_ROOT/training"
echo "     $ source ../venv_test/bin/activate"
echo "     $ python real_http_integration_tests.py"
echo ""
echo "  3. Monitor performance:"
echo "     $ python performance_monitor.py"
echo ""

echo "📋 Next Steps:"
echo "  • Test real HTTP communication"
echo "  • Verify timeout and retry logic"
echo "  • Compare with simulation performance"
echo "  • Proceed to Phase B: Stress Testing"
echo ""

echo "✅ Week 8 Phase A is ready!"
echo ""
