#!/bin/bash
# Startup script for BrowserAI Python API Server

set -e

echo "================================================"
echo "BrowserAI Python API Server Startup"
echo "================================================"

# Check if we're in the training directory
if [ ! -f "api_server.py" ]; then
    echo "Error: api_server.py not found in current directory"
    echo "Please run this script from the /training directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r ../requirements-python-server.txt --quiet

# Set environment variables
export FLASK_APP=api_server.py
export FLASK_ENV=development
export API_HOST=127.0.0.1
export API_PORT=5000
export LOG_LEVEL=INFO
export LATENT_DIM=256
export FEATURE_DIM=48
export LEARNING_RATE=0.001
export BATCH_SIZE=32

echo ""
echo "================================================"
echo "Configuration"
echo "================================================"
echo "API Host: $API_HOST"
echo "API Port: $API_PORT"
echo "Feature Dimension: $FEATURE_DIM"
echo "Latent Dimension: $LATENT_DIM"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE"
echo ""

# Start the server
echo "Starting BrowserAI API Server..."
echo "Server will be available at: http://$API_HOST:$API_PORT"
echo "Press Ctrl+C to stop"
echo ""

python3 api_server.py
