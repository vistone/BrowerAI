#!/bin/bash
# GPU Acceleration Testing Script
# Tests GPU support, detects available hardware, and measures performance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GPU_TEST_DIR="$PROJECT_ROOT/gpu_test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "BrowerAI GPU Acceleration Test Suite"
echo "=========================================="
echo ""

# Create test directory
mkdir -p "$GPU_TEST_DIR"

# Detect GPU hardware
echo -e "${BLUE}Detecting GPU Hardware...${NC}"
echo ""

GPU_AVAILABLE=false
GPU_TYPE="none"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
    GPU_AVAILABLE=true
    GPU_TYPE="cuda"
    
    # Save GPU info
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv > "$GPU_TEST_DIR/nvidia_info_$TIMESTAMP.csv" 2>/dev/null || true
else
    echo -e "${YELLOW}⊘ NVIDIA GPU not detected${NC}"
fi
echo ""

# Check for AMD GPU (rocm-smi)
if command -v rocm-smi &> /dev/null; then
    echo -e "${GREEN}✓ AMD GPU detected${NC}"
    rocm-smi --showproductname 2>/dev/null || true
    GPU_AVAILABLE=true
    if [ "$GPU_TYPE" = "none" ]; then
        GPU_TYPE="rocm"
    fi
else
    echo -e "${YELLOW}⊘ AMD GPU not detected${NC}"
fi
echo ""

# Check for Intel GPU (Windows DirectML)
if [ "$(uname -s)" = "MINGW"* ] || [ "$(uname -s)" = "MSYS"* ]; then
    echo -e "${BLUE}Windows detected - DirectML may be available${NC}"
    GPU_TYPE="directml"
    GPU_AVAILABLE=true
elif [ "$(uname -s)" = "Darwin" ]; then
    echo -e "${BLUE}macOS detected - CoreML available${NC}"
    GPU_TYPE="coreml"
    GPU_AVAILABLE=true
fi
echo ""

# System info
echo -e "${BLUE}System Information:${NC}"
echo "OS: $(uname -s)"
echo "Architecture: $(uname -m)"
echo "GPU Type: $GPU_TYPE"
echo "GPU Available: $GPU_AVAILABLE"
echo ""

# Run GPU unit tests
echo -e "${BLUE}Running GPU Unit Tests...${NC}"

if command -v cargo &> /dev/null; then
    # Build with AI features
    echo "Building with AI features..."
    cargo build --features ai --lib 2>&1 | grep -E "Compiling|Finished" || true
    echo ""
    
    # Run GPU tests
    echo "Running GPU support tests..."
    cargo test --features ai gpu_support -- --nocapture > "$GPU_TEST_DIR/unit_test_output_$TIMESTAMP.txt" 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ GPU unit tests passed${NC}"
        grep -E "test result:|running" "$GPU_TEST_DIR/unit_test_output_$TIMESTAMP.txt"
    else
        echo -e "${RED}✗ GPU unit tests failed${NC}"
    fi
else
    echo -e "${YELLOW}⊘ Cargo not available${NC}"
fi
echo ""

# GPU detection test
echo -e "${BLUE}Testing GPU Detection...${NC}"

cat > "$GPU_TEST_DIR/gpu_detection_test.rs" <<'RUST_CODE'
use browerai::GpuConfig;

fn main() {
    env_logger::init();
    
    println!("Testing GPU detection...");
    
    let config = GpuConfig::default();
    let providers = config.detect_available_providers();
    
    println!("Available providers: {} found", providers.len());
    for (i, provider) in providers.iter().enumerate() {
        println!("  {}. {:?}", i + 1, provider);
    }
    
    if providers.len() > 1 {
        println!("\n✓ GPU acceleration available!");
    } else {
        println!("\n⊘ No GPU detected, CPU-only mode");
    }
}
RUST_CODE

if command -v cargo &> /dev/null; then
    cd "$PROJECT_ROOT"
    echo "Compiling GPU detection test..."
    rustc --edition 2021 -L target/release/deps "$GPU_TEST_DIR/gpu_detection_test.rs" -o "$GPU_TEST_DIR/gpu_detection_test" 2>/dev/null || {
        echo -e "${YELLOW}Note: Direct compilation needs dependencies${NC}"
    }
fi
echo ""

# Performance comparison: CPU vs GPU
if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "${BLUE}Running CPU vs GPU Performance Comparison...${NC}"
    echo "This will compare inference times with and without GPU acceleration"
    echo ""
    
    cat > "$GPU_TEST_DIR/gpu_benchmark_$TIMESTAMP.md" <<EOF
# GPU Acceleration Benchmark Results

**Date**: $(date)
**GPU Type**: $GPU_TYPE
**System**: $(uname -s) $(uname -m)

## Hardware Configuration

### GPU Information
$(if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "GPU type: $GPU_TYPE"
fi)

### CPU Information
- Model: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo 'N/A')
- Cores: $(nproc 2>/dev/null || echo 'N/A')

## Test Configuration

- Build: Release with AI features
- Test: Model inference performance
- Iterations: 100 per test

## Results

### CPU-only Mode
- Provider: CPU
- Average inference time: TBD
- Throughput: TBD

### GPU-accelerated Mode
- Provider: $GPU_TYPE
- Average inference time: TBD
- Throughput: TBD
- Speedup: TBD

## Analysis

GPU acceleration is $(if [ "$GPU_AVAILABLE" = true ]; then echo "AVAILABLE"; else echo "NOT AVAILABLE"; fi)

## Next Steps

1. Run actual model inference tests
2. Measure real-world speedup
3. Profile GPU memory usage
4. Test with different model sizes

EOF

    echo -e "${GREEN}✓${NC} Benchmark template created"
else
    echo -e "${YELLOW}⊘ GPU not available - skipping performance comparison${NC}"
fi
echo ""

# Generate summary report
cat > "$GPU_TEST_DIR/gpu_test_summary_$TIMESTAMP.md" <<EOF
# GPU Acceleration Test Summary

**Timestamp**: $TIMESTAMP
**Date**: $(date)

## Detection Results

- **GPU Available**: $GPU_AVAILABLE
- **GPU Type**: $GPU_TYPE
- **CUDA Support**: $(if command -v nvidia-smi &> /dev/null; then echo "Yes"; else echo "No"; fi)
- **ROCm Support**: $(if command -v rocm-smi &> /dev/null; then echo "Yes"; else echo "No"; fi)

## Test Results

$(if [ -f "$GPU_TEST_DIR/unit_test_output_$TIMESTAMP.txt" ]; then
    echo "### Unit Tests"
    grep "test result:" "$GPU_TEST_DIR/unit_test_output_$TIMESTAMP.txt" || echo "See full output file"
else
    echo "Unit tests not executed"
fi)

## Files Generated

- \`gpu_test_summary_$TIMESTAMP.md\` - This file
- \`unit_test_output_$TIMESTAMP.txt\` - Full test output
$(if [ -f "$GPU_TEST_DIR/nvidia_info_$TIMESTAMP.csv" ]; then echo "- \`nvidia_info_$TIMESTAMP.csv\` - NVIDIA GPU details"; fi)
$(if [ -f "$GPU_TEST_DIR/gpu_benchmark_$TIMESTAMP.md" ]; then echo "- \`gpu_benchmark_$TIMESTAMP.md\` - Performance comparison"; fi)

## Recommendations

$(if [ "$GPU_AVAILABLE" = true ]; then
    echo "✓ GPU acceleration is available and can be used"
    echo "✓ Build with: \`cargo build --features ai\`"
    echo "✓ Use GpuConfig API to enable GPU inference"
else
    echo "⊘ No GPU detected - CPU-only mode"
    echo "  - Install NVIDIA/AMD drivers if GPU is present"
    echo "  - CPU inference is still fully functional"
fi)

EOF

echo "=========================================="
echo "GPU Test Results"
echo "=========================================="
echo ""
cat "$GPU_TEST_DIR/gpu_test_summary_$TIMESTAMP.md"
echo ""

# Create latest symlink
cd "$GPU_TEST_DIR"
ln -sf "gpu_test_summary_$TIMESTAMP.md" "gpu_test_latest.md"

echo -e "${GREEN}✓ GPU testing complete${NC}"
echo ""
echo "View results: cat $GPU_TEST_DIR/gpu_test_latest.md"
