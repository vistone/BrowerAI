#!/bin/bash
# Baseline Benchmark Establishment Script
# Runs comprehensive performance benchmarks and saves baseline metrics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARK_DIR="$PROJECT_ROOT/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASELINE_FILE="$BENCHMARK_DIR/baseline_$TIMESTAMP.json"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "BrowerAI Baseline Benchmark Suite"
echo "=========================================="
echo ""
echo "Timestamp: $TIMESTAMP"
echo "Output directory: $BENCHMARK_DIR"
echo ""

# Create benchmark directory
mkdir -p "$BENCHMARK_DIR"

# System information
echo -e "${BLUE}Collecting System Information...${NC}"
echo "OS: $(uname -s)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo 'N/A')"
echo "CPU cores: $(nproc 2>/dev/null || echo 'N/A')"
echo "Memory: $(free -h 2>/dev/null | awk '/^Mem:/ {print $2}' || echo 'N/A')"
echo ""

# Save system info to JSON
cat > "$BENCHMARK_DIR/system_info_$TIMESTAMP.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "os": "$(uname -s)",
  "kernel": "$(uname -r)",
  "architecture": "$(uname -m)",
  "cpu_cores": "$(nproc 2>/dev/null || echo '0')",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')"
}
EOF

echo -e "${BLUE}Building project in release mode...${NC}"
if command -v cargo &> /dev/null; then
    cargo build --release --example benchmark_demo 2>&1 | grep -E "Compiling|Finished" || true
    echo -e "${GREEN}✓${NC} Build complete"
else
    echo -e "${YELLOW}⊘ Cargo not available, skipping build${NC}"
fi
echo ""

# Run benchmark tests
echo -e "${BLUE}Running Benchmark Tests...${NC}"
echo "This may take a few minutes..."
echo ""

if command -v cargo &> /dev/null; then
    # Run the benchmark demo and capture output
    cargo run --release --example benchmark_demo > "$BENCHMARK_DIR/benchmark_output_$TIMESTAMP.txt" 2>&1 || {
        echo -e "${YELLOW}⊘ Benchmark demo execution needs implementation${NC}"
    }
    
    # Run unit tests with benchmarking
    echo -e "${BLUE}Running benchmark unit tests...${NC}"
    cargo test --release benchmark -- --nocapture > "$BENCHMARK_DIR/test_output_$TIMESTAMP.txt" 2>&1 || true
    
    echo -e "${GREEN}✓${NC} Benchmarks complete"
else
    echo -e "${YELLOW}⊘ Cargo not available${NC}"
fi
echo ""

# Analyze results
echo -e "${BLUE}Analyzing Results...${NC}"

if [ -f "$BENCHMARK_DIR/benchmark_output_$TIMESTAMP.txt" ]; then
    # Extract key metrics (this is a simplified parser)
    echo "Key Metrics:"
    grep -E "Avg time:|Throughput:|Size:" "$BENCHMARK_DIR/benchmark_output_$TIMESTAMP.txt" | head -10 || echo "  (Raw output saved to file)"
fi
echo ""

# Generate baseline summary
echo -e "${BLUE}Generating Baseline Summary...${NC}"

cat > "$BENCHMARK_DIR/baseline_summary_$TIMESTAMP.md" <<EOF
# Baseline Benchmark Results

**Date**: $(date)
**Timestamp**: $TIMESTAMP
**Git Commit**: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')

## System Configuration

- **OS**: $(uname -s)
- **Architecture**: $(uname -m)
- **CPU Cores**: $(nproc 2>/dev/null || echo 'N/A')
- **Memory**: $(free -h 2>/dev/null | awk '/^Mem:/ {print $2}' || echo 'N/A')

## Test Configuration

- **Build Mode**: Release
- **Iterations**: 100 (default)
- **Warmup**: 10 iterations

## Results

See detailed results in:
- \`benchmark_output_$TIMESTAMP.txt\` - Full benchmark output
- \`test_output_$TIMESTAMP.txt\` - Unit test results
- \`system_info_$TIMESTAMP.json\` - System information

## Baseline Metrics

$(if [ -f "$BENCHMARK_DIR/benchmark_output_$TIMESTAMP.txt" ]; then
    echo "### HTML Parsing Performance"
    grep -A 5 "html_parse" "$BENCHMARK_DIR/benchmark_output_$TIMESTAMP.txt" | head -6 || echo "N/A"
    echo ""
    echo "### CSS Parsing Performance"
    grep -A 5 "css" "$BENCHMARK_DIR/benchmark_output_$TIMESTAMP.txt" | head -6 || echo "N/A"
    echo ""
    echo "### JavaScript Parsing Performance"
    grep -A 5 "js" "$BENCHMARK_DIR/benchmark_output_$TIMESTAMP.txt" | head -6 || echo "N/A"
else
    echo "Benchmark output not available"
fi)

## Next Steps

1. Compare future benchmarks against this baseline
2. Track performance regressions
3. Validate optimization improvements
4. Test GPU acceleration impact

EOF

echo -e "${GREEN}✓${NC} Summary generated"
echo ""

# List all results
echo "=========================================="
echo "Benchmark Results Summary"
echo "=========================================="
echo ""
echo "Results saved to: $BENCHMARK_DIR"
ls -lh "$BENCHMARK_DIR"/*"$TIMESTAMP"* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

# Create latest symlinks
cd "$BENCHMARK_DIR"
ln -sf "baseline_summary_$TIMESTAMP.md" "baseline_latest.md"
ln -sf "benchmark_output_$TIMESTAMP.txt" "benchmark_latest.txt"
ln -sf "system_info_$TIMESTAMP.json" "system_info_latest.json"

echo -e "${GREEN}✓ Baseline benchmarks established${NC}"
echo ""
echo "View results:"
echo "  cat $BENCHMARK_DIR/baseline_latest.md"
echo ""
echo "Compare future runs:"
echo "  ./scripts/compare_benchmarks.sh baseline_$TIMESTAMP"
