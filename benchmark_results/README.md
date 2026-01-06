# Benchmark Results Directory

This directory stores performance benchmark results for BrowerAI.

## Structure

```
benchmark_results/
├── baseline_YYYYMMDD_HHMMSS.json      # Baseline metrics
├── baseline_summary_YYYYMMDD.md       # Human-readable summary
├── benchmark_output_YYYYMMDD.txt      # Raw benchmark output
├── system_info_YYYYMMDD.json          # System configuration
├── baseline_latest.md -> ...          # Symlink to latest summary
├── benchmark_latest.txt -> ...        # Symlink to latest output
└── system_info_latest.json -> ...     # Symlink to latest system info
```

## Usage

### Generate Baseline
```bash
./scripts/run_baseline_benchmarks.sh
```

### View Latest Results
```bash
cat benchmark_results/baseline_latest.md
```

### Compare Runs
```bash
# Compare two benchmark runs
diff benchmark_results/baseline_20260105.md benchmark_results/baseline_20260106.md
```

## Metrics Tracked

- **HTML Parsing**: Time to parse various HTML sizes
- **CSS Parsing**: Selector optimization performance
- **JS Parsing**: JavaScript syntax analysis speed
- **Throughput**: MB/s processing rate
- **Memory**: Peak memory usage
- **Latency**: Min/max/avg/median times

## Interpretation

### Good Performance
- Avg time < 1ms for small documents
- Throughput > 10 MB/s
- Std deviation < 20% of average

### Performance Regression
- Avg time increased > 10%
- Throughput decreased > 10%
- New outliers in max time

### Performance Improvement
- Reduced avg time
- Lower std deviation
- Higher throughput

## CI Integration

Benchmarks run automatically:
- **Weekly**: Sunday 1 AM UTC
- **On Push**: To main branch
- **Manual**: Via GitHub Actions

Results uploaded as artifacts in CI runs.

## Historical Tracking

Keep at least 3 baseline files:
- Current baseline
- Previous baseline (for regression detection)
- Original baseline (for long-term trends)

Archive old results monthly:
```bash
mkdir -p benchmark_results/archive/2026-01
mv benchmark_results/*202601* benchmark_results/archive/2026-01/
```

## Notes

- Run on same hardware for consistent comparison
- Use release builds only
- Disable background processes during benchmarking
- Multiple runs recommended for accuracy
