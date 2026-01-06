# CI/CD Integration Guide

## Overview

BrowerAI now includes comprehensive CI/CD workflows for automated testing, benchmarking, and continuous integration.

## 🔄 GitHub Actions Workflows

### 1. CI Tests (`ci-tests.yml`)

**Trigger**: Push/PR to main or develop branches
**Frequency**: On every commit + Daily at 2 AM UTC

**Jobs**:
- ✅ **Test Suite** - Runs on Ubuntu, macOS, Windows with stable & nightly Rust
- ✅ **Benchmark** - Performance benchmarks in release mode
- ✅ **E2E Tests** - End-to-end website testing
- ✅ **Model Validation** - Python model generation validation
- ✅ **GPU Detection** - GPU support testing
- ✅ **Security Audit** - Cargo audit for vulnerabilities
- ✅ **Code Coverage** - Coverage reporting with Codecov

**Manual Run**:
```bash
# Locally simulate CI tests
cargo test --all-targets
cargo test --features ai
cargo clippy --all-targets
cargo fmt -- --check
```

### 2. E2E Website Tests (`e2e-tests.yml`)

**Trigger**: Push to main, Weekly on Monday, Manual dispatch
**Categories**: Quick, Full, Stress testing

**Jobs**:
- **Quick E2E** - Fast tests on core websites (10 min)
- **Full E2E** - Comprehensive test suite (30 min)
- **Stress E2E** - Heavy load testing (45 min, manual only)
- **Report** - Aggregated test report

**Test Categories**:
- Simple: example.com, info.cern.ch
- Medium: wikipedia.org, news.ycombinator.com
- Complex: github.com, docs.rust-lang.org

**Manual Trigger**:
```bash
# Via GitHub UI: Actions → E2E Website Tests → Run workflow
# Or using gh CLI:
gh workflow run e2e-tests.yml
```

### 3. Performance Benchmarks (`benchmark.yml`)

**Trigger**: Push to main, Weekly on Sunday, Manual dispatch
**Platforms**: Ubuntu, macOS, Windows

**Jobs**:
- **Baseline Benchmark** - Cross-platform performance
- **Compare Baseline** - Regression detection
- **GPU Benchmark** - GPU acceleration testing (if available)
- **Regression Check** - Performance analysis

**Custom Iterations**:
```bash
# Manual run with custom iterations
gh workflow run benchmark.yml -f iterations=500
```

## 📦 Artifacts

All workflows generate downloadable artifacts:

| Workflow | Artifact | Contents |
|----------|----------|----------|
| ci-tests | `benchmark-results` | Performance metrics |
| ci-tests | `e2e-test-results` | E2E test logs |
| ci-tests | `generated-models` | ONNX test models |
| e2e-tests | `e2e-summary-report` | Consolidated report |
| benchmark | `benchmark-{os}` | Platform-specific results |
| benchmark | `gpu-benchmark-results` | GPU performance data |

**Download Artifacts**:
```bash
# Via GitHub UI: Actions → Workflow Run → Artifacts section
# Or using gh CLI:
gh run download <run-id>
```

## 🚀 Local Testing Scripts

### Run Baseline Benchmarks
```bash
./scripts/run_baseline_benchmarks.sh
```

**Output**:
- `benchmark_results/baseline_YYYYMMDD_HHMMSS.json`
- `benchmark_results/benchmark_output_*.txt`
- `benchmark_results/baseline_summary_*.md`
- Symlinks: `baseline_latest.md`, `benchmark_latest.txt`

**Features**:
- System information collection
- Release mode benchmarking
- Statistical analysis
- Comparison baseline

### Test GPU Acceleration
```bash
./scripts/test_gpu_acceleration.sh
```

**Output**:
- `gpu_test_results/gpu_test_summary_*.md`
- `gpu_test_results/unit_test_output_*.txt`
- `gpu_test_results/nvidia_info_*.csv` (if NVIDIA GPU)
- `gpu_test_results/gpu_benchmark_*.md`

**Features**:
- Multi-vendor GPU detection (CUDA, ROCm, DirectML, CoreML)
- Hardware capability assessment
- CPU vs GPU comparison
- Performance recommendations

## 🔧 Configuration

### Workflow Schedules

Edit schedules in `.github/workflows/*.yml`:

```yaml
schedule:
  - cron: '0 2 * * *'  # Daily at 2 AM UTC
  - cron: '0 1 * * 0'  # Weekly on Sunday at 1 AM
  - cron: '0 3 * * 1'  # Weekly on Monday at 3 AM
```

### Test Timeouts

Adjust per-job timeouts:

```yaml
jobs:
  e2e-quick:
    timeout-minutes: 10  # Default: unlimited
```

### Matrix Testing

Customize OS/Rust combinations:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    rust: [stable, nightly]
```

## 📊 Monitoring & Reports

### GitHub Actions Summary

Each workflow adds a summary to the Actions run:

- 📈 Key performance metrics
- ✅ Test pass/fail counts
- 📉 Regression alerts
- 🔗 Links to detailed artifacts

### Pull Request Comments

CI status automatically appears on PRs:

```
✅ ci-tests / Test Suite (ubuntu-latest, stable) 
✅ ci-tests / Benchmark
⚠️  e2e-tests / Quick E2E Tests (some failures)
```

### Badges

Add to README.md:

```markdown
![CI Tests](https://github.com/vistone/BrowerAI/workflows/CI%20Tests/badge.svg)
![E2E Tests](https://github.com/vistone/BrowerAI/workflows/E2E%20Website%20Tests/badge.svg)
![Benchmarks](https://github.com/vistone/BrowerAI/workflows/Performance%20Benchmarks/badge.svg)
```

## 🔐 Secrets & Variables

Required secrets (if using optional features):

| Secret | Purpose | Required |
|--------|---------|----------|
| `CODECOV_TOKEN` | Coverage reporting | Optional |
| `CARGO_REGISTRY_TOKEN` | Crate publishing | For releases |

Set in: Repository Settings → Secrets and variables → Actions

## 🐛 Troubleshooting

### Common Issues

**Issue**: Tests fail in CI but pass locally
```bash
# Check with same Rust version as CI
rustup install stable
cargo +stable test
```

**Issue**: E2E tests timeout
```yaml
# Increase timeout in workflow
timeout-minutes: 30  # Default: 10
```

**Issue**: GPU tests always skip
```yaml
# GPU not available in standard GitHub runners
# Use self-hosted runners with GPU for actual GPU testing
```

**Issue**: Model generation fails
```bash
# Ensure Python dependencies in CI
pip install numpy onnx
```

### Debug Mode

Enable debug output:

```yaml
env:
  RUST_LOG: debug
  RUST_BACKTRACE: full
```

## 📝 Best Practices

1. **Test Before Push**
   ```bash
   cargo test --all-features
   cargo clippy --all-targets
   cargo fmt
   ```

2. **Use Draft PRs** - For experimental work that shouldn't trigger full CI

3. **Manual Workflows** - Use for expensive tests (stress, GPU)

4. **Cache Aggressively** - Workflows use cargo cache to speed up builds

5. **Monitor Artifacts** - Review benchmark results weekly

6. **Update Baselines** - Refresh performance baselines after major changes

## 🔄 Continuous Improvement

### Weekly Review
- Check E2E test failure patterns
- Review benchmark trends
- Update test URLs if sites change
- Validate GPU detection accuracy

### Monthly Tasks
- Update dependencies
- Refresh baseline benchmarks
- Review coverage reports
- Optimize workflow performance

## 📚 Additional Resources

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Rust CI Best Practices](https://github.com/actions-rs)
- [Benchmark Tracking](benchmark_results/README.md)

---

**All CI/CD workflows are now active and ready! 🎉**

For questions or issues, see: [CONTRIBUTING.md](../CONTRIBUTING.md)
