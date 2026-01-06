# GPU Test Results Directory

This directory stores GPU acceleration test results.

## Structure

```
gpu_test_results/
├── gpu_test_summary_YYYYMMDD.md       # Test summary
├── unit_test_output_YYYYMMDD.txt      # Unit test logs
├── nvidia_info_YYYYMMDD.csv           # NVIDIA GPU info (if present)
├── gpu_benchmark_YYYYMMDD.md          # Performance comparison
└── gpu_test_latest.md -> ...          # Symlink to latest
```

## Usage

### Run GPU Tests
```bash
./scripts/test_gpu_acceleration.sh
```

### View Latest Results
```bash
cat gpu_test_results/gpu_test_latest.md
```

## GPU Support Matrix

| Provider | Platform | Detection | Status |
|----------|----------|-----------|--------|
| CUDA | Linux/Windows | nvidia-smi | ✅ Supported |
| DirectML | Windows | OS detection | ✅ Supported |
| CoreML | macOS/iOS | OS detection | ✅ Supported |
| ROCm | Linux | rocm-smi | ⚠️  Experimental |

## Performance Expectations

### With GPU Acceleration
- 2-5x speedup for model inference
- Lower CPU usage
- Higher throughput for batch processing

### CPU-only Mode
- Fully functional fallback
- Optimized for CPU
- Still production-ready

## Test Results Interpretation

### GPU Available
```
✓ GPU detected
✓ Provider initialized
✓ Inference speedup: 3.2x
```

### CPU-only
```
⊘ No GPU detected
⊘ Using CPU provider
✓ CPU inference functional
```

## CI/CD Integration

GPU tests run in CI but typically don't detect GPU:
- Standard GitHub runners don't have GPU
- Tests verify code compiles and runs
- Actual GPU testing needs self-hosted runners

## Manual GPU Testing

On GPU-enabled machines:
```bash
# Install GPU drivers first (CUDA, ROCm, etc.)
# Then run tests
cargo test --features ai gpu_support -- --nocapture
```
