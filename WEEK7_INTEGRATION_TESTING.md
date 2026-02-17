# Week 6 Integration Testing Guide

## Overview

This guide covers the comprehensive integration testing suite for BrowserAI Week 6, which validates the complete Rust ↔ Python communication pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Integration Test Suite                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │                    │
            ┌──────────────┐      ┌──────────────┐
            │ Rust Tests   │      │ Python Tests │
            └──────────────┘      └──────────────┘
                    │                    │
        48D Features│             Features│ 256D Encoding
        Extraction  │             Code Gen│ Learning
        Validation  │             Feedback│
                    │                    │
                    └─────────────┬──────┘
                                  │
                          ┌───────────────┐
                          │ Python Server │
                          │  /api/v1/*    │
                          └───────────────┘
                                  │
                          ┌───────────────┐
                          │  5 Endpoints  │
                          │  • health     │
                          │  • generate   │
                          │  • feedback   │
                          │  • stats      │
                          │  • version    │
                          └───────────────┘
```

## Quick Start

### 1. Run All Tests (Recommended)

```bash
cd /home/stone/BrowerAI
python training/run_integration_tests.py
```

This will:
- Check Python dependencies
- Start the Python API server
- Run integration tests
- Run performance benchmarks
- Stop the server

### 2. Run Python Tests Only

```bash
cd /home/stone/BrowerAI/training
python integration_test_runner.py
```

**Requirements**:
- Python API server must be running on `http://127.0.0.1:5000`
- Install dependencies: `pip install requests psutil pydantic flask numpy`

### 3. Run Rust Tests

```bash
cd /home/stone/BrowerAI
cargo test --test week6_integration_tests
```

### 4. Run Performance Benchmark

```bash
cd /home/stone/BrowerAI/training
python performance_monitor.py
```

## Test Components

### Python Integration Tests (`training/integration_test_runner.py`)

Seven comprehensive test scenarios:

#### Test 1: Health Check
- **Purpose**: Verify server readiness
- **Endpoint**: `GET /api/v1/health`
- **Validation**: 
  - Status = "healthy"
  - Models loaded count = 3
  - Has uptime metric
- **Target**: < 5ms latency

#### Test 2: Feature to Code Generation
- **Purpose**: Validate complete pipeline
- **Endpoint**: `POST /api/v1/generate`
- **Input**: 48-dimensional feature vector
- **Output**: HTML + CSS + JavaScript
- **Validation**:
  - Response contains all 3 code types
  - Confidence score in [0.0, 1.0]
  - Non-empty content
- **Target**: 10-20ms latency

#### Test 3: Feedback Processing
- **Purpose**: Test learning loop
- **Endpoint**: `POST /api/v1/feedback`
- **Input**: Quality metrics + feedback
- **Validation**:
  - Status = "ok"
  - Metrics updated (loss, convergence)
  - Buffer size tracked
- **Target**: 5-10ms latency

#### Test 4: Learning Loop (3 Iterations)
- **Purpose**: Complete learning cycle
- **Operations**: Generate → Evaluate → Feedback (3x)
- **Validation**: 
  - All 3 iterations complete
  - Quality trend (should improve or stabilize)
  - Metrics converge
- **Target**: 60-90ms total

#### Test 5: Throughput (10 Requests)
- **Purpose**: Measure request throughput
- **Operations**: 10 sequential code generation requests
- **Validation**:
  - All 10 requests succeed
  - Throughput > 50 req/s
  - Consistent latency across requests
- **Target**: 10+ RPS

#### Test 6: Error Handling
- **Purpose**: Validate input validation
- **Tests**:
  - Invalid feature dimension (47 instead of 48)
  - Invalid quality score (1.5 out of [0, 1])
- **Expected**: HTTP 400 Bad Request
- **Target**: Proper error messages

#### Test 7: Latency Consistency
- **Purpose**: Check performance stability
- **Operations**: 5 sequential requests
- **Validation**:
  - Variance < 5ms
  - No outliers
  - Consistent performance
- **Target**: ±5ms variance

### Rust Integration Tests (`tests/week6_integration_tests.rs`)

Seven corresponding test scenarios:

#### Test 1: Health Check
```rust
pub fn test_health_check() -> TestResult
```
- Validates server health response
- Checks response structure

#### Test 2: Feature Extraction
```rust
pub fn test_feature_extraction() -> TestResult
```
- Validates 48D feature extraction
- Checks normalization

#### Test 3: Feature Encoding
```rust
pub fn test_feature_encoding() -> TestResult
```
- Validates 48D → 256D transformation
- Checks latent space validity

#### Test 4: Feedback Processing
```rust
pub fn test_feedback_processing() -> TestResult
```
- Validates feedback structure
- Checks quality score ranges

#### Test 5: Code Generation
```rust
pub fn test_code_generation() -> TestResult
```
- Validates code output structure
- Checks confidence scoring

#### Test 6: Error Handling
```rust
pub fn test_error_handling() -> TestResult
```
- Validates dimension checking
- Validates range checking

#### Test 7: Performance Metrics
```rust
pub fn test_performance_metrics() -> TestResult
```
- Measures operation latency
- Targets < 20ms average

## Performance Monitoring

### `training/performance_monitor.py`

Comprehensive performance analysis:

```
Performance Metrics:
├─ Endpoints
│  ├─ GET /api/v1/health (50 requests)
│  │  └─ Latency: mean, median, min, max, p95, p99
│  ├─ POST /api/v1/generate (20 requests)
│  │  └─ Latency + Throughput
│  └─ POST /api/v1/feedback (30 requests)
│     └─ Latency + Throughput
├─ Overall
│  ├─ Average latency across endpoints
│  └─ Max latency
└─ System Resources
   ├─ CPU usage
   ├─ Memory usage
   └─ API server memory
```

### Output Report

Saved to: `/home/stone/BrowerAI/data/week6_performance_report.json`

```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "endpoints": {
    "health": {
      "endpoint": "/api/v1/health",
      "method": "GET",
      "total_requests": 50,
      "successful": 50,
      "success_rate": "100.0%",
      "latency_ms": {
        "mean": "2.34",
        "median": "2.12",
        "min": "1.89",
        "max": "5.43",
        "p95": "3.21",
        "p99": "4.56"
      },
      "throughput_rps": "2137.2"
    },
    ...
  },
  "overall": {
    "avg_latency_ms": "8.45",
    "max_latency_ms": "18.92"
  },
  "resources": {
    "cpu_percent": "12.5%",
    "memory_percent": "8.3%",
    "api_server_memory_mb": "45.2"
  }
}
```

## Success Criteria

### Integration Tests
- ✅ All 7 tests PASS
- ✅ Health check < 5ms
- ✅ Code generation 10-20ms
- ✅ Feedback processing 5-10ms
- ✅ Throughput > 50 RPS
- ✅ Error handling validates inputs

### Performance Metrics
- ✅ Average latency < 15ms
- ✅ Throughput > 100 RPS total
- ✅ Memory usage < 100MB
- ✅ CPU usage < 20%

### System Stability
- ✅ Server runs without crashes
- ✅ No memory leaks
- ✅ Consistent performance across requests
- ✅ Proper error handling

## Troubleshooting

### Server Won't Start

```bash
# Check if port 5000 is in use
lsof -i :5000

# Kill existing process if needed
kill -9 <PID>

# Try custom port
FLASK_PORT=5001 python training/api_server.py
```

### Import Errors

```bash
# Install missing dependencies
pip install flask numpy requests psutil pydantic

# Verify installation
python -c "import flask; print(flask.__version__)"
```

### Connection Refused

```bash
# Check server is running
curl http://127.0.0.1:5000/api/v1/health

# If not running, start it manually
python training/api_server.py
```

### Test Failures

Check logs in API server output:
```bash
# Run with verbose output
FLASK_DEBUG=1 python training/api_server.py
```

## Next Steps

1. **Compile and Run**
   ```bash
   cargo test --test week6_integration_tests
   python training/run_integration_tests.py
   ```

2. **Analyze Results**
   - Check all tests pass
   - Review performance metrics
   - Identify bottlenecks

3. **Optimization**
   - Profile slow endpoints
   - Optimize encoding/generation
   - Reduce latency

4. **Deployment**
   - Docker containerization
   - Load testing
   - Production deployment

## Files Reference

| File | Purpose | Type |
|------|---------|------|
| `training/integration_test_runner.py` | Python test runner | Test |
| `training/performance_monitor.py` | Performance measurement | Monitor |
| `training/run_integration_tests.py` | Orchestrator | Script |
| `tests/week6_integration_tests.rs` | Rust tests | Test |
| `data/week6_performance_report.json` | Benchmark results | Report |

## Running the Full Suite

```bash
# Change to workspace root
cd /home/stone/BrowerAI

# Run complete test suite with orchestrator
python training/run_integration_tests.py

# Expected output:
# ✓ Health Check - 2.34ms
# ✓ Feature Generation - 12.45ms
# ✓ Feedback Processing - 6.78ms
# ✓ Learning Loop - 75.32ms
# ✓ Throughput - 150.5 RPS
# ✓ Error Handling - tests passed
# ✓ Latency Consistency - variance 2.1ms
#
# Summary: 7/7 tests passed
```

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Health check latency | < 5ms | TBD |
| Code generation latency | 10-20ms | TBD |
| Feedback processing | 5-10ms | TBD |
| Throughput | > 100 RPS | TBD |
| Memory usage | < 100MB | TBD |
| CPU usage | < 20% | TBD |

---

**Status**: Integration Testing Framework Ready
**Last Updated**: Week 6
**Next Phase**: Week 7 - Performance Optimization
