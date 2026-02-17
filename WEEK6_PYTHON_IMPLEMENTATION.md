# BrowserAI Week 6 Python API Server Implementation Guide

## Overview

This guide covers the complete Python API Server implementation for BrowserAI's Week 6 learning system. The server bridges the Rust engine and Python ML backend through REST APIs.

## Components

### 1. Feature Encoder (`feature_encoder.py`)

**Purpose**: Convert 48-dimensional feature vectors to 256-dimensional latent space

**Key Classes**:
- `FeatureEncoder`: Main encoder with 48→256 transformation

**Algorithm**:
```
Input: 48-dim feature vector + intent + style
  ↓
Normalize features to [0, 1]
  ↓
Matrix multiply: features @ encoding_matrix (48×256)
  ↓
Add intent embedding (8-dim) for website type
  ↓
Add style embedding (7-dim) for design style
  ↓
ReLU activation
  ↓
L2 normalize to unit length
  ↓
Output: 256-dim latent vector
```

**Key Methods**:
- `encode(features, intent, design_style)`: Main encoding pipeline
- `decode(latent)`: Reverse operation for debugging
- `get_feature_statistics()`: Analytics and debugging
- `update_weights()`: Training support

**Usage**:
```python
from feature_encoder import FeatureEncoder

encoder = FeatureEncoder(feature_dim=48, latent_dim=256)
features = [0.1] * 48
latent = encoder.encode(features, "blog", "modern")
# latent is now 256-dimensional
```

### 2. Code Generator (`code_generator.py`)

**Purpose**: Generate HTML/CSS/JavaScript from latent vectors

**Key Classes**:
- `CodeGenerator`: Generates web code from latent representations

**Features**:
- Template-based code generation
- Latent-guided parameter selection
- Three complexity levels: simple, moderate, complex
- Confidence scoring

**Key Methods**:
- `generate(latent_vector, session_id)`: Generate code
- `_latent_to_params()`: Extract generation parameters
- `_generate_html()`: HTML structure generation
- `_generate_css()`: CSS styling generation
- `_generate_javascript()`: JavaScript code generation
- `_calculate_confidence()`: Confidence scoring

**Output Quality**:
- Confidence score: 0.5-0.99 (normalized)
- Loss: 1.0 - confidence
- Generated code includes:
  - Semantic HTML structure
  - Responsive CSS layouts
  - Interactive JavaScript

**Usage**:
```python
from code_generator import CodeGenerator
import numpy as np

generator = CodeGenerator(latent_dim=256)
latent = np.random.randn(256)  # 256-dimensional
result = generator.generate(latent, session_id="sess-1")

html = result["html"]
css = result["css"]
javascript = result["javascript"]
confidence = result["confidence"]
```

### 3. Online Learner (`online_learner.py`)

**Purpose**: Update model weights based on rendering feedback

**Key Classes**:
- `OnlineLearner`: Main learning engine with Adam optimizer
- `FeedbackBuffer`: Batch feedback accumulation

**Learning Algorithm**:
```
Input: Features + Generated Latent + Quality Feedback
  ↓
Compute reconstruction loss (expected vs generated latent)
  ↓
Compute quality loss (inverse of quality score)
  ↓
Combine losses with weighting
  ↓
Compute gradients
  ↓
Adam optimizer update:
  m = β₁m + (1-β₁)∇ (momentum)
  v = β₂v + (1-β₂)∇² (velocity)
  w ← w - α·m̂/(√v̂ + ε)
  ↓
Update encoding matrix weights
```

**Key Methods**:
- `process_feedback()`: Main feedback processing
- `_compute_loss()`: Loss computation
- `_compute_gradients()`: Gradient computation
- `_update_weights()`: Adam optimizer update
- `get_metrics()`: Training statistics
- `adaptive_learning_rate()`: Dynamic learning rate adjustment
- `reset_statistics()`: Clear training data

**Metrics Tracking**:
- Training losses (full history)
- Quality scores (full history)
- Update count (total weight updates)
- Feedback count (total feedback samples)
- Convergence metric (stability of recent losses)
- Improvement metric (comparison of early vs recent)

**Usage**:
```python
from online_learner import OnlineLearner, FeedbackBuffer
import numpy as np

learner = OnlineLearner(
    feature_dim=48,
    latent_dim=256,
    learning_rate=0.001,
    batch_size=32
)

# Process feedback
features = np.random.randn(48)
latent = np.random.randn(256)
feedback = {
    "quality_score": 0.85,
    "html_quality": 0.88,
    "css_quality": 0.82,
    "js_quality": 0.80,
}

result = learner.process_feedback(features, latent, feedback)
print(f"Loss: {result['loss']}")
print(f"Weights updated: {result['weights_updated']}")

# Get metrics
metrics = learner.get_metrics()
print(f"Average loss: {metrics['average_loss']}")
print(f"Convergence: {metrics['convergence']}")
```

### 4. Flask API Server (`api_server.py`)

**Purpose**: REST API server connecting Rust and Python

**Key Classes**:
- `Config`: Configuration management
- `BrowserAIServer`: Flask application

**Endpoints**:
1. `GET /api/v1/health` - Health check
2. `POST /api/v1/generate` - Code generation
3. `POST /api/v1/feedback` - Feedback collection
4. `GET /metrics` - Server metrics
5. `GET /` - API information

**Data Models** (Pydantic):
- `FeaturePacketRequest`: Request format
- `GeneratedCodeResponse`: Response format
- `FeedbackPacketRequest`: Feedback format
- `HealthResponse`: Health format

**Usage**:
```bash
# Install dependencies
pip install -r ../requirements-python-server.txt

# Set environment variables
export API_HOST=127.0.0.1
export API_PORT=5000
export LEARNING_RATE=0.001

# Run server
python api_server.py

# Test endpoints
curl http://127.0.0.1:5000/api/v1/health
```

## Complete Integration Flow

```
RUST SIDE                    PYTHON SIDE
┌──────────────────┐
│ Extract Features │
│ (48-dimensional) │
└────────┬─────────┘
         │
         │ POST /api/v1/generate
         │ FeaturePacketRequest
         ↓
     ┌────────────────────┐
     │ FeatureEncoder     │
     │ 48 → 256 latent    │
     └────────┬───────────┘
              │
         ┌────▼───────────┐
         │ CodeGenerator  │
         │ 256 → code     │
         └────────┬───────┘
                  │
                  │ GeneratedCodeResponse
                  ↓
┌──────────────────────┐
│ Render Generated Code │
│ Evaluate Quality     │
└────────┬─────────────┘
         │
         │ POST /api/v1/feedback
         │ FeedbackPacketRequest
         ↓
     ┌────────────────────┐
     │ OnlineLearner      │
     │ Process feedback   │
     │ Update weights     │
     └────────┬───────────┘
              │
         ┌────▼──────────┐
         │ FeedbackBuffer│
         │ Batch updates │
         └───────────────┘
```

## Configuration

### Environment Variables

```bash
# Server Configuration
FLASK_DEBUG=False              # Debug mode (development only)
API_HOST=0.0.0.0              # Bind address
API_PORT=5000                 # Port number
LOG_LEVEL=INFO                # Logging level

# Model Configuration
FEATURE_DIM=48                # Input feature dimension (fixed)
LATENT_DIM=256                # Latent space dimension (fixed)

# Learning Configuration
LEARNING_RATE=0.001           # Adam optimizer learning rate
BATCH_SIZE=32                 # Batch size for feedback
MAX_QUEUE_SIZE=1000           # Max feedback queue size
```

### Configuration File

Create `.env` in training directory:
```
FLASK_DEBUG=False
API_HOST=127.0.0.1
API_PORT=5000
LOG_LEVEL=INFO
LEARNING_RATE=0.001
BATCH_SIZE=32
```

## Running the Server

### Quick Start

```bash
cd /home/stone/BrowerAI/training

# Make script executable
chmod +x start_api_server.sh

# Run server
./start_api_server.sh
```

### Manual Start

```bash
cd /home/stone/BrowerAI/training

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r ../requirements-python-server.txt

# Set environment variables
export FLASK_DEBUG=False
export API_HOST=127.0.0.1
export API_PORT=5000

# Run server
python3 api_server.py
```

## Testing

### Unit Tests

```bash
cd /home/stone/BrowerAI/training

# Run all tests
python3 test_api_server.py

# Expected output
# ✓ Health Check Endpoint
# ✓ Feature Encoding (48→256)
# ✓ Code Generation Endpoint
# ✓ Feedback Processing
# ✓ Metrics Endpoint
# ✓ Error Handling
# ✓ Online Learning Pipeline
# Total: 7/7 tests passed
```

### Integration Testing

```bash
# Terminal 1: Start server
./start_api_server.sh

# Terminal 2: Test endpoints
python3 test_integration.py
```

## API Examples

### Health Check

```bash
curl -X GET http://127.0.0.1:5000/api/v1/health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": 1704067200,
  "uptime_seconds": 3600.0,
  "models_loaded": 3,
  "version": "1.0.0"
}
```

### Generate Code

```bash
curl -X POST http://127.0.0.1:5000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "features": [0.1, 0.2, ...], // 48 values
    "website_intent": "blog",
    "design_style": "modern",
    "session_id": "sess-1",
    "timestamp": 1704067200
  }'
```

**Response**:
```json
{
  "html": "<!DOCTYPE html>...",
  "css": "/* CSS */...",
  "javascript": "// JS...",
  "confidence": 0.85,
  "should_use": true,
  "training_metrics": {...},
  "timestamp": 1704067200
}
```

### Send Feedback

```bash
curl -X POST http://127.0.0.1:5000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "overall_quality": 0.85,
    "html_similarity": 0.88,
    "css_accuracy": 0.82,
    "layout_similarity": 0.85,
    "matched_elements": 45,
    "mismatched_elements": 5,
    "session_id": "sess-1",
    "timestamp": 1704067200
  }'
```

**Response**:
```json
{
  "status": "ok",
  "quality_score": 0.85,
  "buffer_size": 5,
  "buffer_ready": false,
  "learner_metrics": {...},
  "timestamp": 1704067200
}
```

## Monitoring and Debugging

### Logging

All components use Python logging with timestamps and levels:
- INFO: Normal operations
- DEBUG: Detailed execution (disable in production)
- WARNING: Unexpected conditions
- ERROR: Failures

### Metrics Endpoint

```bash
curl http://127.0.0.1:5000/metrics
```

Returns:
- Request statistics (total, success, error)
- Model status (loaded/failed)
- Configuration
- Server uptime

### Performance Profiling

Monitor latency:
```python
import time

start = time.time()
response = requests.post(url, json=data)
duration = time.time() - start
print(f"Request took {duration*1000:.1f}ms")
```

## Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>
```

**2. Module Import Errors**
```bash
# Ensure dependencies installed
pip install -r ../requirements-python-server.txt

# Check Python version (3.8+)
python3 --version
```

**3. Connection Refused**
```bash
# Check server is running
curl -v http://127.0.0.1:5000/api/v1/health

# Check host/port configuration
export API_HOST=0.0.0.0
export API_PORT=5000
```

**4. Memory Issues**
```bash
# Monitor process memory
ps aux | grep python

# Reduce batch size if needed
export BATCH_SIZE=16
```

## Production Deployment

### Using Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 api_server:server.app
```

### Using Docker

```dockerfile
FROM python:3.11

WORKDIR /app

COPY requirements-python-server.txt .
RUN pip install -r requirements-python-server.txt

COPY training/ .

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "api_server:server.app"]
```

### Performance Tuning

1. **Increase Workers**: `gunicorn -w 8` for multi-core
2. **Enable Caching**: Cache encoding matrices
3. **Batch Feedback**: Process feedback in larger batches
4. **Model Quantization**: Compress weight matrices

## Next Steps

1. ✅ Complete Python API Server implementation
2. ✅ Implement feature encoder module
3. ✅ Implement code generator module
4. ✅ Implement online learner module
5. 🔄 Integration testing with Rust
6. 🔄 Performance benchmarking
7. 🔄 Production deployment

## References

- [API Specification](WEEK6_API_SPEC.md)
- [Feature Vector Design](WEEK6_FEATURES.md)
- [Online Learning Algorithm](WEEK6_ONLINE_LEARNING.md)
