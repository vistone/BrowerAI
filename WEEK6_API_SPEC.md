# BrowserAI Python API Server Specification

## Overview

The Python API Server is the machine learning backend for BrowserAI's Week 6 learning system. It provides REST endpoints for:
- Converting 48-dimensional feature vectors to code through latent space generation
- Processing rendering quality feedback for online learning
- Monitoring server health and system metrics

## Architecture

```
Rust Browser Engine
        ↓
   (HTTP REST)
        ↓
Python Flask Server
   ├─ FeatureEncoder (48→256 dimensions)
   ├─ CodeGenerator (256→HTML/CSS/JS)
   ├─ OnlineLearner (feedback→model updates)
   └─ FeedbackBuffer (batch feedback processing)
```

## API Endpoints

### 1. Health Check

**Endpoint**: `GET /api/v1/health`

**Purpose**: Check if API server is running and healthy

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": 1704067200,
  "uptime_seconds": 3600.0,
  "models_loaded": 3,
  "version": "1.0.0"
}
```

**Error Response** (503 Service Unavailable):
```json
{
  "status": "unhealthy",
  "error": "Model loading failed"
}
```

---

### 2. Code Generation

**Endpoint**: `POST /api/v1/generate`

**Purpose**: Generate HTML/CSS/JavaScript from website features

**Request Body**:
```json
{
  "url": "https://example.com",
  "features": [0.1, 0.2, ..., 0.15],  // 48-dimensional array
  "website_intent": "blog",
  "design_style": "modern",
  "session_id": "sess-12345",
  "timestamp": 1704067200
}
```

**Feature Vector Composition** (48 dimensions):
- [0-9]: HTML metrics (10)
  - 0: semantic HTML tags count
  - 1: form elements count
  - 2: media elements count
  - 3: text content ratio
  - 4-9: specific tag counts
- [10-17]: CSS metrics (8)
  - 10: color palette diversity
  - 11: font family count
  - 12: animation count
  - 13-17: CSS property distributions
- [18-27]: JavaScript metrics (10)
  - 18: function count
  - 19: class count
  - 20: variable count
  - 21-27: event handler distributions
- [28-35]: Page structure (8)
  - 28: header presence
  - 29: navigation complexity
  - 30: main content ratio
  - 31-35: footer and section metrics
- [36-42]: Design style (7)
  - 36: formality score
  - 37: colorfulness score
  - 38: minimalism score
  - 39: modernity score
  - 40-42: visual hierarchy metrics
- [43-47]: Complexity (5)
  - 43: total resource count
  - 44: external scripts ratio
  - 45: CDN dependency ratio
  - 46-47: compression and minification metrics

**Website Intent Types**:
- `blog`: Blog or article-focused website
- `ecommerce`: E-commerce or product catalog
- `documentation`: Technical documentation
- `portfolio`: Portfolio or personal site
- `landing`: Landing page or promotional
- `social`: Social media or community
- `news`: News or media publication

**Design Style Types**:
- `modern`: Contemporary, trendy design
- `minimal`: Minimalist, clean design
- `classic`: Traditional, formal design
- `playful`: Fun, creative design
- `professional`: Corporate, business design
- `creative`: Artistic, experimental design

**Response** (200 OK):
```json
{
  "html": "<!DOCTYPE html>...",
  "css": "/* Generated CSS */...",
  "javascript": "// Generated JS...",
  "confidence": 0.85,
  "should_use": true,
  "training_metrics": {
    "loss": 0.15,
    "accuracy": 0.85,
    "learning_rate": 0.001,
    "epoch": 42,
    "latent_dim": 256
  },
  "timestamp": 1704067200
}
```

**Error Response** (400 Bad Request):
```json
{
  "error": "Invalid request format",
  "details": [
    {
      "loc": ["features"],
      "msg": "Expected 48 features, got 47"
    }
  ]
}
```

**Error Response** (500 Internal Server Error):
```json
{
  "error": "Feature encoding failed"
}
```

---

### 3. Feedback Collection

**Endpoint**: `POST /api/v1/feedback`

**Purpose**: Submit rendering quality feedback for model training

**Request Body**:
```json
{
  "url": "https://example.com",
  "overall_quality": 0.85,
  "html_similarity": 0.88,
  "css_accuracy": 0.82,
  "layout_similarity": 0.85,
  "matched_elements": 45,
  "mismatched_elements": 5,
  "feedback_text": "Generated layout closely matches original",
  "session_id": "sess-12345",
  "timestamp": 1704067200
}
```

**Quality Metrics**:
- `overall_quality`: Overall quality score (0.0-1.0)
  - Weighted combination of all component scores
  - Used for loss computation
- `html_similarity`: HTML structure similarity (0.0-1.0)
  - Measures semantic correctness
  - Affects HTML component loss
- `css_accuracy`: CSS rule accuracy (0.0-1.0)
  - Measures visual fidelity
  - Affects styling component loss
- `layout_similarity`: Layout matching (0.0-1.0)
  - Measures spatial correctness
  - Affects layout component loss
- `matched_elements`: Count of correctly generated elements
- `mismatched_elements`: Count of incorrectly generated elements

**Response** (200 OK):
```json
{
  "status": "ok",
  "quality_score": 0.85,
  "buffer_size": 5,
  "buffer_ready": false,
  "learner_metrics": {
    "average_loss": 0.23,
    "recent_average_loss": 0.18,
    "best_loss": 0.12,
    "worst_loss": 0.45,
    "loss_std": 0.08,
    "average_quality": 0.78,
    "recent_average_quality": 0.82,
    "best_quality": 0.95,
    "worst_quality": 0.52,
    "quality_std": 0.12,
    "update_count": 42,
    "feedback_count": 128,
    "convergence": 0.75,
    "improvement": 0.23,
    "learning_rate": 0.001,
    "weight_matrix_norm": 2.34
  },
  "timestamp": 1704067200
}
```

**Error Response** (400 Bad Request):
```json
{
  "error": "Invalid request format",
  "details": [
    {
      "loc": ["overall_quality"],
      "msg": "Value must be between 0.0 and 1.0"
    }
  ]
}
```

---

### 4. Metrics

**Endpoint**: `GET /metrics`

**Purpose**: Get server and model metrics

**Response** (200 OK):
```json
{
  "timestamp": 1704067200,
  "uptime_seconds": 3600.0,
  "requests": {
    "total": 256,
    "success": 248,
    "error": 8,
    "success_rate": "96.88%"
  },
  "models": {
    "feature_encoder": "loaded",
    "code_generator": "loaded",
    "online_learner": "loaded"
  },
  "configuration": {
    "feature_dim": 48,
    "latent_dim": 256,
    "learning_rate": 0.001,
    "batch_size": 32
  }
}
```

---

### 5. Root Information

**Endpoint**: `GET /`

**Purpose**: Get API information

**Response** (200 OK):
```json
{
  "name": "BrowserAI Learning System API",
  "version": "1.0.0",
  "description": "REST API for AI-powered website learning",
  "endpoints": {
    "health": "GET /api/v1/health",
    "generate": "POST /api/v1/generate",
    "feedback": "POST /api/v1/feedback",
    "metrics": "GET /metrics"
  },
  "documentation": "See WEEK6_API_SPEC.md for full documentation",
  "uptime": {
    "start_time": "2024-01-01T12:00:00",
    "uptime_seconds": 3600.0
  }
}
```

---

## Data Models

### FeaturePacketRequest

Sent from Rust to Python for code generation.

```python
class FeaturePacketRequest(BaseModel):
    url: str                          # Source website URL
    features: List[float]             # 48-dimensional feature vector
    website_intent: str               # Intent category
    design_style: str                 # Design style category
    feedback: Optional[Dict] = None   # Optional previous feedback
    timestamp: int                    # Unix timestamp
    session_id: str                   # Unique session identifier
```

### GeneratedCodeResponse

Sent from Python to Rust with generated code.

```python
class GeneratedCodeResponse(BaseModel):
    html: str                         # Generated HTML
    css: str                          # Generated CSS
    javascript: str                   # Generated JavaScript
    confidence: float                 # Generation confidence (0.0-1.0)
    should_use: bool                  # Whether to use generated code
    training_metrics: Optional[Dict]  # Training metadata
    timestamp: int                    # Unix timestamp
```

### FeedbackPacketRequest

Sent from Rust to Python with quality feedback.

```python
class FeedbackPacketRequest(BaseModel):
    url: str                          # Website URL
    overall_quality: float            # Overall quality score (0.0-1.0)
    html_similarity: float            # HTML structure similarity
    css_accuracy: float               # CSS accuracy score
    layout_similarity: float          # Layout matching score
    matched_elements: int             # Count of matched elements
    mismatched_elements: int          # Count of mismatched elements
    feedback_text: Optional[str]      # Human-readable feedback
    session_id: str                   # Unique session identifier
    timestamp: int                    # Unix timestamp
```

### HealthResponse

Response from health check endpoint.

```python
class HealthResponse(BaseModel):
    status: str                       # "healthy" or "unhealthy"
    timestamp: int                    # Unix timestamp
    uptime_seconds: float             # Server uptime in seconds
    models_loaded: int                # Number of loaded models
    version: str                      # API version
```

---

## System Components

### FeatureEncoder

**Purpose**: Convert 48-dimensional feature vectors to 256-dimensional latent vectors

**Algorithm**:
1. Normalize features to [0, 1] using min-max scaling
2. Matrix multiply: features @ encoding_matrix (48×256)
3. Add intent embedding: intent_embeddings[website_intent]
4. Add style embedding: style_embeddings[design_style]
5. Apply ReLU activation
6. L2 normalize output

**Outputs**: 256-dimensional latent vector

### CodeGenerator

**Purpose**: Generate HTML/CSS/JavaScript from latent vectors

**Features**:
- Template-based code generation
- Latent vector → generation parameters
- Three levels of complexity: simple, moderate, complex
- Confidence scoring based on vector properties
- CSS color, typography, spacing, animations selection

**Outputs**:
- HTML structure (DOCTYPE, semantic tags, content)
- CSS styling (colors, fonts, animations, layout)
- JavaScript (initialization, navigation, animations)

### OnlineLearner

**Purpose**: Update model weights based on feedback

**Algorithm**:
- Loss computation from quality metrics
- Gradient computation from reconstruction error
- Adam optimizer for weight updates
- Convergence tracking
- Improvement metrics

**Methods**:
- `process_feedback()`: Main feedback processing
- `get_metrics()`: Training statistics
- `adaptive_learning_rate()`: Dynamic LR adjustment

### FeedbackBuffer

**Purpose**: Batch feedback for efficient processing

**Features**:
- Accumulates feedback samples
- Triggers batch processing at batch_size
- Prevents buffer overflow with max_buffer_size
- FIFO ordering

---

## Configuration

Environment variables for server configuration:

```bash
# Server Configuration
FLASK_DEBUG=False              # Debug mode
API_HOST=0.0.0.0              # Server host
API_PORT=5000                 # Server port
LOG_LEVEL=INFO                # Logging level

# Model Configuration
FEATURE_DIM=48                # Input feature dimension
LATENT_DIM=256                # Latent vector dimension

# Learning Configuration
LEARNING_RATE=0.001           # Optimizer learning rate
BATCH_SIZE=32                 # Batch size for processing
MAX_QUEUE_SIZE=1000           # Max feedback queue size
```

---

## Performance Characteristics

### Latency

- Feature encoding: ~1-5ms
- Code generation: ~5-10ms
- Feedback processing: ~2-5ms
- Total request latency: ~10-20ms

### Throughput

- Requests per second: 50+ (single-threaded)
- Feature processing: 1000s per second
- Feedback buffer: 1000 items max

### Memory Usage

- Feature encoder: ~2MB
- Code generator: ~5MB
- Online learner: ~1MB
- Typical process: 50-100MB total

---

## Error Handling

All errors return appropriate HTTP status codes:

| Code | Meaning | Response |
|------|---------|----------|
| 200 | Success | Standard response with data |
| 400 | Bad Request | Validation errors |
| 500 | Internal Error | Processing error |
| 503 | Unavailable | Server unhealthy |

---

## Integration with Rust

### Request Flow

1. Rust extracts features from website
2. Rust sends FeaturePacketRequest to `/api/v1/generate`
3. Python encodes features and generates code
4. Python returns GeneratedCodeResponse
5. Rust renders generated code
6. Rust evaluates rendering quality
7. Rust sends FeedbackPacketRequest to `/api/v1/feedback`
8. Python updates model with feedback

### Serialization

All communication uses JSON with standard serialization:
- Lists/Arrays: `[...]`
- Objects: `{...}`
- Numbers: IEEE 754 floats for features, integers for counts
- Strings: UTF-8 encoded

---

## Testing

Run the server with test client:

```python
# test_api_client.py
import requests

# Test health
response = requests.get('http://127.0.0.1:5000/api/v1/health')
assert response.status_code == 200

# Test code generation
request_data = {
    "url": "https://example.com",
    "features": [0.1] * 48,
    "website_intent": "blog",
    "design_style": "modern",
    "session_id": "test-1",
    "timestamp": 1704067200
}
response = requests.post('http://127.0.0.1:5000/api/v1/generate', json=request_data)
assert response.status_code == 200
assert "html" in response.json()

# Test feedback
feedback_data = {
    "url": "https://example.com",
    "overall_quality": 0.85,
    "html_similarity": 0.88,
    "css_accuracy": 0.82,
    "layout_similarity": 0.85,
    "matched_elements": 45,
    "mismatched_elements": 5,
    "session_id": "test-1",
    "timestamp": 1704067200
}
response = requests.post('http://127.0.0.1:5000/api/v1/feedback', json=feedback_data)
assert response.status_code == 200
```

---

## Version History

- **v1.0.0** (Week 6): Initial release with feature encoding, code generation, and online learning
