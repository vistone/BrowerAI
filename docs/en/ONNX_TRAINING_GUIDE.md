# BrowerAI ONNX Model Library and Training Guide

## 📦 Required Model Libraries

BrowerAI requires a two-tier model library architecture:

### 1. Training Environment (Python)

**Location**: `training/`

**Purpose**: Use PyTorch to train models and export them to ONNX format

**Core Libraries**:
- **PyTorch** (⭐⭐⭐⭐⭐ Recommended): Train deep learning models
- **ONNX**: Model format standard
- **ONNXRuntime**: Python-side model validation

**Why PyTorch**:
- ✅ Most mature Python ML framework
- ✅ Simple ONNX export (`torch.onnx.export`)
- ✅ Rich ecosystem (pre-trained models, tools)
- ✅ Best community support
- ✅ Debugging-friendly

**Alternatives**:
- TensorFlow + tf2onnx (also good, but slightly weaker ecosystem)
- scikit-learn + skl2onnx (suitable for traditional ML)

### 2. Inference Environment (Rust)

**Location**: `src/ai/`, `models/local/`

**Purpose**: Load ONNX models and perform high-speed inference

**Core Libraries**:
- **ort** (⭐⭐⭐⭐⭐ Currently used): ONNX Runtime for Rust
  - GitHub: https://github.com/pykeio/ort
  - Documentation: https://docs.rs/ort/

**Why ort**:
- ✅ Official ONNX Runtime Rust bindings
- ✅ Microsoft-supported, stable and reliable
- ✅ CPU/GPU acceleration
- ✅ Friendly API, type-safe
- ✅ Actively maintained

**Alternatives**:
- `tract` (Pure Rust ML inference): No C++ dependencies, but limited model support

## 🎯 Model Type Design

BrowerAI supports the following model types:

### 1. HtmlParser
- **Purpose**: Analyze HTML structure and complexity
- **Input**: HTML text (tokenized)
- **Output**: Complexity score (0.0-1.0)

### 2. CssParser
- **Purpose**: CSS optimization and deduplication
- **Input**: CSS rules
- **Output**: Optimization suggestions

### 3. JsParser
- **Purpose**: JavaScript pattern recognition and obfuscation detection
- **Input**: JavaScript code (tokenized)
- **Output**: Pattern classification, obfuscation score

### 4. LayoutOptimizer
- **Purpose**: Optimize layout calculations
- **Input**: DOM tree structure
- **Output**: Layout optimization hints

### 5. RenderingOptimizer
- **Purpose**: Optimize rendering process
- **Input**: Render tree
- **Output**: Rendering optimization strategy

### 6. JsDeobfuscator
- **Purpose**: Detect and analyze obfuscated JavaScript
- **Input**: JavaScript code
- **Output**: Obfuscation type, complexity score

## 🔧 Model Training Workflow

### Step 1: Collect Feedback Data

```bash
# Visit websites to collect data
cargo run -- --learn https://example.com https://www.mozilla.org

# Check collected data
ls -lh training/data/feedback_*.json
```

### Step 2: Prepare Training Data

```bash
cd training

# Extract features from feedback data
python scripts/extract_features.py

# Verify feature data
cat features/*.jsonl | head -5
```

### Step 3: Train Model

```bash
# Train HTML parser model
python scripts/train_html_parser_v2.py --epochs 10

# Train CSS parser model
python scripts/train_css_parser.py --epochs 10

# Train JS deobfuscator model
python scripts/train_js_deobfuscator.py --epochs 10
```

### Step 4: Export to ONNX

Models are automatically exported to ONNX format during training:

```bash
ls -lh training/models/*.onnx
```

### Step 5: Deploy Models

```bash
# Copy trained models to deployment directory
cp training/models/*.onnx models/local/

# Update model configuration
cat > models/model_config.toml << 'EOF'
[[models]]
name = "html_parser_v2"
model_type = "HtmlParser"
path = "html_parser_v2.onnx"
description = "HTML complexity analyzer v2"
version = "2.0.0"
priority = 100
EOF
```

### Step 6: Test Deployed Models

```bash
# Build with AI features enabled
cargo build --features ai

# Test model inference
cargo run -- --ai-report

# Test on real websites
cargo run -- --learn https://example.com
```

## 📈 Model Performance

### Monitoring

```bash
# View model health status
cargo run -- --ai-report

# Check inference metrics
grep "model_inference" training/data/feedback_*.json | jq .
```

### Optimization Tips

1. **Batch Size**: Increase for faster training (with more memory)
2. **Learning Rate**: Adjust based on loss convergence
3. **Model Architecture**: Simplify for faster inference
4. **Data Quality**: Clean and diverse training data improves accuracy
5. **Validation**: Always validate on separate test set

## 🔍 Model Architecture

### HTML Parser Model

```python
class HtmlComplexityModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        output = self.sigmoid(self.fc(hidden[-1]))
        return output
```

### CSS Parser Model

```python
class CssDeduplicationModel(nn.Module):
    def __init__(self, vocab_size=500, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.sigmoid(self.fc(hidden[-1]))
```

### JS Deobfuscator Model

```python
class JsObfuscationDetector(nn.Module):
    def __init__(self, vocab_size=2000, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, 256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 3)  # 3 obfuscation types
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.softmax(self.fc(hidden[-1]))
```

## 🛠️ Troubleshooting

### Issue: ONNX Export Fails
- Check PyTorch and ONNX versions compatibility
- Ensure model uses supported operations
- Verify input/output shapes

### Issue: Rust Inference Fails
- Verify ONNX model file exists
- Check model configuration syntax
- Ensure ort crate version matches

### Issue: Poor Model Performance
- Collect more diverse training data
- Increase model complexity (more layers/units)
- Tune hyperparameters (learning rate, epochs)
- Add validation and early stopping

## 📚 See Also

- [Training Quick Start](../../training/QUICKSTART.md) - Quick training guide
- [Learning Guide](LEARNING_GUIDE.md) - Data collection guide
- [Implementation Guide](IMPLEMENTATION_GUIDE.md) - Technical details
