# BrowerAI Model Training Pipeline

This directory contains the model training pipeline for BrowerAI's AI-powered browser capabilities.

## Directory Structure

```
training/
├── data/               # Training data storage
│   ├── html/          # HTML samples for training
│   ├── css/           # CSS samples for training
│   └── js/            # JavaScript samples for training
├── models/            # Trained model outputs
├── scripts/           # Training and testing scripts
│   ├── prepare_data.py                       # Data preparation script
│   ├── train_html_parser.py                  # HTML parser model training
│   ├── train_html_parser_v2.py               # HTML parser v2 model training
│   ├── train_css_parser.py                   # CSS parser model training
│   ├── train_css_optimizer.py                # CSS optimizer model training
│   ├── train_css_deduplication.py            # CSS rule deduplication model
│   ├── train_css_selector_optimizer.py       # CSS selector optimization model
│   ├── train_css_minifier.py                 # CSS minification model
│   ├── train_js_parser.py                    # JS parser model training
│   ├── train_js_optimizer.py                 # JS optimizer model training
│   ├── train_js_tokenizer_enhancer.py        # JS tokenization enhancer model
│   ├── train_js_ast_predictor.py             # JS AST predictor model
│   ├── train_js_optimization_suggestions.py  # JS optimization suggestions model
│   ├── train_layout_optimizer.py             # Layout optimizer model training
│   ├── train_paint_optimizer.py              # Paint optimizer model training
│   ├── benchmark_models.py                   # Model benchmarking
│   ├── test_real_world_websites.py           # Real-world website testing (Phase 2.4)
│   ├── measure_accuracy.py                   # Accuracy measurement (Phase 2.4)
│   ├── profile_performance.py                # Performance profiling (Phase 2.4)
│   ├── test_visual_regression.py             # Visual regression testing (Phase 3.4)
│   ├── benchmark_rendering.py                # Rendering performance benchmarking (Phase 3.4)
│   ├── compare_cross_browser.py              # Cross-browser comparison (Phase 3.4)
│   ├── test_rendering_realworld.py           # Real-world rendering testing (Phase 3.4)
│   └── collect_data.py                       # Data collection script
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Quick Start

### 1. Install Dependencies

```bash
cd training
pip install -r requirements.txt
```

### 2. Prepare Training Data

```bash
python scripts/prepare_data.py
```

This script will:
- Download sample HTML/CSS/JS from popular websites
- Generate synthetic training data
- Create train/validation/test splits
- Store preprocessed data in the `data/` directory

### 3. Train Models

Train each model type:

#### HTML Models
```bash
# Train HTML parser model
python scripts/train_html_parser.py

# Train HTML parser v2 model
python scripts/train_html_parser_v2.py
```

#### CSS Models
```bash
# Train CSS parser model
python scripts/train_css_parser.py

# Train CSS optimizer model (comprehensive)
python scripts/train_css_optimizer.py

# Train CSS rule deduplication model
python scripts/train_css_deduplication.py

# Train CSS selector optimization model
python scripts/train_css_selector_optimizer.py

# Train CSS minification model
python scripts/train_css_minifier.py
```

#### JavaScript Models
```bash
# Train JavaScript parser model
python scripts/train_js_parser.py

# Train JavaScript optimizer model (comprehensive)
python scripts/train_js_optimizer.py

# Train JavaScript tokenization enhancer model
python scripts/train_js_tokenizer_enhancer.py

# Train JavaScript AST predictor model
python scripts/train_js_ast_predictor.py

# Train JavaScript optimization suggestions model
python scripts/train_js_optimization_suggestions.py
```

#### Rendering Models
```bash
# Train layout optimizer model
python scripts/train_layout_optimizer.py

# Train paint optimizer model
python scripts/train_paint_optimizer.py
```

### 4. Export to ONNX

The training scripts automatically export trained models to ONNX format in the `models/` directory.

### 5. Testing & Validation (Phase 2.4)

After training models, run comprehensive testing and validation:

#### Test on Real-World Websites
```bash
# Test models on popular websites
python scripts/test_real_world_websites.py --num-sites 10

# Test with specific models directory
python scripts/test_real_world_websites.py --num-sites 5 --models-dir ../models
```

Fetches content from real websites and tests model performance on production data.

#### Measure Accuracy Improvements
```bash
# Measure accuracy vs traditional parsing
python scripts/measure_accuracy.py

# With custom paths
python scripts/measure_accuracy.py --models-dir ../models --data-dir data
```

Calculates precision, recall, F1 score, and accuracy improvements over baseline.

#### Profile Performance Impact
```bash
# Profile model inference performance
python scripts/profile_performance.py --iterations 100

# Extended profiling
python scripts/profile_performance.py --iterations 1000
```

Measures inference time, memory usage, throughput, and identifies bottlenecks.

#### Benchmark Models
```bash
# Comprehensive benchmarking suite
python scripts/benchmark_models.py --models-dir ../models --data-dir data
```

Compares AI models against traditional parsing methods.

### 6. Rendering Engine Testing (Phase 3.4)

After implementing the rendering engine, run comprehensive rendering tests:

#### Visual Regression Testing
```bash
# Test for visual regressions in rendering
python scripts/test_visual_regression.py

# With custom directories
python scripts/test_visual_regression.py --baseline-dir baselines --output-dir outputs
```

Creates baseline images and compares rendering output to detect visual changes.

#### Rendering Performance Benchmarking
```bash
# Benchmark rendering performance
python scripts/benchmark_rendering.py --iterations 100

# Extended benchmarking
python scripts/benchmark_rendering.py --iterations 1000
```

Measures layout calculation, paint operations, and overall rendering time.

#### Cross-Browser Comparison
```bash
# Compare rendering with major browsers
python scripts/compare_cross_browser.py

# Compare with specific browsers
python scripts/compare_cross_browser.py --browsers Chrome Firefox Safari
```

Validates rendering compatibility across different browser engines.

#### Real-World Rendering Testing
```bash
# Test rendering on real websites
python scripts/test_rendering_realworld.py --num-sites 10

# Custom site count
python scripts/test_rendering_realworld.py --num-sites 5
```

Tests the rendering engine on production websites to validate real-world performance.

### 7. Deploy to BrowerAI

Copy the ONNX models to the main models directory:

```bash
cp models/*.onnx ../models/local/
```

Update `../models/model_config.toml` with your model configurations.

## Model Architectures

### HTML Parser Model
- **Architecture**: Transformer-based sequence model
- **Input**: Tokenized HTML sequences (max length 512)
- **Output**: Structure predictions, malformed HTML fixes
- **Training Data**: 10,000+ HTML documents
- **Metrics**: Structure accuracy, parsing speed

### CSS Parser Models

#### CSS Parser (Basic)
- **Architecture**: LSTM with attention
- **Input**: Tokenized CSS rules (max length 256)
- **Output**: Optimization suggestions, unused rule detection
- **Training Data**: 5,000+ CSS files
- **Metrics**: Optimization accuracy, compression ratio

#### CSS Optimizer (Comprehensive)
- **Architecture**: Multi-layer feedforward network
- **Input**: 18-dimensional CSS features (rules, selectors, properties, etc.)
- **Output**: 4 optimization scores (deduplication, selector simplification, minification safety, merge opportunity)
- **Training Data**: 4,000+ synthetic CSS samples
- **Metrics**: Optimization accuracy, prediction speed

#### CSS Deduplication Model
- **Architecture**: Deep neural network with batch normalization
- **Input**: 15-dimensional CSS rule pair features
- **Output**: 3 scores (duplicate probability, merge opportunity, safety confidence)
- **Training Data**: 5,000+ rule pair samples
- **Metrics**: Duplicate detection accuracy, false positive rate

#### CSS Selector Optimizer
- **Architecture**: Deep neural network with batch normalization
- **Input**: 16-dimensional selector features (length, specificity, combinators, etc.)
- **Output**: 4 scores (complexity, simplification potential, performance impact, specificity balance)
- **Training Data**: 6,000+ selector samples
- **Metrics**: Simplification accuracy, performance prediction

#### CSS Minifier
- **Architecture**: Multi-layer network with dropout
- **Input**: 17-dimensional CSS file features
- **Output**: 5 scores (whitespace removal safety, comment removal safety, shorthand potential, value optimization, overall minification score)
- **Training Data**: 5,500+ CSS file samples
- **Metrics**: Minification safety, size reduction prediction

### JS Parser Models

#### JS Parser (Basic)
- **Architecture**: Bidirectional LSTM
- **Input**: JavaScript token sequences (max length 1024)
- **Output**: Code patterns, optimization hints
- **Training Data**: 8,000+ JavaScript files
- **Metrics**: Pattern detection accuracy, tokenization speed

#### JS Optimizer (Comprehensive)
- **Architecture**: Multi-layer feedforward network
- **Input**: 20-dimensional JS features (tokens, statements, complexity, etc.)
- **Output**: 5 optimization scores (minification safety, dead code, optimization potential, bundle score, async conversion)
- **Training Data**: 4,500+ synthetic JS samples
- **Metrics**: Optimization accuracy, prediction speed

#### JS Tokenization Enhancer
- **Architecture**: Deep neural network with batch normalization
- **Input**: 18-dimensional token features (length, character ratios, context, etc.)
- **Output**: 4 scores (token validity, type confidence, correction needed, syntax complexity)
- **Training Data**: 7,000+ token samples
- **Metrics**: Tokenization accuracy, error detection rate

#### JS AST Predictor
- **Architecture**: Deep neural network with multiple hidden layers
- **Input**: 20-dimensional code features (keywords, operators, brackets, etc.)
- **Output**: 5 scores (statement type probability, expression complexity, nesting depth, AST confidence, declaration pattern)
- **Training Data**: 8,000+ code snippet samples
- **Metrics**: AST prediction accuracy, parsing speedup

#### JS Optimization Suggestions
- **Architecture**: Deep neural network with dropout
- **Input**: 22-dimensional code analysis features
- **Output**: 6 optimization scores (loop optimization, function optimization, memory optimization, modern syntax upgrade, async conversion, bundle size reduction)
- **Training Data**: 7,500+ code samples
- **Metrics**: Suggestion accuracy, optimization impact

## Training Configuration

Default training hyperparameters:

```python
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
OPTIMIZER = "Adam"
VALIDATION_SPLIT = 0.2
```

Modify these in the training scripts as needed.

## Data Sources

Training data is collected from:
1. **Public Datasets**: Common Crawl, GitHub repositories
2. **Synthetic Data**: Generated test cases
3. **Validation Sets**: Hand-curated examples

All data is preprocessed to remove:
- Personal information
- Authentication tokens
- Proprietary code

## Model Evaluation

After training, evaluate models with:

```bash
python scripts/evaluate_models.py
```

This generates:
- Accuracy metrics
- Inference speed benchmarks
- Model size reports
- Comparison with baseline parsers

## Advanced Usage

### Custom Training Data

Add your own training data:

1. Place files in `data/html/`, `data/css/`, or `data/js/`
2. Run `python scripts/prepare_data.py --custom-data`
3. Train models as usual

### Hyperparameter Tuning

Use the configuration files:

```bash
python scripts/train_html_parser.py --config configs/html_parser_config.json
```

### Transfer Learning

Fine-tune pre-trained models:

```bash
python scripts/train_html_parser.py --pretrained models/html_parser_base.onnx
```

## Performance Optimization

Tips for faster training:
1. Use GPU acceleration (CUDA)
2. Increase batch size (if memory allows)
3. Use mixed precision training
4. Enable data augmentation

## Troubleshooting

### Common Issues

**Issue**: Out of memory during training
**Solution**: Reduce batch size or sequence length

**Issue**: Model not converging
**Solution**: Adjust learning rate or increase epochs

**Issue**: ONNX export fails
**Solution**: Ensure all operations are ONNX-compatible

## Contributing

To contribute new training scripts or model architectures:

1. Create a new script in `scripts/`
2. Follow the existing naming convention
3. Include documentation and comments
4. Test the model export to ONNX
5. Update this README with usage instructions

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [ONNX Export Guide](https://pytorch.org/docs/stable/onnx.html)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [BrowerAI Main Documentation](../README.md)

## Next Steps

After completing the initial training pipeline:

1. Collect more diverse training data
2. Experiment with different architectures
3. Implement ensemble methods
4. Add online learning capabilities
5. Create model versioning system

For questions or issues, please open a GitHub issue.
