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
├── scripts/           # Training scripts
│   ├── prepare_data.py      # Data preparation script
│   ├── train_html_parser.py # HTML parser model training
│   ├── train_css_parser.py  # CSS parser model training
│   └── train_js_parser.py   # JS parser model training
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

```bash
# Train HTML parser model
python scripts/train_html_parser.py

# Train CSS parser model
python scripts/train_css_parser.py

# Train JavaScript parser model
python scripts/train_js_parser.py
```

### 4. Export to ONNX

The training scripts automatically export trained models to ONNX format in the `models/` directory.

### 5. Deploy to BrowerAI

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

### CSS Parser Model
- **Architecture**: LSTM with attention
- **Input**: Tokenized CSS rules (max length 256)
- **Output**: Optimization suggestions, unused rule detection
- **Training Data**: 5,000+ CSS files
- **Metrics**: Optimization accuracy, compression ratio

### JS Parser Model
- **Architecture**: Bidirectional LSTM
- **Input**: JavaScript token sequences (max length 1024)
- **Output**: Code patterns, optimization hints
- **Training Data**: 8,000+ JavaScript files
- **Metrics**: Pattern detection accuracy, tokenization speed

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
