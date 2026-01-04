# BrowerAI Model Training - Quick Start Guide

This guide will help you set up and run the model training pipeline for BrowerAI.

## Prerequisites

- Python 3.8 or later
- pip (Python package manager)
- 4GB+ RAM for training
- (Optional) CUDA-capable GPU for faster training

## Step 1: Set up Python Environment

It's recommended to use a virtual environment:

```bash
cd training

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Prepare Training Data

Generate synthetic training data:

```bash
python scripts/prepare_data.py --num-samples 1000
```

This will create:
- `data/html/train.json`, `data/html/val.json`, `data/html/test.json`
- `data/css/train.json`, `data/css/val.json`, `data/css/test.json`
- `data/js/train.json`, `data/js/val.json`, `data/js/test.json`

## Step 3: Train Models

Train each model (this may take several minutes):

```bash
# Train HTML parser model
python scripts/train_html_parser.py --epochs 10

# Train CSS parser model
python scripts/train_css_parser.py --epochs 10

# Train JavaScript parser model
python scripts/train_js_parser.py --epochs 15
```

Training options:
- `--epochs N`: Number of training epochs (default: 10-15)
- `--batch-size N`: Batch size (default: 32)
- `--lr FLOAT`: Learning rate (default: 0.001)
- `--data-dir PATH`: Custom data directory
- `--output-dir PATH`: Custom output directory

## Step 4: Verify Model Export

After training, you should see ONNX models in the `models/` directory:

```bash
ls -lh models/*.onnx
```

Expected output:
- `html_parser_v1.onnx`
- `css_optimizer_v1.onnx`
- `js_analyzer_v1.onnx`

## Step 5: Deploy to BrowerAI

Copy trained models to BrowerAI:

```bash
# Copy ONNX models
cp models/*.onnx ../models/local/

# Verify models are in place
ls -lh ../models/local/
```

Update `../models/model_config.toml`:

```toml
[[models]]
name = "html_parser_v1"
model_type = "HtmlParser"
path = "html_parser_v1.onnx"
description = "HTML structure prediction model"
version = "1.0.0"

[[models]]
name = "css_optimizer_v1"
model_type = "CssParser"
path = "css_optimizer_v1.onnx"
description = "CSS optimization model"
version = "1.0.0"

[[models]]
name = "js_analyzer_v1"
model_type = "JsParser"
path = "js_analyzer_v1.onnx"
description = "JavaScript analysis model"
version = "1.0.0"
```

## Step 6: Build and Test BrowerAI

Build with AI features enabled:

```bash
cd ..
cargo build --features ai --release
cargo run --features ai
```

## Training Tips

### Faster Training
- Use a GPU if available (requires CUDA setup)
- Increase batch size: `--batch-size 64`
- Reduce epochs for quick testing: `--epochs 5`

### Better Accuracy
- Increase training samples: `--num-samples 5000`
- Train for more epochs: `--epochs 50`
- Fine-tune learning rate: `--lr 0.0005`

### Monitor Training
All training scripts show:
- Training loss and accuracy
- Validation loss and accuracy
- Best model checkpoint saving

## Troubleshooting

### Issue: ImportError or ModuleNotFoundError
**Solution**: Ensure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: Out of memory during training
**Solution**: Reduce batch size:
```bash
python scripts/train_html_parser.py --batch-size 16
```

### Issue: Training is very slow
**Solution**: 
- Reduce number of samples: `--num-samples 500`
- Reduce epochs: `--epochs 5`
- Use GPU if available

### Issue: ONNX export fails
**Solution**: Ensure PyTorch and ONNX versions are compatible:
```bash
pip install --upgrade torch onnx onnxruntime
```

## Advanced Usage

### Custom Training Data

Add your own HTML/CSS/JS files to `data/` directories and modify the data preparation script to include them.

### Model Architecture Changes

Edit the training scripts to experiment with different:
- Network architectures (LSTM, Transformer, etc.)
- Hyperparameters (hidden dimensions, layers)
- Loss functions and optimizers

### Model Evaluation

Create an evaluation script to test on real-world data:

```python
import torch
from train_html_parser import HTMLParserModel, HTMLDataset

# Load model
model = HTMLParserModel(vocab_size)
model.load_state_dict(torch.load('models/html_parser_best.pth'))
model.eval()

# Test on custom data
# ... your evaluation code ...
```

## Next Steps

1. Experiment with different model architectures
2. Collect real-world training data from websites
3. Implement online learning for continuous improvement
4. Create ensemble models for better accuracy
5. Optimize models for inference speed

## Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [ONNX Documentation](https://onnx.ai/onnx/)
- [BrowerAI Main README](../README.md)
- [BrowerAI Roadmap](../ROADMAP.md)

For questions or issues, please open a GitHub issue or refer to the main documentation.
