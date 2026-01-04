# BrowerAI Models Directory

This directory contains the local model library for BrowerAI's AI-powered browser capabilities.

## Directory Structure

```
models/
├── local/              # Local ONNX model files
│   └── .gitkeep       # (Place your .onnx files here)
└── model_config.toml  # Model configuration file
```

## Model Types

BrowerAI supports the following types of AI models:

### 1. HtmlParser Models
- **Purpose**: Understanding HTML structure and semantics
- **Input**: Tokenized HTML content
- **Output**: Enhanced parsing decisions, structure predictions

### 2. CssParser Models
- **Purpose**: CSS optimization and rule analysis
- **Input**: CSS rules and selectors
- **Output**: Optimization suggestions, unused rule detection

### 3. JsParser Models
- **Purpose**: JavaScript code analysis and tokenization
- **Input**: JavaScript code tokens
- **Output**: Code patterns, optimization hints

### 4. LayoutOptimizer Models
- **Purpose**: Optimizing layout calculations
- **Input**: DOM tree and CSS rules
- **Output**: Optimized layout decisions

### 5. RenderingOptimizer Models
- **Purpose**: Rendering performance optimization
- **Input**: Render tree and viewport information
- **Output**: Rendering strategy optimizations

## Adding Models

1. **Obtain or train an ONNX model** for one of the supported types
2. **Place the .onnx file** in the `local/` directory
3. **Update `model_config.toml`** with the model information:

```toml
[[models]]
name = "my_html_parser"
model_type = "HtmlParser"
path = "my_html_parser.onnx"
description = "Custom HTML parsing model"
version = "1.0.0"
```

4. **Restart the application** to load the new model

## Model Training

To train custom models for BrowerAI:

1. **Collect training data** from HTML/CSS/JS parsing tasks
2. **Design your model architecture** using PyTorch, TensorFlow, or other frameworks
3. **Train the model** on your dataset
4. **Export to ONNX format**:
   - PyTorch: Use `torch.onnx.export()`
   - TensorFlow: Use `tf2onnx`
5. **Test with ONNX Runtime** to ensure compatibility
6. **Add to BrowerAI** following the steps above

## Example Model Export (PyTorch)

```python
import torch
import torch.onnx

# Assuming you have a trained model
model = YourModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 128)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "html_parser_v1.onnx",
    export_params=True,
    opset_version=15,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

## Resources

- [ONNX Documentation](https://onnx.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [TensorFlow to ONNX](https://github.com/onnx/tensorflow-onnx)

## Notes

- Model files (.onnx) are gitignored by default due to their size
- Keep models lightweight for fast inference
- Test models thoroughly before deploying in production
- Consider model versioning for backward compatibility
