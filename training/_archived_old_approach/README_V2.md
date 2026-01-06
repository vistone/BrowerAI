# BrowerAI Training Framework v2.0

Modern, unified training pipeline for BrowerAI models following best practices from GPT, BERT, T5, and recent ML research.

## 🎯 Key Improvements

### Architecture
- **Unified Framework**: Single codebase for all model types (HTML/CSS/JS/deobfuscation)
- **Modern Architectures**: Transformer-based models with attention mechanisms
- **Multi-task Learning**: Shared representations across related tasks
- **Modular Design**: Easy to extend with new models and tasks

### Training Features
- **Automatic Mixed Precision (AMP)**: Faster training on GPUs
- **Gradient Accumulation**: Simulate large batch sizes
- **Learning Rate Warmup**: Stable training for large models
- **Advanced Optimizers**: AdamW with weight decay
- **Multiple Schedulers**: Cosine annealing, linear decay, plateau
- **Gradient Clipping**: Prevent gradient explosions

### Modern Techniques
- **Copy Mechanism**: For deobfuscation (preserve identifiers)
- **Beam Search**: Better generation quality
- **Contrastive Learning**: Obfuscation-invariant representations
- **Curriculum Learning**: Easy → hard training progression
- **Meta-learning**: Few-shot adaptation to new patterns

## 📁 New Structure

```
training/
├── core/                           # Core framework (NEW)
│   ├── models/                     # Neural architectures
│   │   ├── base.py                # Base model class
│   │   ├── transformer.py         # Transformer components
│   │   ├── attention.py           # Attention mechanisms
│   │   ├── parsers.py             # HTML/CSS/JS parsers
│   │   └── deobfuscator.py        # Deobfuscation models
│   ├── data/                      # Data pipeline
│   │   ├── tokenizers.py          # Modern tokenizers (BPE, WordPiece)
│   │   ├── datasets.py            # Dataset classes
│   │   └── augmentation.py        # Data augmentation
│   ├── trainers/                  # Training loop
│   │   ├── trainer.py             # Unified trainer
│   │   └── callbacks.py           # Training callbacks
│   └── utils/                     # Utilities
│
├── configs/                        # YAML configurations (NEW)
│   ├── html_parser.yaml
│   ├── css_parser.yaml
│   ├── js_parser.yaml
│   └── deobfuscator.yaml
│
├── train_unified.py               # Unified training script (NEW)
├── scripts/                       # Old scripts (deprecated)
└── data/                          # Training data
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd training
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# For HTML parser
python scripts/prepare_html_data.py

# For deobfuscator
python scripts/prepare_deobfuscation_data.py
```

### 3. Train a Model

```bash
# Train HTML parser (default config)
python train_unified.py --task html_parser

# Train deobfuscator with custom settings
python train_unified.py \
    --task deobfuscator \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.0003 \
    --use-amp

# Train with specific config file
python train_unified.py \
    --task html_parser \
    --config configs/html_parser.yaml
```

### 4. Export to ONNX

Models are automatically exported to ONNX after training:
```
models/local/
├── html_parser_v2.onnx
├── html_parser_v2.json
├── deobfuscator_v2.onnx
└── deobfuscator_v2.json
```

## 📊 Model Architectures

### HTML Parser
- **Architecture**: Transformer encoder (BERT-style)
- **Parameters**: ~5M
- **Tasks**: Validity classification, complexity estimation, semantic tagging
- **Use case**: HTML validation and structure understanding

### CSS Parser
- **Architecture**: Lightweight transformer
- **Parameters**: ~2M
- **Tasks**: Rule validation, selector analysis, optimization suggestions
- **Use case**: CSS optimization and validation

### JS Parser
- **Architecture**: Deep transformer encoder
- **Parameters**: ~15M
- **Tasks**: Syntax validation, obfuscation detection, framework recognition
- **Use case**: JavaScript analysis and security scanning

### Deobfuscator
- **Architecture**: Transformer encoder-decoder (T5-style)
- **Parameters**: ~25M
- **Features**: Copy mechanism, beam search, pointer network
- **Use case**: JavaScript deobfuscation and code transformation

## 🎓 Training Best Practices

### Learning Rate Selection
- **Small models** (<10M params): `lr=0.001`
- **Medium models** (10-50M): `lr=0.0005`
- **Large models** (>50M): `lr=0.0003`

### Batch Size
- **GPU Memory**: 8GB → batch_size=16, 16GB → batch_size=32, 24GB → batch_size=64
- **Use gradient accumulation** if memory limited

### Training Time
- **HTML/CSS Parser**: 2-4 hours on single GPU
- **JS Parser**: 6-8 hours
- **Deobfuscator**: 12-24 hours

### Checkpointing
- Models saved every epoch to `checkpoints/{task}/`
- Best model saved as `best_model.pt`
- Resume with `--resume checkpoints/{task}/checkpoint_epoch_10.pt`

## 🔧 Configuration

All models configured via YAML files in `configs/`:

```yaml
# Example: html_parser.yaml
model:
  d_model: 256
  num_heads: 8
  num_layers: 6
  
training:
  epochs: 20
  batch_size: 32
  
optimizer:
  type: adamw
  lr: 0.0005
  weight_decay: 0.01
  
scheduler:
  type: cosine
  min_lr: 1e-6
```

## 📈 Monitoring Training

Training metrics are logged to:
- **Console**: Real-time progress bars
- **Checkpoints**: Loss history saved with models
- **TensorBoard** (optional): `tensorboard --logdir runs/`

## 🔬 Advanced Features

### Multi-GPU Training
```bash
# Distributed training (coming soon)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_unified.py --task deobfuscator
```

### Custom Models
```python
from core.models import BaseModel, TransformerEncoder

class MyCustomModel(TransformerEncoder):
    def __init__(self, config):
        super().__init__(config)
        # Add custom layers
```

### Custom Tokenizers
```python
from core.data.tokenizers import BPETokenizer

tokenizer = BPETokenizer(vocab_size=15000)
tokenizer.learn_bpe(texts)
tokenizer.save("tokenizers/my_tokenizer.json")
```

## 📚 Migration from Old Scripts

Old scripts are **deprecated** but kept for reference:

| Old Script | New Command |
|------------|-------------|
| `train_html_parser.py` | `train_unified.py --task html_parser` |
| `train_css_parser.py` | `train_unified.py --task css_parser` |
| `train_js_parser.py` | `train_unified.py --task js_parser` |
| `train_enhanced_deobfuscator.py` | `train_unified.py --task deobfuscator` |
| `train_seq2seq_deobfuscator.py` | Same as above |
| `train_transformer_generator.py` | Replaced by unified transformer |

### Key Changes
- ✅ **Unified interface**: One script for all tasks
- ✅ **YAML configs**: Easier parameter management
- ✅ **Modern architectures**: Transformer-based
- ✅ **Better training**: AMP, gradient accumulation, warmup
- ✅ **Automatic ONNX export**: No manual conversion

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 8

# Use gradient accumulation
--gradient-accumulation 4

# Enable AMP
--use-amp
```

### Slow Training
```bash
# Enable AMP (2-3x speedup on GPU)
--use-amp

# Increase batch size
--batch-size 64

# Use multiple workers for data loading
# (set in config: num_workers: 4)
```

### Poor Performance
- **Check data quality**: Balanced classes, sufficient samples
- **Adjust learning rate**: Try 0.0001 - 0.001 range
- **Increase model size**: `--d-model 512 --num-layers 8`
- **Train longer**: `--epochs 50`

## 📖 References

### Model Architectures
- Vaswani et al., "Attention is All You Need" (Transformer)
- Devlin et al., "BERT" (Encoder architecture)
- Radford et al., "GPT-2/GPT-3" (Decoder architecture)
- Raffel et al., "T5" (Encoder-decoder)

### Training Techniques
- Loshchilov & Hutter, "Decoupled Weight Decay" (AdamW)
- Chen et al., "SimCLR" (Contrastive learning)
- Bengio et al., "Curriculum Learning"
- Finn et al., "MAML" (Meta-learning)

### Code Generation
- Lu et al., "CodeBERT"
- Feng et al., "CodeGen"
- Li et al., "GraphCodeBERT"

## 🤝 Contributing

To add new models:

1. Create model class in `core/models/`
2. Inherit from `BaseModel`
3. Add config file in `configs/`
4. Update `train_unified.py` task list

Example:
```python
# core/models/my_model.py
from .base import BaseModel

class MyModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Your architecture
    
    def forward(self, x):
        # Your forward pass
        return output
```

## 📄 License

Same as BrowerAI main project.

## 🔗 Links

- [Main Documentation](../docs/README.md)
- [ONNX Training Guide](../docs/ONNX_TRAINING_GUIDE.md)
- [Model Zoo](../models/MODEL_ZOO.md)
