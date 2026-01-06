# Holistic Website Learning - Quick Reference

## 🚀 Quick Start (3 Steps)

```bash
# 1. Install dependencies
cd /workspaces/BrowerAI/training
pip install torch pyyaml aiohttp beautifulsoup4 tqdm

# 2. Collect training data
python scripts/prepare_website_data.py \
    --output data/websites/train.jsonl \
    --num-sites 100

# 3. Train the model
python scripts/train_holistic_website.py \
    --config configs/website_learner.yaml \
    --checkpoint-dir checkpoints/website_learner
```

## 📋 Command Reference

### Data Preparation

```bash
# Use example URLs (default)
python scripts/prepare_website_data.py --output data/websites/train.jsonl --num-sites 20

# Use custom URL list
python scripts/prepare_website_data.py --urls-file my_urls.txt --output data/websites/train.jsonl
```

**URL file format** (`my_urls.txt`):
```
https://www.amazon.com,ecommerce
https://www.cnn.com,news
https://www.coursera.org,education
```

### Training

```bash
# Start fresh training
python scripts/train_holistic_website.py --config configs/website_learner.yaml

# Resume from checkpoint
python scripts/train_holistic_website.py \
    --config configs/website_learner.yaml \
    --resume checkpoints/website_learner/checkpoint_epoch_10.pt

# Custom checkpoint directory
python scripts/train_holistic_website.py \
    --config configs/website_learner.yaml \
    --checkpoint-dir my_checkpoints
```

### Testing

```bash
# Quick system test
python scripts/test_holistic_simple.py

# Full integration test (after implementing full test suite)
python scripts/test_holistic_system.py
```

## 📊 Configuration

**File**: `configs/website_learner.yaml`

### Key Parameters

```yaml
model:
  vocab_size: 20000    # Vocabulary size
  d_model: 512         # Model dimension
  num_heads: 16        # Attention heads
  num_layers: 8        # Transformer layers
  num_categories: 10   # Website categories

training:
  epochs: 50           # Training epochs
  batch_size: 8        # Batch size
  
  # Multi-task weights
  task_weights:
    category: 1.0      # Intent classification
    framework: 0.8     # Framework detection
    build_tool: 0.5    # Build tool detection
    company_style: 0.5 # Company patterns
    dependency: 0.7    # Dependency graph
    loading_order: 0.6 # Loading order
    adaptation: 0.4    # Device adaptation

optimizer:
  type: adamw
  lr: 0.0002           # Learning rate
  weight_decay: 0.01

data:
  data_dir: data/websites
  train_file: websites_train.jsonl
  val_file: websites_val.jsonl
  max_html_len: 2048
  max_css_len: 1024
  max_js_len: 2048
```

## 🎯 Website Categories (10)

1. **ecommerce** - Shopping sites (Amazon, eBay, Shopify)
2. **news** - News/media (CNN, BBC, Reuters)
3. **education** - Learning platforms (Coursera, Khan Academy)
4. **entertainment** - Streaming, games (Netflix, YouTube, Twitch)
5. **social** - Social networks (Twitter, LinkedIn, Reddit)
6. **business** - Corporate sites, SaaS
7. **government** - Government sites (.gov)
8. **personal** - Blogs, portfolios
9. **documentation** - API docs (MDN, docs sites)
10. **tools** - Web apps, utilities (GitHub, Stack Overflow)

## 🔧 Detected Patterns

### Frameworks (50+)
- React, Vue, Angular, Svelte
- Next.js, Nuxt, Gatsby
- jQuery, Bootstrap, Tailwind
- Material-UI, Ant Design
- And many more...

### Build Tools (20+)
- Webpack, Vite, Rollup
- Parcel, esbuild, Browserify
- Gulp, Grunt

### Company Styles (30+)
- Google, Amazon, Facebook
- Shopify, Stripe
- And more...

## 📁 Data Format

**JSONL** - One website per line:

```json
{
  "url": "https://example.com",
  "category": "ecommerce",
  "html": "<html>...</html>",
  "css_files": [
    {"path": "main.css", "content": "...", "order": 0}
  ],
  "js_files": [
    {"path": "app.js", "content": "...", "order": 0, "type": "defer"}
  ],
  "dependencies": [
    {"from": "app.js", "to": "vendor.js", "type": "import"}
  ],
  "metadata": {
    "framework": "React",
    "build_tool": "Webpack",
    "company": "Amazon",
    "responsive": true
  }
}
```

## 💻 Python API

### Basic Usage

```python
import torch
from core.models import HolisticWebsiteLearner
from core.data import WebsiteDataset, collate_website_batch
from torch.utils.data import DataLoader

# Load configuration
config = {
    "vocab_size": 20000,
    "d_model": 512,
    "num_heads": 16,
    "num_layers": 8,
    "d_ff": 2048,
    "dropout": 0.1,
    "max_len": 2048
}

# Initialize model
model = HolisticWebsiteLearner(config)

# Load dataset
dataset = WebsiteDataset(
    data_dir='data/websites',
    data_file='train.jsonl'
)

# Create data loader
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_website_batch
)

# Training loop
model.train()
for batch in loader:
    outputs = model(
        batch['html'],
        batch['css'],
        batch['js'],
        batch.get('adjacency_matrix'),
        batch.get('url_features')
    )
    
    # outputs contains:
    # - website_embedding: 512-dim representation
    # - category_logits: Intent prediction
    # - framework_logits: Framework detection
    # - build_tool_logits: Build tool detection
    # - company_logits: Company style
    # - style_fingerprint: 256-dim fingerprint
```

### Inference

```python
# Load trained model
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Get sample
sample = dataset[0]

# Predict
with torch.no_grad():
    outputs = model(
        sample['html'].unsqueeze(0),
        sample['css'].unsqueeze(0),
        sample['js'].unsqueeze(0),
        sample.get('adjacency_matrix').unsqueeze(0) if 'adjacency_matrix' in sample else None,
        sample.get('url_features').unsqueeze(0) if 'url_features' in sample else None
    )

# Extract predictions
category_id = outputs['category_logits'].argmax().item()
category = WebsiteIntentClassifier.CATEGORIES[category_id]

framework_id = outputs['framework_logits'].argmax().item()
fingerprint = outputs['style_fingerprint'][0]

print(f"Category: {category}")
print(f"Framework ID: {framework_id}")
print(f"Fingerprint shape: {fingerprint.shape}")
```

### Website Similarity

```python
# Compare two websites
embedding1 = outputs1['website_embedding']
embedding2 = outputs2['website_embedding']

similarity = model.compute_website_similarity(embedding1, embedding2)
print(f"Similarity: {similarity.item():.4f}")
```

## 🔍 Model Components

| Component | Purpose | Output |
|-----------|---------|--------|
| `shared_embedding` | Unified vocabulary | Embeddings |
| `html_encoder` | HTML structure | HTML repr |
| `css_encoder` | CSS styling | CSS repr |
| `js_encoder` | JS behavior | JS repr |
| `cross_modal_attention` | Learn relationships | Fused repr |
| `intent_classifier` | Website category | 10 logits |
| `style_analyzer` | Fingerprints | 256-dim + logits |
| `dependency_learner` | Loading order | Graph embeddings |
| `device_analyzer` | Responsive design | Strategy logits |

## 📈 Performance Metrics

### Training (Expected)
- **Time**: ~2 days (GPU) / ~1 week (CPU)
- **Memory**: ~12GB GPU RAM
- **Batch Size**: 8 (effective 32 with gradient accumulation)

### Inference
- **Speed**: ~50ms per website (CPU)
- **Model Size**: 205 MB
- **Hardware**: CPU-only, no GPU needed

### Accuracy (Target)
- **Intent**: 85%+ accuracy
- **Framework**: 90%+ F1
- **Loading Order**: 0.7+ Kendall's Tau

## 🐛 Troubleshooting

### Out of Memory
```yaml
# Reduce batch size in config
training:
  batch_size: 4  # Lower from 8
  
# Or use gradient accumulation
advanced:
  gradient_accumulation_steps: 8  # Higher
```

### Training Too Slow
```yaml
# Reduce model size
model:
  d_model: 256      # Lower from 512
  num_layers: 4     # Lower from 8
  
# Use mixed precision
advanced:
  use_amp: true
```

### Low Accuracy
```yaml
# Adjust task weights
training:
  task_weights:
    category: 2.0   # Increase important tasks
    
# More training epochs
training:
  epochs: 100       # Increase from 50
  
# Better optimizer
optimizer:
  lr: 0.0001        # Lower learning rate
```

## 📚 Documentation

- **Full Guide**: [HOLISTIC_LEARNING_GUIDE.md](HOLISTIC_LEARNING_GUIDE.md)
- **Implementation**: [HOLISTIC_IMPLEMENTATION_SUMMARY.md](HOLISTIC_IMPLEMENTATION_SUMMARY.md)
- **General Training**: [README.md](README.md)

## 🔗 Related Files

```
training/
├── configs/website_learner.yaml       # Configuration
├── core/
│   ├── models/website_learner.py     # Model implementation
│   └── data/website_dataset.py       # Dataset loader
├── scripts/
│   ├── prepare_website_data.py       # Data collection
│   ├── train_holistic_website.py     # Training
│   └── test_holistic_simple.py       # Testing
└── docs/
    ├── HOLISTIC_LEARNING_GUIDE.md    # Full guide
    ├── HOLISTIC_IMPLEMENTATION_SUMMARY.md  # Summary
    └── HOLISTIC_QUICKREF.md          # This file
```

## 💡 Tips

1. **Start Small**: Train on 100 websites first to verify setup
2. **Validate Early**: Check metrics after first epoch
3. **Balance Tasks**: Adjust weights based on validation performance
4. **Monitor GPU**: Use `nvidia-smi` to watch memory usage
5. **Save Often**: Checkpoints saved every epoch automatically
6. **ONNX Export**: Happens automatically after training completes

## ⚡ One-Liners

```bash
# Full pipeline (assuming data exists)
python scripts/train_holistic_website.py --config configs/website_learner.yaml && echo "Training complete!"

# Quick test
python scripts/test_holistic_simple.py && echo "System OK!"

# Collect data + train (sequential)
python scripts/prepare_website_data.py --num-sites 50 --output data/websites/train.jsonl && \
python scripts/train_holistic_website.py --config configs/website_learner.yaml

# Check model size
du -sh checkpoints/website_learner/best_model.pt

# Count training samples
wc -l data/websites/train.jsonl
```

---

**Quick Start**: 3 commands, 30 minutes to first model! 🚀
