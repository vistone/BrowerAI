# Holistic Website Learning System

## Overview

The Holistic Website Learning System treats **websites as complete, integrated systems** rather than isolated components (HTML, CSS, JS). This approach mirrors how browsers actually load and render websites.

## Core Philosophy

> "学习应该是一个系统的过程,而不是分开,html一个库,js一个库,css一个库"
> 
> "Learning should be a systemic process, not separated into HTML library, JS library, CSS library"

When a browser loads a website:
1. HTML is parsed to build the DOM
2. CSS files are loaded **in order** to style elements
3. JS files are loaded with specific **timing** (blocking, defer, async)
4. Files have **dependencies** on each other
5. The website has an **intent** (shopping, news, education)
6. Code has **company-specific patterns** and fingerprints
7. Layout adapts to **different devices**

Our model learns **all of these aspects together**.

## Architecture

### Multi-Modal Learning Pipeline

```
Website Input
    ├── HTML Structure
    │   └── TransformerEncoder (4 layers)
    │
    ├── CSS Styling (multiple files, ordered)
    │   └── TransformerEncoder (3 layers)
    │
    └── JavaScript Behavior (multiple files, ordered)
        └── TransformerEncoder (6 layers)
                ↓
        Cross-Modal Attention
    (HTML-CSS, HTML-JS, CSS-JS fusion)
                ↓
        Global Website Encoder
                ↓
        512-dim Website Embedding
                ↓
    ┌───────────────┬───────────────┬───────────────┐
    │               │               │               │
Intent          Style        Dependency      Device
Classifier      Analyzer     Learner        Analyzer
    │               │               │               │
10 categories  Framework     GNN            Responsive
               Build Tool    Loading Order   Breakpoints
               Company       Dependencies    Strategy
```

### Key Components

#### 1. **HolisticWebsiteLearner** (Main Model)
- **Shared Embedding**: All code types use same vocabulary
- **Component Encoders**: Specialized layers for HTML/CSS/JS
- **Cross-Modal Fusion**: Attention between different code types
- **Global Understanding**: Unified website representation

#### 2. **WebsiteIntentClassifier**
Recognizes website purpose from 10 categories:
- Ecommerce (shopping sites)
- News (media sites)
- Education (learning platforms)
- Entertainment (streaming, games)
- Social (social networks)
- Business (corporate sites)
- Government (.gov sites)
- Personal (blogs, portfolios)
- Documentation (API docs, guides)
- Tools (web apps, utilities)

**How it works**: Analyzes URL structure, content patterns, and DOM structure

#### 3. **CodeStyleAnalyzer**
Extracts **fingerprints** and detects patterns:
- **Framework Detection**: 50+ frameworks (React, Vue, Angular, jQuery, etc.)
- **Build Tool Detection**: 20+ tools (Webpack, Vite, Rollup, etc.)
- **Company Style**: 30+ company-specific patterns (Google, Amazon, Facebook, etc.)
- **Fingerprint Extraction**: 256-dim vector capturing unique coding style

**Example patterns**:
```python
# React fingerprint
"_jsx(", "React.createElement", "useState"

# Shopify fingerprint
"cdn.shopify.com", "Shopify.theme", "shopify-analytics"

# Webpack fingerprint
"__webpack", "webpackJsonp"
```

#### 4. **DependencyGraphLearner** (GNN)
Models **file loading relationships**:
- **Graph Attention Network**: 3 layers
- **Node Features**: File embeddings
- **Edge Features**: Dependency types (import, require, link)
- **Tasks**:
  - Predict loading order
  - Classify dependency types
  - Identify critical rendering path

**Why this matters**: Loading order affects performance and correctness

#### 5. **DeviceAdaptationAnalyzer**
Understands **responsive design**:
- **Strategies**: Fluid, adaptive, responsive, mobile-first
- **Breakpoints**: Detect screen size thresholds
- **Interaction Models**: Touch, mouse, hybrid

## Training Data Format

### JSONL Structure
```json
{
  "url": "https://example.com",
  "category": "ecommerce",
  "html": "<html>...</html>",
  "css_files": [
    {
      "path": "styles/main.css",
      "content": "body { ... }",
      "order": 0
    },
    {
      "path": "styles/theme.css",
      "content": ".header { ... }",
      "order": 1
    }
  ],
  "js_files": [
    {
      "path": "js/vendor.js",
      "content": "...",
      "order": 0,
      "type": "blocking"
    },
    {
      "path": "js/app.js",
      "content": "import './vendor.js'; ...",
      "order": 1,
      "type": "defer"
    }
  ],
  "dependencies": [
    {
      "from": "app.js",
      "to": "vendor.js",
      "type": "import"
    },
    {
      "from": "theme.css",
      "to": "main.css",
      "type": "import"
    }
  ],
  "metadata": {
    "framework": "React",
    "build_tool": "Webpack",
    "company": "Amazon",
    "responsive": true
  }
}
```

### Key Properties
- **Preserves Loading Order**: Files have `order` field
- **Captures Dependencies**: Explicit graph structure
- **Rich Metadata**: Framework, build tool, company patterns
- **Multi-File Support**: Not limited to single HTML/CSS/JS

## Usage

### 1. Prepare Training Data
```bash
# Crawl example websites
python scripts/prepare_website_data.py \
    --output data/websites/websites_train.jsonl \
    --num-sites 100

# Or use custom URL list
python scripts/prepare_website_data.py \
    --urls-file my_urls.txt \
    --output data/websites/websites_train.jsonl
```

**URL file format** (`my_urls.txt`):
```
https://www.amazon.com,ecommerce
https://www.cnn.com,news
https://www.coursera.org,education
```

### 2. Train the Model
```bash
python scripts/train_holistic_website.py \
    --config configs/website_learner.yaml \
    --checkpoint-dir checkpoints/website_learner
```

### 3. Resume Training
```bash
python scripts/train_holistic_website.py \
    --config configs/website_learner.yaml \
    --resume checkpoints/website_learner/checkpoint_epoch_10.pt
```

### 4. Use in Python
```python
import torch
from core.models import HolisticWebsiteLearner
from core.data import WebsiteDataset, collate_website_batch

# Load model
model = HolisticWebsiteLearner(
    vocab_size=20000,
    d_model=512,
    num_heads=16,
    num_layers=8,
    num_categories=10
)
model.load_state_dict(torch.load('checkpoints/best_model.pt'))
model.eval()

# Load dataset
dataset = WebsiteDataset(
    data_dir='data/websites',
    data_file='websites_train.jsonl'
)

# Get a sample
sample = dataset[0]
html, css, js = sample['html'], sample['css'], sample['js']
adj_matrix = sample.get('adjacency_matrix')

# Forward pass
with torch.no_grad():
    outputs = model(
        html.unsqueeze(0),
        css.unsqueeze(0),
        js.unsqueeze(0),
        adj_matrix.unsqueeze(0) if adj_matrix is not None else None
    )

# Get predictions
category = outputs['category_logits'].argmax(dim=1)
framework = outputs['framework_logits'].argmax(dim=1)
website_embedding = outputs['website_embedding']

print(f"Category: {category.item()}")
print(f"Framework: {framework.item()}")
print(f"Embedding shape: {website_embedding.shape}")
```

## Multi-Task Learning

The model is trained on **7 tasks simultaneously**:

1. **Website Intent Classification**
   - Weight: 1.0
   - Metric: Accuracy
   - 10 categories

2. **Framework Detection**
   - Weight: 0.8
   - Metric: F1 Score
   - 50+ frameworks

3. **Build Tool Detection**
   - Weight: 0.5
   - Metric: Accuracy
   - 20+ tools

4. **Company Style Recognition**
   - Weight: 0.5
   - Metric: Accuracy
   - 30+ companies

5. **Dependency Prediction**
   - Weight: 0.7
   - Metric: Precision
   - Binary adjacency matrix

6. **Loading Order Ranking**
   - Weight: 0.6
   - Metric: Kendall's Tau
   - Relative ordering

7. **Device Adaptation Analysis**
   - Weight: 0.4
   - Metric: Accuracy
   - Responsive strategies

## Advanced Features

### Contrastive Learning
Learn that websites with same framework should have similar embeddings:
```python
# In config
contrastive_learning:
  enabled: true
  temperature: 0.07
  negative_samples: 16
```

### Curriculum Learning
Train on increasingly complex websites:
1. **Simple Sites** (epochs 1-10): Static HTML, single page
2. **Framework Sites** (epochs 11-25): React/Vue apps
3. **Complex Sites** (epochs 26-50): Full applications with many dependencies

### Data Augmentation
Simulate real-world variations:
- Minify CSS/JS (30% chance)
- Remove comments (40% chance)
- Reorder CSS rules (20% chance)
- Inject framework patterns (50% chance)

## Evaluation Metrics

### Classification Metrics
- **Category Accuracy**: How often intent is correctly classified
- **Framework F1**: Precision and recall for framework detection

### Ranking Metrics
- **Kendall's Tau**: Correlation for loading order prediction
- **Mean Reciprocal Rank**: Website similarity retrieval

### Clustering Metrics
- **Normalized Mutual Information**: How well websites cluster by category/framework

## Integration with Rust Browser

After training, export to ONNX:
```bash
# Automatically exported after training
# Or manually:
python -c "
from core.models import HolisticWebsiteLearner
model = HolisticWebsiteLearner(...)
model.export_to_onnx('models/local/website_learner_v1.onnx')
"
```

Then use in Rust (see main project README):
```rust
use browerai::{
    ai::ModelManager,
    parser::{HTMLParser, CSSParser, JSParser}
};

// Load website learner model
let manager = ModelManager::new("models/model_config.toml")?;
let website_model = manager.load_model("website_learner_v1")?;

// Parse website holistically
let html = HTMLParser::new().with_ai().parse(html_content)?;
let css = CSSParser::new().with_ai().parse(css_content)?;
let js = JSParser::new().with_ai().parse(js_content)?;

// Get website intent and style
let intent = website_model.predict_intent(&html, &css, &js)?;
let framework = website_model.detect_framework(&js)?;
```

## Comparison: Old vs New Approach

### ❌ Old Approach (Separate Learning)
```
HTML Parser → HTML Model → HTML AST
CSS Parser → CSS Model → CSS Rules
JS Parser → JS Model → JS AST

❌ No understanding of relationships
❌ No loading order awareness
❌ No intent recognition
❌ No style fingerprinting
```

### ✅ New Approach (Holistic Learning)
```
Complete Website → Multi-Modal Model → 
    - Unified Representation
    - Intent Classification
    - Framework Detection
    - Dependency Graph
    - Loading Order
    - Style Fingerprint
    - Device Adaptation

✅ Understands website as a system
✅ Learns company-specific patterns
✅ Models file dependencies
✅ Recognizes intent and purpose
```

## Performance Expectations

### Model Size
- **Parameters**: ~150M (512-dim, 8 layers)
- **ONNX File**: ~600MB
- **Inference Time**: ~50ms per website (CPU)

### Accuracy Targets
- **Intent Classification**: 85%+ accuracy
- **Framework Detection**: 90%+ F1 score
- **Loading Order**: 0.7+ Kendall's Tau

### Training Time
- **Hardware**: Single RTX 3090 (24GB)
- **Time**: ~2 days for 50 epochs on 10K websites
- **Batch Size**: 8 with gradient accumulation (effective 32)

## Future Enhancements

1. **Incremental Learning**: Update model with new websites online
2. **Template Detection**: Identify when websites use same template
3. **Performance Prediction**: Estimate page load time
4. **Security Analysis**: Detect malicious patterns
5. **Accessibility Scoring**: Evaluate WCAG compliance

## FAQ

**Q: Why not train separate models for HTML/CSS/JS?**
A: Websites are holistic systems. Files depend on each other, loading order matters, and the intent affects all components. Separate models miss these relationships.

**Q: How is this different from existing web parsers?**
A: Traditional parsers (like html5ever, cssparser) use hand-coded rules. Our approach **learns patterns** from real websites, including company-specific styles and framework fingerprints.

**Q: Can this replace traditional parsers?**
A: No, this **augments** them. The model provides high-level understanding (intent, style, dependencies) while traditional parsers handle low-level syntax.

**Q: What about privacy?**
A: Training data can be from public websites. For sensitive sites, use federated learning or train locally.

## References

- Original discussion: "学习应该是一个系统的过程"
- Multi-modal learning: CLIP, ALIGN papers
- Graph neural networks: Graph Attention Networks (GAT)
- Code understanding: CodeBERT, GraphCodeBERT

---

**Key Insight**: This system learns that **opening a website is a complete loading process** ("打开一个网站应该是全面的加载的过程"), not just parsing individual files. It understands intent, style, dependencies, and adaptation strategies together.
