# BrowerAI Training - Quick Reference

## Batch Data Collection

```bash
# Collect 50 sites
BATCH_SIZE=10 START=1 STOP=50 training/scripts/collect_sites.sh
```

## Training Pipeline

```bash
# 1. Extract features
python training/scripts/extract_features.py

# 2. Train classifier
python training/scripts/train_classifier.py

# 3. Generate themes
python training/scripts/theme_recommender.py

# 4. Analyze obfuscation
python training/scripts/analyze_obfuscation.py
```

## Directory Layout

- `config/` - Category definitions
- `data/` - Raw feedback JSON
- `features/` - Extracted feature vectors
- `models/` - Trained classifiers
- `scripts/` - Training tools

## Key Capabilities

1. **Site Classification**: Categorize by type (news/shop/tech/social)
2. **Tech Stack Detection**: Identify React/Vue/jQuery/webpack
3. **Obfuscation Detection**: Detect minification/obfuscation
4. **Theme Generation**: Generate 5 alternative color schemes
5. **Layout Analysis**: Detect grid/column/card patterns

All models optimized for browser-side inference.
