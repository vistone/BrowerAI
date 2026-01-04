# BrowerAI Training Data Repository

This repository contains organized training datasets for BrowerAI's AI models.

## Directory Structure

```
data/
├── html/                    # HTML training datasets
│   ├── raw/                # Raw HTML samples
│   ├── processed/          # Preprocessed HTML data
│   └── synthetic/          # Synthetically generated HTML
├── css/                     # CSS training datasets
│   ├── raw/                # Raw CSS samples
│   ├── processed/          # Preprocessed CSS data
│   └── synthetic/          # Synthetically generated CSS
├── js/                      # JavaScript training datasets
│   ├── raw/                # Raw JS samples
│   ├── processed/          # Preprocessed JS data
│   └── synthetic/          # Synthetically generated JS
├── combined/                # Combined multi-format datasets
├── benchmarks/              # Benchmark datasets
└── metadata/                # Dataset metadata and manifests
```

## Dataset Categories

### HTML Datasets

**Structure Prediction Dataset**
- **Purpose**: Train models to predict HTML structure
- **Size**: ~10K samples
- **Format**: JSON with HTML content and structure labels
- **Location**: `html/processed/structure_prediction/`

**Malformed HTML Fixer Dataset**
- **Purpose**: Train models to fix malformed HTML
- **Size**: ~5K samples
- **Format**: Pairs of malformed and corrected HTML
- **Location**: `html/processed/malformed_fixer/`

**Synthetic HTML Dataset**
- **Purpose**: Augment training with synthetic HTML
- **Size**: ~20K samples
- **Format**: Generated HTML with various complexities
- **Location**: `html/synthetic/`

### CSS Datasets

**Rule Deduplication Dataset**
- **Purpose**: Train models to identify duplicate CSS rules
- **Size**: ~5K samples
- **Format**: CSS with duplication labels
- **Location**: `css/processed/deduplication/`

**Selector Optimization Dataset**
- **Purpose**: Train models to optimize CSS selectors
- **Size**: ~8K samples
- **Format**: Selector pairs (original, optimized)
- **Location**: `css/processed/selector_optimization/`

**Minification Dataset**
- **Purpose**: Train models for safe CSS minification
- **Size**: ~6K samples
- **Format**: CSS pairs (original, minified)
- **Location**: `css/processed/minification/`

**Synthetic CSS Dataset**
- **Purpose**: Augment training with synthetic CSS
- **Size**: ~15K samples
- **Format**: Generated CSS with various patterns
- **Location**: `css/synthetic/`

### JavaScript Datasets

**Tokenization Enhancement Dataset**
- **Purpose**: Train models to enhance JS tokenization
- **Size**: ~7K samples
- **Format**: JS code with enhanced token labels
- **Location**: `js/processed/tokenization/`

**AST Prediction Dataset**
- **Purpose**: Train models to predict AST structure
- **Size**: ~10K samples
- **Format**: JS code with AST structure labels
- **Location**: `js/processed/ast_prediction/`

**Optimization Suggestions Dataset**
- **Purpose**: Train models to suggest code optimizations
- **Size**: ~8K samples
- **Format**: JS code pairs (original, optimized) with explanations
- **Location**: `js/processed/optimization/`

**Synthetic JavaScript Dataset**
- **Purpose**: Augment training with synthetic JS
- **Size**: ~18K samples
- **Format**: Generated JS with various patterns
- **Location**: `js/synthetic/`

### Combined Datasets

**Full Page Dataset**
- **Purpose**: Train models on complete HTML+CSS+JS pages
- **Size**: ~3K samples
- **Format**: Complete web pages from real websites
- **Location**: `combined/full_pages/`

**Component Dataset**
- **Purpose**: Train on reusable web components
- **Size**: ~5K samples
- **Format**: HTML/CSS/JS component bundles
- **Location**: `combined/components/`

### Benchmark Datasets

**Real-World Website Dataset**
- **Purpose**: Test models on production websites
- **Size**: 100+ websites
- **Format**: Crawled HTML/CSS/JS from popular sites
- **Location**: `benchmarks/real_world/`

**Cross-Browser Compatibility Dataset**
- **Purpose**: Test rendering across browsers
- **Size**: ~500 test cases
- **Format**: HTML/CSS with expected rendering
- **Location**: `benchmarks/cross_browser/`

**Performance Benchmark Dataset**
- **Purpose**: Measure model performance
- **Size**: Various complexity levels
- **Format**: Timed test cases
- **Location**: `benchmarks/performance/`

## Dataset Metadata

Each dataset includes a `manifest.json` file with:

```json
{
  "name": "dataset_name",
  "version": "1.0.0",
  "created": "2026-01-04T00:00:00Z",
  "updated": "2026-01-04T00:00:00Z",
  "description": "Dataset description",
  "size": {
    "samples": 10000,
    "bytes": 50000000
  },
  "format": "json|csv|parquet",
  "schema": {
    "input": "HTML/CSS/JS string",
    "output": "Labels or predictions"
  },
  "splits": {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1
  },
  "license": "Apache-2.0",
  "source": "synthetic|crawled|manual",
  "tags": ["html", "parsing", "structure"],
  "checksum": "sha256:..."
}
```

## Dataset Versioning

Datasets follow semantic versioning:
- **Major**: Breaking changes to format or schema
- **Minor**: New samples added, backward compatible
- **Patch**: Bug fixes, quality improvements

Version history is tracked in `metadata/versions.json`

## Using Datasets

### Download Datasets

```bash
# Download specific dataset
python scripts/dataset_manager.py download --dataset html/structure_prediction

# Download all datasets for a category
python scripts/dataset_manager.py download --category html

# Download all datasets
python scripts/dataset_manager.py download --all
```

### Load Datasets in Python

```python
from training.data_repository import DatasetManager

# Initialize manager
manager = DatasetManager()

# Load dataset
dataset = manager.load("html/structure_prediction", split="train")

# Iterate over samples
for sample in dataset:
    html_input = sample["input"]
    structure_label = sample["output"]
    # ... training code
```

### Upload New Datasets

```bash
# Upload dataset with metadata
python scripts/dataset_manager.py upload \
  --path /path/to/dataset \
  --name html/custom_dataset \
  --description "Custom HTML dataset" \
  --license Apache-2.0
```

## Dataset Quality Standards

All datasets must meet these standards:

### Data Quality
- **Completeness**: No missing or null values (unless intentional)
- **Accuracy**: Manual verification on sample subset
- **Consistency**: Standardized format across samples
- **Diversity**: Representative of real-world usage

### Documentation
- **README**: Description, usage, examples
- **Manifest**: Complete metadata in manifest.json
- **Schema**: Clear input/output schema definition
- **Examples**: Sample data for quick inspection

### Licensing
- **Open Source**: Prefer Apache-2.0 or MIT licenses
- **Attribution**: Credit original data sources
- **Compliance**: Ensure no copyright violations
- **Privacy**: No personal or sensitive data

## Contributing Datasets

To contribute a new dataset:

1. **Prepare Data**
   - Clean and preprocess data
   - Split into train/val/test
   - Create manifest.json

2. **Document**
   - Write README with description
   - Add usage examples
   - Document any preprocessing

3. **Validate**
   - Run quality checks
   - Verify format compliance
   - Test loading and iteration

4. **Submit**
   - Create pull request
   - Include dataset statistics
   - Provide sample outputs

## Dataset Statistics

| Category | Datasets | Total Samples | Total Size |
|----------|----------|---------------|------------|
| HTML | 3 | ~35K | ~150 MB |
| CSS | 4 | ~34K | ~120 MB |
| JavaScript | 4 | ~43K | ~180 MB |
| Combined | 2 | ~8K | ~80 MB |
| Benchmarks | 3 | ~600+ | ~200 MB |
| **Total** | **16** | **~120K** | **~730 MB** |

## Maintenance

### Update Schedule
- **Weekly**: Add new samples from web crawls
- **Monthly**: Quality review and cleanup
- **Quarterly**: Major version updates

### Quality Monitoring
- Automated tests for data integrity
- Regular manual audits
- Community feedback integration

### Backup Strategy
- Daily snapshots of all datasets
- Cloud storage backup
- Version history preservation

## Support

For questions or issues:
- GitHub Issues: https://github.com/vistone/BrowerAI/issues
- Discussions: https://github.com/vistone/BrowerAI/discussions

## License

All training datasets are provided under the Apache-2.0 license unless otherwise specified in the dataset manifest.
