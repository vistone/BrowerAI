#!/usr/bin/env python3
"""
Data preparation script for BrowerAI model training.

This script prepares training data for HTML, CSS, and JavaScript parsers by:
1. Generating synthetic training examples
2. Creating train/validation/test splits
3. Preprocessing and tokenizing data
4. Saving processed data for training

Usage:
    python prepare_data.py [--output-dir DATA_DIR] [--num-samples N]
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def generate_html_samples(num_samples: int = 1000) -> List[Dict[str, str]]:
    """Generate synthetic HTML training samples."""
    samples = []
    
    # HTML elements and attributes for generation
    elements = ['div', 'span', 'p', 'h1', 'h2', 'h3', 'ul', 'ol', 'li', 'a', 'img', 'table', 'tr', 'td']
    attributes = ['class', 'id', 'style', 'title', 'alt', 'href', 'src']
    
    for i in range(num_samples):
        # Generate random HTML structure
        depth = random.randint(1, 5)
        html_parts = ['<!DOCTYPE html>', '<html>', '<head>', '<title>Sample Page</title>', '</head>', '<body>']
        
        # Build nested structure
        for d in range(depth):
            elem = random.choice(elements)
            if random.random() > 0.5:
                attr = random.choice(attributes)
                html_parts.append(f'<{elem} {attr}="value{i}">')
            else:
                html_parts.append(f'<{elem}>')
            
            if elem in ['p', 'h1', 'h2', 'h3', 'span', 'li', 'td']:
                html_parts.append(f'Sample text content {i}')
            
            html_parts.append(f'</{elem}>')
        
        html_parts.extend(['</body>', '</html>'])
        
        samples.append({
            'input': '\n'.join(html_parts),
            'label': 'valid',
            'type': 'html'
        })
        
        # Also generate some malformed HTML for training
        if i % 3 == 0:
            malformed_parts = html_parts[:-2]  # Remove closing tags
            samples.append({
                'input': '\n'.join(malformed_parts),
                'label': 'malformed',
                'type': 'html'
            })
    
    return samples


def generate_css_samples(num_samples: int = 1000) -> List[Dict[str, str]]:
    """Generate synthetic CSS training samples."""
    samples = []
    
    selectors = ['body', 'div', '.container', '#header', 'p', 'h1', 'a', 'button', 'input', 'table']
    properties = [
        'color', 'background-color', 'font-size', 'margin', 'padding', 
        'border', 'width', 'height', 'display', 'position'
    ]
    values = {
        'color': ['red', '#333', 'rgb(255, 0, 0)', 'rgba(0, 0, 0, 0.5)'],
        'background-color': ['white', '#f0f0f0', 'transparent'],
        'font-size': ['12px', '14px', '16px', '1.5em', '100%'],
        'margin': ['0', '10px', '20px auto', '1em 2em'],
        'padding': ['0', '5px', '10px 20px'],
        'border': ['1px solid black', 'none', '2px dashed #ccc'],
        'width': ['100%', '50%', '300px', 'auto'],
        'height': ['100%', '200px', 'auto'],
        'display': ['block', 'inline', 'flex', 'grid', 'none'],
        'position': ['static', 'relative', 'absolute', 'fixed']
    }
    
    for i in range(num_samples):
        num_rules = random.randint(1, 5)
        css_parts = []
        
        for _ in range(num_rules):
            selector = random.choice(selectors)
            num_props = random.randint(2, 6)
            
            css_parts.append(f'{selector} {{')
            for _ in range(num_props):
                prop = random.choice(properties)
                value = random.choice(values[prop])
                css_parts.append(f'    {prop}: {value};')
            css_parts.append('}')
            css_parts.append('')
        
        samples.append({
            'input': '\n'.join(css_parts),
            'label': 'valid',
            'type': 'css'
        })
    
    return samples


def generate_js_samples(num_samples: int = 1000) -> List[Dict[str, str]]:
    """Generate synthetic JavaScript training samples."""
    samples = []
    
    function_names = ['calculate', 'process', 'validate', 'transform', 'fetch', 'update']
    var_names = ['data', 'result', 'value', 'input', 'output', 'config']
    
    for i in range(num_samples):
        func_name = random.choice(function_names)
        var1 = random.choice(var_names)
        var2 = random.choice(var_names)
        
        js_parts = [
            f'function {func_name}({var1}, {var2}) {{',
            f'    const result = {var1} + {var2};',
            f'    if (result > 0) {{',
            f'        console.log("Positive result:", result);',
            f'        return result;',
            f'    }}',
            f'    return 0;',
            f'}}',
            f'',
            f'const output = {func_name}(10, 20);',
            f'console.log("Output:", output);'
        ]
        
        samples.append({
            'input': '\n'.join(js_parts),
            'label': 'valid',
            'type': 'js'
        })
        
        # Generate some code with syntax errors for training
        if i % 4 == 0:
            malformed_parts = js_parts[:5]  # Incomplete function
            samples.append({
                'input': '\n'.join(malformed_parts),
                'label': 'malformed',
                'type': 'js'
            })
    
    return samples


def split_data(samples: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List, List, List]:
    """Split data into train, validation, and test sets."""
    random.shuffle(samples)
    
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = samples[:train_end]
    val_data = samples[train_end:val_end]
    test_data = samples[val_end:]
    
    return train_data, val_data, test_data


def save_dataset(data: List[Dict], output_path: Path):
    """Save dataset to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for BrowerAI models')
    parser.add_argument('--output-dir', type=str, default='../data', 
                       help='Output directory for processed data')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples to generate for each type')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("BrowerAI Data Preparation Pipeline")
    print("=" * 60)
    
    # Generate HTML samples
    print("\n[1/3] Generating HTML samples...")
    html_samples = generate_html_samples(args.num_samples)
    html_train, html_val, html_test = split_data(html_samples)
    
    save_dataset(html_train, output_dir / 'html' / 'train.json')
    save_dataset(html_val, output_dir / 'html' / 'val.json')
    save_dataset(html_test, output_dir / 'html' / 'test.json')
    
    # Generate CSS samples
    print("\n[2/3] Generating CSS samples...")
    css_samples = generate_css_samples(args.num_samples)
    css_train, css_val, css_test = split_data(css_samples)
    
    save_dataset(css_train, output_dir / 'css' / 'train.json')
    save_dataset(css_val, output_dir / 'css' / 'val.json')
    save_dataset(css_test, output_dir / 'css' / 'test.json')
    
    # Generate JS samples
    print("\n[3/3] Generating JavaScript samples...")
    js_samples = generate_js_samples(args.num_samples)
    js_train, js_val, js_test = split_data(js_samples)
    
    save_dataset(js_train, output_dir / 'js' / 'train.json')
    save_dataset(js_val, output_dir / 'js' / 'val.json')
    save_dataset(js_test, output_dir / 'js' / 'test.json')
    
    # Summary
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"\nHTML samples: {len(html_train)} train, {len(html_val)} val, {len(html_test)} test")
    print(f"CSS samples:  {len(css_train)} train, {len(css_val)} val, {len(css_test)} test")
    print(f"JS samples:   {len(js_train)} train, {len(js_val)} val, {len(js_test)} test")
    print(f"\nData saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Review the generated data")
    print("  2. Run training scripts: python scripts/train_html_parser.py")
    print("  3. Export models to ONNX format")


if __name__ == '__main__':
    main()
