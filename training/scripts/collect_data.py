#!/usr/bin/env python3
"""
Enhanced data collection script for BrowerAI Phase 2.

This script extends the data preparation pipeline with:
1. Real-world sample collection from popular websites
2. Data validation and quality checks
3. Advanced data augmentation techniques
4. Diverse dataset creation with multiple complexity levels
5. Statistical analysis of collected data

Usage:
    python collect_data.py [--output-dir DATA_DIR] [--sources SOURCE_LIST]
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    print("Warning: requests/beautifulsoup4 not available. Real-world collection disabled.")


class DataValidator:
    """Validates and checks quality of collected data."""
    
    def __init__(self):
        self.stats = {
            'html': {'valid': 0, 'malformed': 0, 'total': 0},
            'css': {'valid': 0, 'malformed': 0, 'total': 0},
            'js': {'valid': 0, 'malformed': 0, 'total': 0}
        }
    
    def validate_html(self, html: str) -> Tuple[bool, str]:
        """Validate HTML structure and return status with reason."""
        if not html or len(html.strip()) == 0:
            return False, "Empty HTML"
        
        # Check for basic HTML structure
        has_tags = bool(re.search(r'<[^>]+>', html))
        if not has_tags:
            return False, "No HTML tags found"
        
        # Check for balanced tags (basic check)
        open_tags = re.findall(r'<(\w+)[^>]*>', html)
        close_tags = re.findall(r'</(\w+)>', html)
        
        # Some tags are self-closing
        self_closing = {'img', 'br', 'hr', 'input', 'meta', 'link'}
        open_tags = [tag for tag in open_tags if tag not in self_closing]
        
        if len(open_tags) != len(close_tags):
            return False, "Unbalanced tags"
        
        self.stats['html']['valid'] += 1
        self.stats['html']['total'] += 1
        return True, "Valid HTML"
    
    def validate_css(self, css: str) -> Tuple[bool, str]:
        """Validate CSS structure."""
        if not css or len(css.strip()) == 0:
            return False, "Empty CSS"
        
        # Check for CSS rule structure
        has_rules = bool(re.search(r'[^}]+\{[^}]+\}', css))
        if not has_rules:
            return False, "No CSS rules found"
        
        # Check for balanced braces
        open_braces = css.count('{')
        close_braces = css.count('}')
        
        if open_braces != close_braces:
            return False, "Unbalanced braces"
        
        self.stats['css']['valid'] += 1
        self.stats['css']['total'] += 1
        return True, "Valid CSS"
    
    def validate_js(self, js: str) -> Tuple[bool, str]:
        """Validate JavaScript structure."""
        if not js or len(js.strip()) == 0:
            return False, "Empty JavaScript"
        
        # Check for balanced braces and parentheses
        braces = js.count('{') - js.count('}')
        parens = js.count('(') - js.count(')')
        brackets = js.count('[') - js.count(']')
        
        if braces != 0:
            return False, "Unbalanced braces"
        if parens != 0:
            return False, "Unbalanced parentheses"
        if brackets != 0:
            return False, "Unbalanced brackets"
        
        self.stats['js']['valid'] += 1
        self.stats['js']['total'] += 1
        return True, "Valid JavaScript"
    
    def print_stats(self):
        """Print validation statistics."""
        print("\n" + "=" * 60)
        print("Data Validation Statistics")
        print("=" * 60)
        for data_type, stats in self.stats.items():
            total = stats['total']
            if total > 0:
                valid_pct = (stats['valid'] / total) * 100
                print(f"{data_type.upper()}: {stats['valid']}/{total} valid ({valid_pct:.1f}%)")


class DataAugmenter:
    """Augments training data with various techniques."""
    
    @staticmethod
    def augment_html(html: str, num_variations: int = 3) -> List[str]:
        """Create augmented variations of HTML."""
        variations = [html]
        
        # Variation 1: Add extra whitespace
        variation1 = re.sub(r'>', '>\n  ', html)
        variations.append(variation1)
        
        # Variation 2: Remove unnecessary whitespace
        variation2 = re.sub(r'\s+', ' ', html)
        variations.append(variation2)
        
        # Variation 3: Change attribute order (if present)
        def swap_attrs(match):
            tag = match.group(0)
            # Simple attribute swap simulation
            return tag
        variation3 = re.sub(r'<\w+[^>]+>', swap_attrs, html)
        variations.append(variation3)
        
        return variations[:num_variations]
    
    @staticmethod
    def augment_css(css: str, num_variations: int = 3) -> List[str]:
        """Create augmented variations of CSS."""
        variations = [css]
        
        # Variation 1: Minified version
        variation1 = re.sub(r'\s+', ' ', css)
        variation1 = re.sub(r'\s*([{}:;,])\s*', r'\1', variation1)
        variations.append(variation1)
        
        # Variation 2: Formatted version
        variation2 = css.replace('{', ' {\n  ').replace(';', ';\n  ').replace('}', '\n}')
        variations.append(variation2)
        
        # Variation 3: Add comments (simulated)
        variation3 = f"/* Generated styles */\n{css}"
        variations.append(variation3)
        
        return variations[:num_variations]
    
    @staticmethod
    def augment_js(js: str, num_variations: int = 3) -> List[str]:
        """Create augmented variations of JavaScript."""
        variations = [js]
        
        # Variation 1: Minified
        variation1 = re.sub(r'\s+', ' ', js)
        variations.append(variation1)
        
        # Variation 2: Formatted
        variation2 = js.replace('{', '{\n  ').replace(';', ';\n')
        variations.append(variation2)
        
        # Variation 3: With comments
        variation3 = f"// Auto-generated code\n{js}"
        variations.append(variation3)
        
        return variations[:num_variations]


class EnhancedDataCollector:
    """Enhanced data collection with multiple sources."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.validator = DataValidator()
        self.augmenter = DataAugmenter()
    
    def collect_real_world_samples(self, urls: List[str]) -> Dict[str, List[str]]:
        """Collect real-world HTML/CSS/JS samples from URLs."""
        if not SCRAPING_AVAILABLE:
            print("Skipping real-world collection: dependencies not available")
            return {'html': [], 'css': [], 'js': []}
        
        samples = {'html': [], 'css': [], 'js': []}
        
        print("\nCollecting real-world samples...")
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Collect HTML
                    samples['html'].append(response.text[:10000])  # Limit size
                    
                    # Parse and extract inline CSS/JS
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract CSS
                    for style in soup.find_all('style'):
                        if style.string:
                            samples['css'].append(style.string[:5000])
                    
                    # Extract JS
                    for script in soup.find_all('script'):
                        if script.string:
                            samples['js'].append(script.string[:5000])
            except Exception as e:
                print(f"Error collecting from {url}: {e}")
        
        return samples
    
    def generate_complex_html(self, num_samples: int = 500) -> List[Dict]:
        """Generate complex HTML samples with various patterns."""
        samples = []
        
        elements = ['div', 'span', 'p', 'h1', 'h2', 'h3', 'ul', 'ol', 'li', 
                   'a', 'img', 'table', 'tr', 'td', 'form', 'input', 'button',
                   'nav', 'header', 'footer', 'section', 'article', 'aside']
        
        attributes = ['class', 'id', 'style', 'data-value', 'aria-label', 
                     'role', 'title', 'alt', 'href', 'src', 'type']
        
        for i in range(num_samples):
            depth = random.randint(2, 8)
            html_parts = ['<!DOCTYPE html>', '<html lang="en">', '<head>',
                         '<meta charset="UTF-8">', 
                         f'<title>Document {i}</title>', '</head>', '<body>']
            
            # Create nested structure with more complexity
            current_depth = 0
            open_tags = []
            
            for _ in range(depth * 3):
                if current_depth < depth and random.random() > 0.3:
                    # Open new tag
                    elem = random.choice(elements)
                    attrs = []
                    if random.random() > 0.5:
                        num_attrs = random.randint(1, 3)
                        for _ in range(num_attrs):
                            attr = random.choice(attributes)
                            attrs.append(f'{attr}="value-{i}-{random.randint(1, 100)}"')
                    
                    if attrs:
                        html_parts.append(f'<{elem} {" ".join(attrs)}>')
                    else:
                        html_parts.append(f'<{elem}>')
                    
                    if elem not in ['img', 'input', 'br', 'hr']:
                        open_tags.append(elem)
                        current_depth += 1
                    
                    # Add content for text elements
                    if elem in ['p', 'h1', 'h2', 'h3', 'span', 'li', 'td', 'button', 'a']:
                        html_parts.append(f'Content {i}-{random.randint(1, 1000)}')
                
                elif open_tags and random.random() > 0.2:
                    # Close a tag
                    elem = open_tags.pop()
                    html_parts.append(f'</{elem}>')
                    current_depth -= 1
            
            # Close remaining tags
            while open_tags:
                elem = open_tags.pop()
                html_parts.append(f'</{elem}>')
            
            html_parts.extend(['</body>', '</html>'])
            html = '\n'.join(html_parts)
            
            is_valid, reason = self.validator.validate_html(html)
            samples.append({
                'input': html,
                'label': 'valid' if is_valid else 'malformed',
                'type': 'html',
                'complexity': 'high' if depth > 5 else 'medium'
            })
        
        return samples
    
    def generate_complex_css(self, num_samples: int = 500) -> List[Dict]:
        """Generate complex CSS samples."""
        samples = []
        
        selectors = ['body', 'div', '.container', '#header', 'p', 'h1', 'a:hover',
                    'button:active', '.card > .title', 'nav ul li', '.flex-item',
                    '@media screen', '.grid-container', '[data-theme="dark"]']
        
        properties = [
            ('color', ['#333', 'rgb(255, 0, 0)', 'var(--primary)', 'inherit']),
            ('background', ['linear-gradient(90deg, #fff, #000)', '#f0f0f0', 'transparent']),
            ('font-size', ['16px', '1.5rem', 'clamp(12px, 2vw, 24px)']),
            ('margin', ['0 auto', '20px', '1em 2em', '10px 20px 30px 40px']),
            ('padding', ['1rem', '20px 40px', '0']),
            ('display', ['flex', 'grid', 'block', 'inline-block', 'none']),
            ('position', ['relative', 'absolute', 'fixed', 'sticky']),
            ('width', ['100%', '50vw', 'calc(100% - 40px)', 'max-content']),
            ('border', ['1px solid #ccc', '2px dashed red', 'none']),
            ('box-shadow', ['0 2px 4px rgba(0,0,0,0.1)', 'inset 0 0 10px #000']),
        ]
        
        for i in range(num_samples):
            num_rules = random.randint(3, 10)
            css_parts = []
            
            for _ in range(num_rules):
                selector = random.choice(selectors)
                num_props = random.randint(3, 8)
                
                css_parts.append(f'{selector} {{')
                for _ in range(num_props):
                    prop, values = random.choice(properties)
                    value = random.choice(values)
                    css_parts.append(f'  {prop}: {value};')
                css_parts.append('}')
                css_parts.append('')
            
            css = '\n'.join(css_parts)
            is_valid, reason = self.validator.validate_css(css)
            
            samples.append({
                'input': css,
                'label': 'valid' if is_valid else 'malformed',
                'type': 'css',
                'complexity': 'high' if num_rules > 6 else 'medium'
            })
        
        return samples
    
    def generate_complex_js(self, num_samples: int = 500) -> List[Dict]:
        """Generate complex JavaScript samples."""
        samples = []
        
        patterns = [
            # Function declarations
            "function {name}({params}) {{\n  const result = {expr};\n  return result;\n}}",
            # Arrow functions
            "const {name} = ({params}) => {{\n  return {expr};\n}};",
            # Class definitions
            "class {name} {{\n  constructor({params}) {{\n    this.value = {expr};\n  }}\n}}",
            # Async functions
            "async function {name}({params}) {{\n  const data = await fetch('{url}');\n  return data.json();\n}}",
            # Object methods
            "const obj = {{\n  {name}({params}) {{\n    return {expr};\n  }}\n}};",
        ]
        
        names = ['calculate', 'process', 'transform', 'validate', 'fetchData', 
                'updateState', 'handleClick', 'initialize', 'render', 'parse']
        
        for i in range(num_samples):
            num_functions = random.randint(2, 5)
            js_parts = []
            
            for _ in range(num_functions):
                pattern = random.choice(patterns)
                name = random.choice(names)
                params = ', '.join([f'param{j}' for j in range(random.randint(0, 3))])
                expr = f'{random.randint(1, 100)} + {random.randint(1, 100)}'
                url = 'https://api.example.com/data'
                
                code = pattern.format(name=name, params=params, expr=expr, url=url)
                js_parts.append(code)
                js_parts.append('')
            
            # Add usage
            js_parts.append(f'// Usage example')
            js_parts.append(f'const result = {random.choice(names)}();')
            js_parts.append(f'console.log(result);')
            
            js = '\n'.join(js_parts)
            is_valid, reason = self.validator.validate_js(js)
            
            samples.append({
                'input': js,
                'label': 'valid' if is_valid else 'malformed',
                'type': 'js',
                'complexity': 'high' if num_functions > 3 else 'medium'
            })
        
        return samples
    
    def save_dataset(self, samples: List[Dict], output_path: Path, augment: bool = True):
        """Save dataset with optional augmentation."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if augment:
            augmented_samples = []
            for sample in samples:
                augmented_samples.append(sample)
                
                # Create augmented variations
                if sample['type'] == 'html':
                    variations = self.augmenter.augment_html(sample['input'], 2)
                elif sample['type'] == 'css':
                    variations = self.augmenter.augment_css(sample['input'], 2)
                else:  # js
                    variations = self.augmenter.augment_js(sample['input'], 2)
                
                for var in variations[1:]:  # Skip first (original)
                    augmented_samples.append({
                        **sample,
                        'input': var,
                        'augmented': True
                    })
            
            samples = augmented_samples
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced data collection for BrowerAI Phase 2')
    parser.add_argument('--output-dir', type=str, default='../data',
                       help='Output directory for collected data')
    parser.add_argument('--num-samples', type=int, default=500,
                       help='Number of samples to generate for each type')
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Apply data augmentation')
    parser.add_argument('--complexity', choices=['low', 'medium', 'high'], default='high',
                       help='Complexity level of generated samples')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    collector = EnhancedDataCollector(output_dir)
    
    print("=" * 60)
    print("BrowerAI Phase 2.1: Enhanced Data Collection")
    print("=" * 60)
    
    # Generate enhanced samples
    print("\n[1/3] Generating complex HTML samples...")
    html_samples = collector.generate_complex_html(args.num_samples)
    
    print("\n[2/3] Generating complex CSS samples...")
    css_samples = collector.generate_complex_css(args.num_samples)
    
    print("\n[3/3] Generating complex JavaScript samples...")
    js_samples = collector.generate_complex_js(args.num_samples)
    
    # Split data
    def split_data(samples, train=0.7, val=0.15, test=0.15):
        random.shuffle(samples)
        n = len(samples)
        train_end = int(n * train)
        val_end = int(n * (train + val))
        return samples[:train_end], samples[train_end:val_end], samples[val_end:]
    
    # Save datasets
    print("\n" + "=" * 60)
    print("Saving datasets...")
    print("=" * 60)
    
    for data_type, samples in [('html', html_samples), ('css', css_samples), ('js', js_samples)]:
        train, val, test = split_data(samples)
        
        collector.save_dataset(train, output_dir / data_type / 'train_enhanced.json', args.augment)
        collector.save_dataset(val, output_dir / data_type / 'val_enhanced.json', False)
        collector.save_dataset(test, output_dir / data_type / 'test_enhanced.json', False)
    
    # Print statistics
    collector.validator.print_stats()
    
    print("\n" + "=" * 60)
    print("Data Collection Complete!")
    print("=" * 60)
    print(f"\nEnhanced datasets saved to: {output_dir.absolute()}")
    print(f"Augmentation: {'Enabled' if args.augment else 'Disabled'}")
    print(f"Complexity level: {args.complexity}")
    print("\nNext steps:")
    print("  1. Review the generated data quality")
    print("  2. Train models with enhanced datasets")
    print("  3. Compare performance with baseline models")


if __name__ == '__main__':
    main()
