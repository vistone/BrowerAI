#!/usr/bin/env python3
"""
Layout and theme recommender
Analyzes site structure and generates alternative visual themes
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib


class LayoutAnalyzer:
    """Analyze page layout structure"""
    
    def __init__(self):
        self.layout_patterns = {
            'single_column': re.compile(r'max-width:\s*\d{3,4}px', re.IGNORECASE),
            'two_column': re.compile(r'(grid-template-columns|columns):\s*\d+\s+\d+', re.IGNORECASE),
            'three_column': re.compile(r'grid-template-columns:\s*\d+\s+\d+\s+\d+', re.IGNORECASE),
            'card_layout': re.compile(r'(card|grid-gap|gap):', re.IGNORECASE),
            'hero_section': re.compile(r'(hero|banner|jumbotron)', re.IGNORECASE),
        }
    
    def analyze_html_structure(self, html: str) -> Dict:
        """Analyze HTML structure patterns"""
        structure = {
            'has_header': bool(re.search(r'<header', html, re.IGNORECASE)),
            'has_nav': bool(re.search(r'<nav', html, re.IGNORECASE)),
            'has_sidebar': bool(re.search(r'(sidebar|aside)', html, re.IGNORECASE)),
            'has_footer': bool(re.search(r'<footer', html, re.IGNORECASE)),
            'has_articles': html.lower().count('<article') > 0,
            'article_count': html.lower().count('<article'),
            'section_count': html.lower().count('<section'),
            'card_count': len(re.findall(r'class=["\'][^"\']*card[^"\']*["\']', html, re.IGNORECASE)),
        }
        
        # Infer layout type
        if structure['card_count'] > 3:
            structure['layout_type'] = 'card_grid'
        elif structure['has_sidebar']:
            structure['layout_type'] = 'two_column'
        elif structure['article_count'] > 3:
            structure['layout_type'] = 'blog_list'
        elif structure['section_count'] > 5:
            structure['layout_type'] = 'multi_section'
        else:
            structure['layout_type'] = 'single_page'
        
        return structure
    
    def analyze_css_theme(self, css: str) -> Dict:
        """Extract visual theme from CSS"""
        theme = {
            'uses_dark_mode': bool(re.search(r'background:\s*#[0-3][0-9a-f]{5}', css, re.IGNORECASE)),
            'uses_light_mode': bool(re.search(r'background:\s*#[e-f][0-9a-f]{5}', css, re.IGNORECASE)),
        }
        
        # Extract color palette
        colors = re.findall(r'#[0-9a-fA-F]{6}', css)
        theme['color_palette'] = list(set(colors))[:10]  # Top 10 unique colors
        
        # Extract font families
        fonts = re.findall(r'font-family:\s*([^;]+);', css, re.IGNORECASE)
        theme['font_families'] = list(set(fonts))[:5]
        
        # Spacing/size analysis
        theme['uses_rem'] = 'rem' in css
        theme['uses_em'] = 'em' in css
        theme['uses_viewport_units'] = 'vw' in css or 'vh' in css
        
        return theme


class ThemeGenerator:
    """Generate alternative visual themes"""
    
    def __init__(self):
        self.color_schemes = {
            'blue_modern': {
                'primary': '#2563eb',
                'secondary': '#3b82f6',
                'accent': '#60a5fa',
                'background': '#f8fafc',
                'text': '#1e293b',
            },
            'green_nature': {
                'primary': '#059669',
                'secondary': '#10b981',
                'accent': '#34d399',
                'background': '#f0fdf4',
                'text': '#064e3b',
            },
            'purple_creative': {
                'primary': '#7c3aed',
                'secondary': '#8b5cf6',
                'accent': '#a78bfa',
                'background': '#faf5ff',
                'text': '#4c1d95',
            },
            'dark_elegant': {
                'primary': '#fbbf24',
                'secondary': '#f59e0b',
                'accent': '#d97706',
                'background': '#1f2937',
                'text': '#f9fafb',
            },
            'minimalist': {
                'primary': '#64748b',
                'secondary': '#475569',
                'accent': '#94a3b8',
                'background': '#ffffff',
                'text': '#0f172a',
            },
        }
        
        self.layout_templates = {
            'card_grid': {
                'container': 'display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 2rem;',
                'card': 'padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);',
            },
            'two_column': {
                'container': 'display: grid; grid-template-columns: 2fr 1fr; gap: 2rem;',
                'main': 'padding: 2rem;',
                'sidebar': 'padding: 1.5rem; background: var(--background-alt);',
            },
            'single_page': {
                'container': 'max-width: 800px; margin: 0 auto; padding: 2rem;',
            },
        }
    
    def generate_theme(self, structure: Dict, original_theme: Dict, 
                       scheme_name: str = 'blue_modern') -> Dict:
        """Generate alternative theme based on structure and scheme"""
        colors = self.color_schemes.get(scheme_name, self.color_schemes['blue_modern'])
        layout_type = structure.get('layout_type', 'single_page')
        layout = self.layout_templates.get(layout_type, self.layout_templates['single_page'])
        
        theme = {
            'scheme_name': scheme_name,
            'colors': colors,
            'layout_css': layout,
            'structure_preserved': True,
            'responsive': True,
        }
        
        # Generate CSS
        css_rules = [
            ':root {',
            f"  --primary: {colors['primary']};",
            f"  --secondary: {colors['secondary']};",
            f"  --accent: {colors['accent']};",
            f"  --background: {colors['background']};",
            f"  --text: {colors['text']};",
            '}',
            '',
            'body {',
            f"  background: {colors['background']};",
            f"  color: {colors['text']};",
            '  font-family: system-ui, -apple-system, sans-serif;',
            '  line-height: 1.6;',
            '}',
            '',
        ]
        
        # Add layout-specific CSS
        for selector, rules in layout.items():
            css_rules.append(f'.{selector} {{')
            css_rules.append(f'  {rules}')
            css_rules.append('}')
            css_rules.append('')
        
        theme['generated_css'] = '\n'.join(css_rules)
        
        return theme
    
    def generate_all_themes(self, structure: Dict, original_theme: Dict) -> List[Dict]:
        """Generate all available theme variations"""
        themes = []
        for scheme_name in self.color_schemes:
            theme = self.generate_theme(structure, original_theme, scheme_name)
            themes.append(theme)
        return themes


class LayoutRecommender:
    """Main recommender combining analysis and generation"""
    
    def __init__(self):
        self.analyzer = LayoutAnalyzer()
        self.generator = ThemeGenerator()
    
    def analyze_and_recommend(self, html: str, css: str) -> Dict:
        """Analyze site and recommend alternative themes"""
        # Analyze current structure
        structure = self.analyzer.analyze_html_structure(html)
        theme = self.analyzer.analyze_css_theme(css)
        
        # Generate alternatives
        alternative_themes = self.generator.generate_all_themes(structure, theme)
        
        return {
            'original_structure': structure,
            'original_theme': theme,
            'alternative_themes': alternative_themes,
            'recommendation_count': len(alternative_themes),
        }
    
    def save_recommendations(self, recommendations: Dict, output_path: Path):
        """Save recommendations to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False)
        print(f"💾 Recommendations saved to: {output_path}")


def main():
    """Generate theme recommendations from feedback data"""
    features_dir = Path(__file__).parent.parent / 'features'
    models_dir = Path(__file__).parent.parent / 'models'
    
    # Load features
    features_path = features_dir / 'extracted_features.jsonl'
    if not features_path.exists():
        print("❌ Features not found. Run extract_features.py first.")
        return
    
    print(f"📂 Loading features from: {features_path}")
    
    recommender = LayoutRecommender()
    sample_count = 0
    
    # Process first few HTML samples to demonstrate
    with open(features_path, 'r', encoding='utf-8') as f:
        for line in f:
            feat = json.loads(line)
            if feat.get('event_type') != 'html_parsing':
                continue
            
            html_feat = feat.get('html_features', {})
            css_feat = feat.get('css_features', {})
            
            if not html_feat.get('has_html'):
                continue
            
            # For demo, create mock HTML/CSS from features
            mock_html = f"<html><body>{'<article>' * html_feat.get('article_count', 0)}</body></html>"
            mock_css = "body { background: #ffffff; }"
            
            recommendations = recommender.analyze_and_recommend(mock_html, mock_css)
            
            # Save first sample recommendations
            if sample_count == 0:
                output_path = models_dir / 'sample_theme_recommendations.json'
                recommender.save_recommendations(recommendations, output_path)
            
            print(f"\n🎨 Recommendations for sample {sample_count + 1}:")
            print(f"   Layout type: {recommendations['original_structure']['layout_type']}")
            print(f"   Generated {recommendations['recommendation_count']} alternative themes")
            
            sample_count += 1
            if sample_count >= 3:
                break
    
    print(f"\n✅ Processed {sample_count} samples")
    print("\n💡 Theme recommender ready!")
    print("   Use LayoutRecommender.analyze_and_recommend(html, css) to get alternatives")


if __name__ == '__main__':
    main()
