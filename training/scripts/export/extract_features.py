#!/usr/bin/env python3
"""
Feature extractor for HTML/CSS/JS content
Converts raw feedback data into structured feature vectors for ML training
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import hashlib


class FeatureExtractor:
    """Extract ML features from HTML/CSS/JS content"""
    
    def __init__(self):
        self.url_patterns = {
            'news': re.compile(r'news|xinhua|cnn|bbc|reuters|headlines', re.IGNORECASE),
            'ecommerce': re.compile(r'taobao|jd|amazon|shop|mall|ebay', re.IGNORECASE),
            'tech': re.compile(r'github|stackoverflow|dev|api|docs', re.IGNORECASE),
            'social': re.compile(r'facebook|twitter|weibo|zhihu|douban', re.IGNORECASE),
            'video': re.compile(r'youtube|bilibili|youku|video', re.IGNORECASE),
            'education': re.compile(r'\.edu|coursera|udemy|mooc', re.IGNORECASE),
            'government': re.compile(r'\.gov|\.mil', re.IGNORECASE),
            'finance': re.compile(r'bank|finance|trading', re.IGNORECASE),
        }
        
        self.content_keywords = {
            'news': ['article', 'news', 'breaking', 'headline', 'reporter'],
            'ecommerce': ['cart', 'buy', 'shop', 'product', 'price', 'checkout'],
            'tech': ['documentation', 'api', 'code', 'developer', 'tutorial'],
            'social': ['follow', 'share', 'post', 'comment', 'profile'],
            'video': ['video', 'watch', 'play', 'channel', 'subscribe'],
            'education': ['course', 'learn', 'student', 'lecture', 'university'],
            'government': ['government', 'official', 'ministry', 'policy'],
            'finance': ['investment', 'stock', 'trading', 'account'],
        }
        
    def extract_url_features(self, url: str) -> Dict:
        """Extract features from URL"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        
        # Infer category from URL
        category = 'other'
        for cat, pattern in self.url_patterns.items():
            if pattern.search(domain) or pattern.search(path):
                category = cat
                break
        
        return {
            'url': url,
            'domain': domain,
            'domain_length': len(domain),
            'path_depth': path.count('/'),
            'has_query': '?' in url,
            'inferred_category': category,
        }
    
    def extract_html_features(self, html: str, metadata: Dict) -> Dict:
        """Extract features from HTML content"""
        if not html:
            return {'has_html': False}
        
        features = {
            'has_html': True,
            'size': len(html),
            'lines': html.count('\n') + 1,
        }
        
        # Structure features
        features['div_count'] = html.lower().count('<div')
        features['section_count'] = html.lower().count('<section')
        features['article_count'] = html.lower().count('<article')
        features['nav_count'] = html.lower().count('<nav')
        features['form_count'] = html.lower().count('<form')
        features['table_count'] = html.lower().count('<table')
        features['img_count'] = html.lower().count('<img')
        features['link_count'] = html.lower().count('<a ')
        
        # External resources
        features['external_js'] = len(re.findall(r'<script[^>]*src=', html, re.IGNORECASE))
        features['external_css'] = len(re.findall(r'<link[^>]*rel=["\']stylesheet', html, re.IGNORECASE))
        features['inline_style'] = len(re.findall(r'<style[^>]*>', html, re.IGNORECASE))
        features['inline_script'] = len(re.findall(r'<script[^>]*>[^<]+</script>', html, re.IGNORECASE))
        
        # Semantic tags ratio
        semantic_tags = features['section_count'] + features['article_count'] + features['nav_count']
        total_tags = features['div_count'] + semantic_tags
        features['semantic_ratio'] = semantic_tags / max(total_tags, 1)
        
        # Extract title and meta
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        features['title'] = title_match.group(1).strip() if title_match else ''
        features['title_length'] = len(features['title'])
        
        # Category inference from content keywords
        html_lower = html.lower()
        category_scores = {}
        for cat, keywords in self.content_keywords.items():
            score = sum(html_lower.count(kw) for kw in keywords)
            category_scores[cat] = score
        
        if category_scores:
            best_cat = max(category_scores, key=category_scores.get)
            if category_scores[best_cat] > 0:
                features['content_category'] = best_cat
                features['content_category_score'] = category_scores[best_cat]
        
        return features
    
    def extract_css_features(self, css: str, metadata: Dict) -> Dict:
        """Extract features from CSS content"""
        if not css:
            return {'has_css': False}
        
        features = {
            'has_css': True,
            'size': len(css),
            'lines': css.count('\n') + 1,
        }
        
        # Minification check
        avg_line_length = features['size'] / max(features['lines'], 1)
        features['is_minified'] = avg_line_length > 500
        
        # Selector complexity
        features['class_selectors'] = css.count('.')
        features['id_selectors'] = css.count('#')
        features['attribute_selectors'] = css.count('[')
        features['pseudo_selectors'] = css.count(':')
        
        # Obfuscation indicators
        short_classes = len(re.findall(r'\.[a-zA-Z0-9_]{1,3}\s*\{', css))
        long_classes = len(re.findall(r'\.[a-zA-Z0-9_]{20,}', css))
        features['short_class_names'] = short_classes
        features['long_class_names'] = long_classes
        features['obfuscation_score'] = (short_classes * 0.5 + long_classes * 0.3) / max(features['class_selectors'], 1) * 100
        
        # Modern CSS features
        features['uses_flexbox'] = 'flex' in css
        features['uses_grid'] = 'grid' in css
        features['uses_variables'] = 'var(' in css or '--' in css
        features['uses_media_queries'] = '@media' in css
        
        return features
    
    def extract_js_features(self, js: str, metadata: Dict) -> Dict:
        """Extract features from JavaScript content"""
        if not js:
            return {'has_js': False}
        
        features = {
            'has_js': True,
            'size': len(js),
            'lines': js.count('\n') + 1,
        }
        
        # Minification check
        avg_line_length = features['size'] / max(features['lines'], 1)
        features['is_minified'] = avg_line_length > 1000
        
        # Obfuscation indicators
        features['has_eval'] = 'eval(' in js
        features['has_function_constructor'] = 'Function(' in js
        features['hex_strings'] = len(re.findall(r'\\x[0-9a-fA-F]{2}', js))
        features['unicode_escapes'] = len(re.findall(r'\\u[0-9a-fA-F]{4}', js))
        features['short_var_names'] = len(re.findall(r'\b[a-z_$][a-z0-9_$]{0,2}\s*[=:]', js, re.IGNORECASE))
        
        # Build tool detection
        features['webpack'] = '__webpack' in js
        features['rollup'] = '__rollup' in js
        features['parcel'] = '__parcel' in js
        
        # Framework detection
        features['react'] = 'React' in js or 'react' in js
        features['vue'] = 'Vue' in js or 'vue' in js
        features['angular'] = 'angular' in js or 'ng-' in js
        features['jquery'] = 'jQuery' in js or '$(' in js
        
        # Obfuscation score
        obf_score = 0
        if features['is_minified']: obf_score += 30
        if features['short_var_names'] > 10: obf_score += 20
        if features['hex_strings'] > 5: obf_score += 15
        if features['unicode_escapes'] > 5: obf_score += 15
        if features['has_eval']: obf_score += 10
        if features['has_function_constructor']: obf_score += 10
        features['obfuscation_score'] = min(obf_score, 100)
        
        return features
    
    def extract_feedback_features(self, feedback_data: Dict, url: str) -> Dict:
        """Extract all features from a feedback event"""
        event_type = feedback_data.get('type')
        content = feedback_data.get('content', '')
        
        # Base features
        features = {
            'event_type': event_type,
            'timestamp': feedback_data.get('timestamp'),
            'success': feedback_data.get('success', True),
            'ai_used': feedback_data.get('ai_used', False),
        }
        
        # URL features
        features['url_features'] = self.extract_url_features(url)
        
        # Content-specific features
        if event_type == 'html_parsing':
            features['html_features'] = self.extract_html_features(content, feedback_data)
        elif event_type == 'css_parsing':
            features['css_features'] = self.extract_css_features(content, feedback_data)
        elif event_type == 'js_parsing':
            features['js_features'] = self.extract_js_features(content, feedback_data)
        
        return features
    
    def process_feedback_file(self, file_path: Path) -> List[Dict]:
        """Process a single feedback JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            events = json.load(f)
        
        # Group events by URL (infer from sequence)
        # Assume first HTML event sets the URL context
        current_url = None
        results = []
        
        for event in events:
            event_type = event.get('type')
            
            # Try to infer URL from event or filename
            if not current_url and event_type == 'html_parsing':
                # Use filename as hint or extract from metadata
                current_url = f"https://unknown-{file_path.stem}"
            
            if current_url:
                features = self.extract_feedback_features(event, current_url)
                features['source_file'] = file_path.name
                results.append(features)
        
        return results
    
    def flatten_features(self, features: Dict) -> Dict:
        """Flatten nested feature dict for ML training"""
        flat = {}
        
        def flatten_dict(d: Dict, prefix: str = ''):
            for k, v in d.items():
                key = f"{prefix}{k}" if prefix else k
                if isinstance(v, dict):
                    flatten_dict(v, f"{key}_")
                elif isinstance(v, (int, float, bool)):
                    flat[key] = float(v) if isinstance(v, bool) else v
                elif isinstance(v, str):
                    # Hash strings to numeric (for domain, title, etc.)
                    if k in ['domain', 'title', 'url', 'timestamp', 'source_file', 'event_type']:
                        flat[key] = k  # Keep as metadata
                    else:
                        flat[key] = v
        
        flatten_dict(features)
        return flat


def main():
    """Extract features from all feedback files"""
    extractor = FeatureExtractor()
    data_dir = Path(__file__).parent.parent / 'data'
    features_dir = Path(__file__).parent.parent / 'features'
    
    # Process all feedback files
    all_features = []
    feedback_files = sorted(data_dir.glob('feedback_*.json'))
    
    print(f"📊 Processing {len(feedback_files)} feedback files...")
    
    for file_path in feedback_files:
        try:
            features_list = extractor.process_feedback_file(file_path)
            all_features.extend(features_list)
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
    
    print(f"✅ Extracted features from {len(all_features)} events\n")
    
    # Save as JSONL (one JSON per line)
    output_path = features_dir / 'extracted_features.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for feat in all_features:
            f.write(json.dumps(feat, ensure_ascii=False) + '\n')
    
    print(f"💾 Features saved to: {output_path}")
    
    # Print statistics
    html_count = sum(1 for f in all_features if f.get('event_type') == 'html_parsing')
    css_count = sum(1 for f in all_features if f.get('event_type') == 'css_parsing')
    js_count = sum(1 for f in all_features if f.get('event_type') == 'js_parsing')
    
    print(f"\n📈 Feature extraction summary:")
    print(f"   HTML events: {html_count}")
    print(f"   CSS events:  {css_count}")
    print(f"   JS events:   {js_count}")
    
    # Category distribution (from HTML events)
    categories = {}
    for f in all_features:
        if f.get('event_type') == 'html_parsing':
            cat = f.get('url_features', {}).get('inferred_category', 'other')
            categories[cat] = categories.get(cat, 0) + 1
    
    if categories:
        print(f"\n🏷️  Inferred site categories:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cat}: {count}")


if __name__ == '__main__':
    main()
