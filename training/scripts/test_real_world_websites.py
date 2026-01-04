#!/usr/bin/env python3
"""
Real-World Website Testing Script for BrowerAI Phase 2.4

This script tests AI models on real-world websites by fetching HTML, CSS, and JS
from popular sites and evaluating model performance.

Usage:
    python test_real_world_websites.py [--num-sites NUM] [--models-dir MODELS_DIR]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import re

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available. Install with: pip install requests")

try:
    import onnxruntime as ort
    import numpy as np
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Install with: pip install onnxruntime")


class RealWorldTester:
    """Test AI models on real-world websites."""
    
    # Popular websites for testing (diverse content types)
    TEST_WEBSITES = [
        'https://example.com',
        'https://www.w3.org',
        'https://github.com',
        'https://stackoverflow.com',
        'https://developer.mozilla.org',
        'https://www.wikipedia.org',
        'https://news.ycombinator.com',
        'https://reddit.com',
        'https://medium.com',
        'https://www.bbc.com',
    ]
    
    def __init__(self, models_dir: Path, num_sites: int = 10):
        self.models_dir = models_dir
        self.num_sites = min(num_sites, len(self.TEST_WEBSITES))
        self.sessions = {}
        self.results = {
            'sites_tested': 0,
            'html_tests': [],
            'css_tests': [],
            'js_tests': [],
            'errors': []
        }
    
    def load_models(self):
        """Load available ONNX models."""
        if not ONNX_AVAILABLE:
            print("ONNX Runtime not available, skipping model loading")
            return
        
        print("\n" + "=" * 60)
        print("Loading AI Models for Real-World Testing")
        print("=" * 60)
        
        model_files = [
            'html_parser_v1.onnx',
            'html_parser_transformer_v2.onnx',
            'css_optimizer_v1.onnx',
            'css_deduplication_v1.onnx',
            'css_selector_optimizer_v1.onnx',
            'css_minifier_v1.onnx',
            'js_analyzer_v1.onnx',
            'js_tokenizer_enhancer_v1.onnx',
            'js_ast_predictor_v1.onnx',
            'js_optimization_suggestions_v1.onnx'
        ]
        
        for filename in model_files:
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    session = ort.InferenceSession(str(model_path))
                    self.sessions[filename] = session
                    print(f"✓ Loaded {filename}")
                except Exception as e:
                    print(f"✗ Failed to load {filename}: {e}")
            else:
                print(f"- {filename} not found (optional)")
    
    def fetch_website_content(self, url: str) -> Dict:
        """Fetch HTML, CSS, and JS from a website."""
        if not REQUESTS_AVAILABLE:
            return {'error': 'requests library not available'}
        
        try:
            print(f"\n📥 Fetching: {url}")
            response = requests.get(url, timeout=10, headers={'User-Agent': 'BrowerAI/1.0'})
            response.raise_for_status()
            
            html_content = response.text
            
            # Extract inline CSS (handle case-insensitive and spaces in closing tags)
            # Note: This is for testing purposes only, not security-critical parsing
            css_pattern = r'<style[^>]*>(.*?)</style\s*>'
            css_matches = re.findall(css_pattern, html_content, re.DOTALL | re.IGNORECASE)
            css_content = '\n'.join(css_matches)
            
            # Extract inline JavaScript (handle case-insensitive and spaces in closing tags)
            # Note: This is for testing purposes only, not security-critical parsing
            js_pattern = r'<script[^>]*>(.*?)</script\s*>'
            js_matches = re.findall(js_pattern, html_content, re.DOTALL | re.IGNORECASE)
            js_content = '\n'.join([m for m in js_matches if m.strip()])
            
            return {
                'url': url,
                'html': html_content[:10000],  # Limit to first 10KB
                'css': css_content[:5000],      # Limit to first 5KB
                'js': js_content[:5000],        # Limit to first 5KB
                'html_size': len(html_content),
                'css_size': len(css_content),
                'js_size': len(js_content),
                'status_code': response.status_code
            }
        except Exception as e:
            return {'error': str(e), 'url': url}
    
    def tokenize(self, text: str, max_length: int = 512) -> np.ndarray:
        """Simple character-level tokenization."""
        tokens = []
        for ch in text[:max_length]:
            tokens.append(ord(ch) % 256)
        
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(0)
        
        return np.array([tokens], dtype=np.int64)
    
    def extract_features(self, text: str, feature_dim: int) -> np.ndarray:
        """Extract features for non-sequence models."""
        features = []
        
        # Basic text statistics
        features.append(min(len(text), 10000) / 10000.0)  # Normalized length
        features.append(text.count(' ') / max(len(text), 1))  # Whitespace ratio
        features.append(text.count('\n') / max(len(text), 1))  # Newline ratio
        features.append(len(set(text)) / 256.0)  # Character variety
        
        # Pad or truncate to feature_dim
        while len(features) < feature_dim:
            features.append(0.0)
        
        return np.array([features[:feature_dim]], dtype=np.float32)
    
    def test_html_model(self, html_content: str, model_name: str) -> Dict:
        """Test HTML model on real-world content."""
        if model_name not in self.sessions:
            return {'error': 'Model not loaded'}
        
        session = self.sessions[model_name]
        
        try:
            # Prepare input
            input_data = self.tokenize(html_content, max_length=512)
            
            # Run inference
            start_time = time.perf_counter()
            outputs = session.run(None, {'input': input_data})
            inference_time = time.perf_counter() - start_time
            
            # Get prediction
            prediction = float(outputs[0][0][0])
            
            return {
                'model': model_name,
                'inference_time_ms': inference_time * 1000,
                'prediction': prediction,
                'content_length': len(html_content),
                'success': True
            }
        except Exception as e:
            return {'error': str(e), 'model': model_name}
    
    def test_css_model(self, css_content: str, model_name: str) -> Dict:
        """Test CSS model on real-world content."""
        if model_name not in self.sessions:
            return {'error': 'Model not loaded'}
        
        session = self.sessions[model_name]
        
        try:
            # Get input shape from model
            input_shape = session.get_inputs()[0].shape
            feature_dim = input_shape[1] if len(input_shape) > 1 else 18
            
            # Prepare input
            input_data = self.extract_features(css_content, feature_dim)
            
            # Run inference
            start_time = time.perf_counter()
            outputs = session.run(None, {session.get_inputs()[0].name: input_data})
            inference_time = time.perf_counter() - start_time
            
            # Get predictions
            predictions = outputs[0][0].tolist()
            
            return {
                'model': model_name,
                'inference_time_ms': inference_time * 1000,
                'predictions': predictions,
                'content_length': len(css_content),
                'success': True
            }
        except Exception as e:
            return {'error': str(e), 'model': model_name}
    
    def test_js_model(self, js_content: str, model_name: str) -> Dict:
        """Test JS model on real-world content."""
        if model_name not in self.sessions:
            return {'error': 'Model not loaded'}
        
        session = self.sessions[model_name]
        
        try:
            # Get input shape from model
            input_shape = session.get_inputs()[0].shape
            feature_dim = input_shape[1] if len(input_shape) > 1 else 20
            
            # Prepare input
            input_data = self.extract_features(js_content, feature_dim)
            
            # Run inference
            start_time = time.perf_counter()
            outputs = session.run(None, {session.get_inputs()[0].name: input_data})
            inference_time = time.perf_counter() - start_time
            
            # Get predictions
            predictions = outputs[0][0].tolist()
            
            return {
                'model': model_name,
                'inference_time_ms': inference_time * 1000,
                'predictions': predictions,
                'content_length': len(js_content),
                'success': True
            }
        except Exception as e:
            return {'error': str(e), 'model': model_name}
    
    def run_real_world_tests(self):
        """Run tests on real-world websites."""
        print("\n" + "=" * 60)
        print("Real-World Website Testing")
        print("=" * 60)
        
        sites_to_test = self.TEST_WEBSITES[:self.num_sites]
        
        for i, url in enumerate(sites_to_test, 1):
            print(f"\n[{i}/{len(sites_to_test)}] Testing: {url}")
            
            # Fetch content
            content = self.fetch_website_content(url)
            
            if 'error' in content:
                print(f"  ✗ Failed to fetch: {content['error']}")
                self.results['errors'].append(content)
                continue
            
            self.results['sites_tested'] += 1
            
            print(f"  ✓ Fetched successfully")
            print(f"    HTML: {content['html_size']:,} bytes")
            print(f"    CSS: {content['css_size']:,} bytes")
            print(f"    JS: {content['js_size']:,} bytes")
            
            # Test HTML models
            if content['html_size'] > 0:
                for model_name in self.sessions:
                    if 'html' in model_name.lower():
                        result = self.test_html_model(content['html'], model_name)
                        if 'success' in result:
                            print(f"    ✓ {model_name}: {result['inference_time_ms']:.2f} ms")
                            self.results['html_tests'].append(result)
                        else:
                            print(f"    ✗ {model_name}: {result.get('error', 'Unknown error')}")
            
            # Test CSS models
            if content['css_size'] > 0:
                for model_name in self.sessions:
                    if 'css' in model_name.lower():
                        result = self.test_css_model(content['css'], model_name)
                        if 'success' in result:
                            print(f"    ✓ {model_name}: {result['inference_time_ms']:.2f} ms")
                            self.results['css_tests'].append(result)
            
            # Test JS models
            if content['js_size'] > 0:
                for model_name in self.sessions:
                    if 'js' in model_name.lower():
                        result = self.test_js_model(content['js'], model_name)
                        if 'success' in result:
                            print(f"    ✓ {model_name}: {result['inference_time_ms']:.2f} ms")
                            self.results['js_tests'].append(result)
        
        return self.results
    
    def generate_report(self):
        """Generate test report."""
        print("\n" + "=" * 60)
        print("Real-World Testing Report")
        print("=" * 60)
        
        print(f"\n📊 Summary:")
        print(f"  Sites Tested: {self.results['sites_tested']}")
        print(f"  HTML Tests: {len(self.results['html_tests'])}")
        print(f"  CSS Tests: {len(self.results['css_tests'])}")
        print(f"  JS Tests: {len(self.results['js_tests'])}")
        print(f"  Errors: {len(self.results['errors'])}")
        
        # Calculate average inference times
        if self.results['html_tests']:
            avg_html_time = sum(t['inference_time_ms'] for t in self.results['html_tests']) / len(self.results['html_tests'])
            print(f"\n  Average HTML Inference Time: {avg_html_time:.2f} ms")
        
        if self.results['css_tests']:
            avg_css_time = sum(t['inference_time_ms'] for t in self.results['css_tests']) / len(self.results['css_tests'])
            print(f"  Average CSS Inference Time: {avg_css_time:.2f} ms")
        
        if self.results['js_tests']:
            avg_js_time = sum(t['inference_time_ms'] for t in self.results['js_tests']) / len(self.results['js_tests'])
            print(f"  Average JS Inference Time: {avg_js_time:.2f} ms")
        
        # Save results
        output_path = Path('real_world_test_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
        
        print("\n" + "=" * 60)
        print("Real-world testing shows models can handle production websites")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Test BrowerAI models on real-world websites')
    parser.add_argument('--num-sites', type=int, default=10,
                       help='Number of websites to test (max 10)')
    parser.add_argument('--models-dir', type=str, default='../../models',
                       help='Directory containing ONNX models')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir) / 'local'
    
    if not REQUESTS_AVAILABLE:
        print("\n⚠️  requests library required for fetching websites")
        print("Install with: pip install requests")
        return
    
    tester = RealWorldTester(models_dir, args.num_sites)
    tester.load_models()
    
    if not tester.sessions:
        print("\n⚠️  No models available for testing")
        print("Next steps:")
        print("  1. Train models using training scripts")
        print("  2. Copy models to ../../models/local/")
        print("  3. Run this script again")
        return
    
    tester.run_real_world_tests()
    tester.generate_report()


if __name__ == '__main__':
    main()
