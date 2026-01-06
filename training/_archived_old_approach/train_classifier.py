#!/usr/bin/env python3
"""
Train site category and tech stack classifier
Small, specialized model for fast inference in browser
"""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter


class SimpleClassifier:
    """Lightweight multi-label classifier for site category + tech stack"""
    
    def __init__(self):
        self.category_weights = {}
        self.tech_weights = {}
        self.feature_names = []
        self.categories = []
        self.tech_labels = []
        
    def extract_numeric_features(self, features: Dict) -> np.ndarray:
        """Extract numeric feature vector from feature dict"""
        numeric_features = []
        
        # URL features
        url_feat = features.get('url_features', {})
        numeric_features.extend([
            url_feat.get('domain_length', 0),
            url_feat.get('path_depth', 0),
            float(url_feat.get('has_query', False)),
        ])
        
        # HTML features
        html_feat = features.get('html_features', {})
        if html_feat.get('has_html'):
            numeric_features.extend([
                html_feat.get('size', 0) / 1000,  # Normalize to KB
                html_feat.get('div_count', 0),
                html_feat.get('section_count', 0),
                html_feat.get('article_count', 0),
                html_feat.get('nav_count', 0),
                html_feat.get('form_count', 0),
                html_feat.get('img_count', 0),
                html_feat.get('link_count', 0),
                html_feat.get('external_js', 0),
                html_feat.get('external_css', 0),
                html_feat.get('semantic_ratio', 0),
                html_feat.get('title_length', 0),
            ])
        else:
            numeric_features.extend([0] * 12)
        
        # CSS features
        css_feat = features.get('css_features', {})
        if css_feat.get('has_css'):
            numeric_features.extend([
                css_feat.get('size', 0) / 1000,
                float(css_feat.get('is_minified', False)),
                css_feat.get('class_selectors', 0),
                css_feat.get('obfuscation_score', 0),
                float(css_feat.get('uses_flexbox', False)),
                float(css_feat.get('uses_grid', False)),
            ])
        else:
            numeric_features.extend([0] * 6)
        
        # JS features
        js_feat = features.get('js_features', {})
        if js_feat.get('has_js'):
            numeric_features.extend([
                js_feat.get('size', 0) / 1000,
                float(js_feat.get('is_minified', False)),
                float(js_feat.get('has_eval', False)),
                js_feat.get('obfuscation_score', 0),
                float(js_feat.get('webpack', False)),
                float(js_feat.get('react', False)),
                float(js_feat.get('vue', False)),
                float(js_feat.get('angular', False)),
                float(js_feat.get('jquery', False)),
            ])
        else:
            numeric_features.extend([0] * 9)
        
        return np.array(numeric_features, dtype=np.float32)
    
    def train(self, features_list: List[Dict]):
        """Train simple linear classifier"""
        print("🏋️ Training classifier...")
        
        # Extract features and labels
        X = []
        category_labels = []
        tech_labels = []
        
        for feat in features_list:
            # Only train on HTML events (most informative)
            if feat.get('event_type') != 'html_parsing':
                continue
            
            x = self.extract_numeric_features(feat)
            X.append(x)
            
            # Category label
            url_cat = feat.get('url_features', {}).get('inferred_category', 'other')
            content_cat = feat.get('html_features', {}).get('content_category', url_cat)
            category_labels.append(content_cat)
            
            # Tech stack labels (multi-label)
            tech = []
            js_feat = feat.get('js_features', {})
            if js_feat.get('react'): tech.append('react')
            if js_feat.get('vue'): tech.append('vue')
            if js_feat.get('angular'): tech.append('angular')
            if js_feat.get('jquery'): tech.append('jquery')
            if js_feat.get('webpack'): tech.append('webpack')
            
            css_feat = feat.get('css_features', {})
            if css_feat.get('obfuscation_score', 0) > 50:
                tech.append('obfuscated_css')
            if js_feat.get('obfuscation_score', 0) > 50:
                tech.append('obfuscated_js')
            
            tech_labels.append(tech)
        
        if not X:
            print("❌ No training data available")
            return
        
        X = np.array(X)
        
        # Store categories and tech labels
        self.categories = list(set(category_labels))
        all_techs = set()
        for techs in tech_labels:
            all_techs.update(techs)
        self.tech_labels = list(all_techs)
        
        # Simple mean-based classifier (centroid for each category)
        self.category_weights = {}
        for cat in self.categories:
            mask = np.array([label == cat for label in category_labels])
            if mask.sum() > 0:
                self.category_weights[cat] = X[mask].mean(axis=0)
        
        # Tech stack weights (for each tech, compute mean of samples with that tech)
        self.tech_weights = {}
        for tech in self.tech_labels:
            mask = np.array([tech in techs for techs in tech_labels])
            if mask.sum() > 0:
                self.tech_weights[tech] = X[mask].mean(axis=0)
        
        print(f"✅ Trained on {len(X)} samples")
        print(f"   Categories: {len(self.categories)}")
        print(f"   Tech labels: {len(self.tech_labels)}")
    
    def predict(self, features: Dict) -> Tuple[str, List[str], Dict]:
        """Predict category and tech stack"""
        x = self.extract_numeric_features(features)
        
        # Predict category (nearest centroid)
        best_cat = 'other'
        best_dist = float('inf')
        category_scores = {}
        
        for cat, centroid in self.category_weights.items():
            dist = np.linalg.norm(x - centroid)
            category_scores[cat] = 1.0 / (1.0 + dist)  # Convert to similarity score
            if dist < best_dist:
                best_dist = dist
                best_cat = cat
        
        # Predict tech stack (threshold-based)
        predicted_tech = []
        tech_scores = {}
        
        for tech, centroid in self.tech_weights.items():
            sim = 1.0 / (1.0 + np.linalg.norm(x - centroid))
            tech_scores[tech] = sim
            if sim > 0.6:  # Threshold
                predicted_tech.append(tech)
        
        return best_cat, predicted_tech, {
            'category_scores': category_scores,
            'tech_scores': tech_scores,
        }
    
    def save(self, path: Path):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'category_weights': self.category_weights,
                'tech_weights': self.tech_weights,
                'categories': self.categories,
                'tech_labels': self.tech_labels,
            }, f)
        print(f"💾 Model saved to: {path}")
    
    def load(self, path: Path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.category_weights = data['category_weights']
            self.tech_weights = data['tech_weights']
            self.categories = data['categories']
            self.tech_labels = data['tech_labels']
        print(f"📥 Model loaded from: {path}")


def main():
    """Train and save classifier"""
    features_dir = Path(__file__).parent.parent / 'features'
    models_dir = Path(__file__).parent.parent / 'models'
    
    # Load extracted features
    features_path = features_dir / 'extracted_features.jsonl'
    if not features_path.exists():
        print(f"❌ Features not found. Run extract_features.py first.")
        return
    
    print(f"📂 Loading features from: {features_path}")
    features_list = []
    with open(features_path, 'r', encoding='utf-8') as f:
        for line in f:
            features_list.append(json.loads(line))
    
    print(f"✅ Loaded {len(features_list)} feature records\n")
    
    # Train classifier
    classifier = SimpleClassifier()
    classifier.train(features_list)
    
    # Save model
    model_path = models_dir / 'site_classifier.pkl'
    classifier.save(model_path)
    
    # Test on some samples
    print("\n🧪 Testing on sample predictions:")
    html_features = [f for f in features_list if f.get('event_type') == 'html_parsing']
    if html_features:
        for i, feat in enumerate(html_features[:3]):
            cat, tech, scores = classifier.predict(feat)
            url = feat.get('url_features', {}).get('url', 'unknown')
            print(f"\n   Sample {i+1}: {url}")
            print(f"   Category: {cat} (score: {scores['category_scores'].get(cat, 0):.2f})")
            print(f"   Tech stack: {', '.join(tech) if tech else 'none'}")


if __name__ == '__main__':
    main()
