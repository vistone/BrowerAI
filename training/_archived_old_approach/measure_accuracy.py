#!/usr/bin/env python3
"""
Accuracy Measurement Script for BrowerAI Phase 2.4

This script measures accuracy improvements of AI models compared to traditional
parsing methods across various metrics.

Usage:
    python measure_accuracy.py [--models-dir MODELS_DIR] [--data-dir DATA_DIR]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

try:
    import onnxruntime as ort
    import numpy as np
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Install with: pip install onnxruntime")


class AccuracyMeasurement:
    """Measure accuracy improvements of AI models."""
    
    def __init__(self, models_dir: Path, data_dir: Path):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.sessions = {}
        self.results = {
            'accuracy_metrics': {},
            'improvement_metrics': {},
            'confusion_matrices': {}
        }
    
    def load_models(self):
        """Load available ONNX models."""
        if not ONNX_AVAILABLE:
            print("ONNX Runtime not available")
            return
        
        print("\n" + "=" * 60)
        print("Loading Models for Accuracy Measurement")
        print("=" * 60)
        
        model_files = {
            'html': ['html_parser_v1.onnx', 'html_parser_transformer_v2.onnx', 'html_parser_lstm_v2.onnx'],
            'css': ['css_optimizer_v1.onnx', 'css_deduplication_v1.onnx', 'css_selector_optimizer_v1.onnx', 'css_minifier_v1.onnx'],
            'js': ['js_analyzer_v1.onnx', 'js_tokenizer_enhancer_v1.onnx', 'js_ast_predictor_v1.onnx', 'js_optimization_suggestions_v1.onnx']
        }
        
        for model_type, files in model_files.items():
            for filename in files:
                model_path = self.models_dir / filename
                if model_path.exists():
                    try:
                        session = ort.InferenceSession(str(model_path))
                        self.sessions[filename] = session
                        print(f"✓ Loaded {filename}")
                    except Exception as e:
                        print(f"✗ Failed to load {filename}: {e}")
    
    def tokenize(self, text: str, max_length: int = 512) -> np.ndarray:
        """Character-level tokenization."""
        tokens = [ord(ch) % 256 for ch in text[:max_length]]
        tokens.extend([0] * (max_length - len(tokens)))
        return np.array([tokens], dtype=np.int64)
    
    def calculate_confusion_matrix(self, predictions: List[bool], ground_truth: List[bool]) -> Dict:
        """Calculate confusion matrix metrics."""
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
        tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def traditional_html_parser(self, html: str) -> bool:
        """Traditional HTML parsing heuristic."""
        html_lower = html.lower()
        return any(tag in html_lower for tag in ['<html', '<body', '<head', '<div', '<p'])
    
    def traditional_css_parser(self, css: str) -> bool:
        """Traditional CSS parsing heuristic."""
        return '{' in css and '}' in css and ':' in css
    
    def traditional_js_parser(self, js: str) -> bool:
        """Traditional JS parsing heuristic."""
        keywords = ['function', 'const', 'let', 'var', 'class', 'return']
        return any(kw in js for kw in keywords)
    
    def measure_html_accuracy(self) -> Dict:
        """Measure HTML parser accuracy."""
        test_file = self.data_dir / 'html' / 'test.json'
        if not test_file.exists():
            return {'error': 'Test data not found'}
        
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        results = {}
        ground_truth = [sample['label'] == 'valid' for sample in test_data]
        
        # Traditional baseline
        traditional_preds = [self.traditional_html_parser(sample['input']) for sample in test_data]
        traditional_metrics = self.calculate_confusion_matrix(traditional_preds, ground_truth)
        results['traditional'] = traditional_metrics
        
        print(f"\n📊 HTML Parser Accuracy")
        print("-" * 60)
        print(f"Traditional Parser:")
        print(f"  Accuracy: {traditional_metrics['accuracy']:.2%}")
        print(f"  Precision: {traditional_metrics['precision']:.2%}")
        print(f"  Recall: {traditional_metrics['recall']:.2%}")
        print(f"  F1 Score: {traditional_metrics['f1_score']:.3f}")
        
        # AI models
        for model_name in self.sessions:
            if 'html' in model_name.lower():
                session = self.sessions[model_name]
                ai_preds = []
                
                for sample in test_data:
                    input_data = self.tokenize(sample['input'])
                    outputs = session.run(None, {'input': input_data})
                    prediction = outputs[0][0][0] > 0.5
                    ai_preds.append(prediction)
                
                ai_metrics = self.calculate_confusion_matrix(ai_preds, ground_truth)
                results[model_name] = ai_metrics
                
                # Calculate improvement
                acc_improvement = ((ai_metrics['accuracy'] - traditional_metrics['accuracy']) / 
                                 traditional_metrics['accuracy'] * 100)
                
                print(f"\n{model_name}:")
                print(f"  Accuracy: {ai_metrics['accuracy']:.2%} ({acc_improvement:+.1f}%)")
                print(f"  Precision: {ai_metrics['precision']:.2%}")
                print(f"  Recall: {ai_metrics['recall']:.2%}")
                print(f"  F1 Score: {ai_metrics['f1_score']:.3f}")
        
        return results
    
    def measure_css_accuracy(self) -> Dict:
        """Measure CSS parser accuracy."""
        test_file = self.data_dir / 'css' / 'test.json'
        if not test_file.exists():
            return {'error': 'Test data not found'}
        
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        results = {}
        ground_truth = [sample['label'] == 'valid' for sample in test_data]
        
        # Traditional baseline
        traditional_preds = [self.traditional_css_parser(sample['input']) for sample in test_data]
        traditional_metrics = self.calculate_confusion_matrix(traditional_preds, ground_truth)
        results['traditional'] = traditional_metrics
        
        print(f"\n📊 CSS Parser Accuracy")
        print("-" * 60)
        print(f"Traditional Parser:")
        print(f"  Accuracy: {traditional_metrics['accuracy']:.2%}")
        print(f"  Precision: {traditional_metrics['precision']:.2%}")
        print(f"  Recall: {traditional_metrics['recall']:.2%}")
        print(f"  F1 Score: {traditional_metrics['f1_score']:.3f}")
        
        return results
    
    def measure_js_accuracy(self) -> Dict:
        """Measure JS parser accuracy."""
        test_file = self.data_dir / 'js' / 'test.json'
        if not test_file.exists():
            return {'error': 'Test data not found'}
        
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        results = {}
        ground_truth = [sample['label'] == 'valid' for sample in test_data]
        
        # Traditional baseline
        traditional_preds = [self.traditional_js_parser(sample['input']) for sample in test_data]
        traditional_metrics = self.calculate_confusion_matrix(traditional_preds, ground_truth)
        results['traditional'] = traditional_metrics
        
        print(f"\n📊 JS Parser Accuracy")
        print("-" * 60)
        print(f"Traditional Parser:")
        print(f"  Accuracy: {traditional_metrics['accuracy']:.2%}")
        print(f"  Precision: {traditional_metrics['precision']:.2%}")
        print(f"  Recall: {traditional_metrics['recall']:.2%}")
        print(f"  F1 Score: {traditional_metrics['f1_score']:.3f}")
        
        return results
    
    def run_accuracy_measurements(self):
        """Run comprehensive accuracy measurements."""
        print("\n" + "=" * 60)
        print("BrowerAI Accuracy Measurement Suite")
        print("=" * 60)
        
        # Measure all parsers
        html_results = self.measure_html_accuracy()
        css_results = self.measure_css_accuracy()
        js_results = self.measure_js_accuracy()
        
        self.results['accuracy_metrics'] = {
            'html': html_results,
            'css': css_results,
            'js': js_results
        }
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive accuracy report."""
        print("\n" + "=" * 60)
        print("Accuracy Measurement Report")
        print("=" * 60)
        
        # Calculate overall improvements
        improvements = []
        
        for parser_type, results in self.results['accuracy_metrics'].items():
            if 'error' not in results and 'traditional' in results:
                traditional_acc = results['traditional']['accuracy']
                
                for model_name, metrics in results.items():
                    if model_name != 'traditional':
                        ai_acc = metrics['accuracy']
                        improvement = ((ai_acc - traditional_acc) / traditional_acc * 100)
                        improvements.append({
                            'parser': parser_type,
                            'model': model_name,
                            'traditional_accuracy': traditional_acc,
                            'ai_accuracy': ai_acc,
                            'improvement_percent': improvement
                        })
        
        if improvements:
            print("\n📈 Accuracy Improvements Summary:")
            print("-" * 60)
            
            for imp in improvements:
                print(f"\n{imp['parser'].upper()} - {imp['model']}:")
                print(f"  Traditional: {imp['traditional_accuracy']:.2%}")
                print(f"  AI Model: {imp['ai_accuracy']:.2%}")
                print(f"  Improvement: {imp['improvement_percent']:+.1f}%")
            
            avg_improvement = sum(imp['improvement_percent'] for imp in improvements) / len(improvements)
            print(f"\n📊 Average Accuracy Improvement: {avg_improvement:+.1f}%")
        
        # Save results
        output_path = Path('accuracy_measurement_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
        
        print("\n" + "=" * 60)
        print("Accuracy measurements demonstrate model effectiveness")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Measure BrowerAI model accuracy improvements')
    parser.add_argument('--models-dir', type=str, default='../../models',
                       help='Directory containing ONNX models')
    parser.add_argument('--data-dir', type=str, default='../data',
                       help='Directory containing test data')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir) / 'local'
    data_dir = Path(args.data_dir)
    
    if not ONNX_AVAILABLE:
        print("\n⚠️  onnxruntime required for accuracy measurement")
        print("Install with: pip install onnxruntime")
        return
    
    measurement = AccuracyMeasurement(models_dir, data_dir)
    measurement.load_models()
    
    if not measurement.sessions:
        print("\n⚠️  No models available for testing")
        print("Train models first using training scripts")
        return
    
    measurement.run_accuracy_measurements()
    measurement.generate_report()


if __name__ == '__main__':
    main()
