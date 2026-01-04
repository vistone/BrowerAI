#!/usr/bin/env python3
"""
AI Model Benchmarking Suite for BrowerAI Phase 2.4.

This script benchmarks AI models against traditional parsing methods and
measures accuracy improvements and performance impact.

Usage:
    python benchmark_models.py [--models-dir MODELS_DIR]
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


class ModelBenchmark:
    """Benchmark AI models for parsing tasks."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.sessions = {}
        self.results = {
            'html': {'accuracy': [], 'inference_time': [], 'throughput': []},
            'css': {'accuracy': [], 'inference_time': [], 'throughput': []},
            'js': {'accuracy': [], 'inference_time': [], 'throughput': []}
        }
    
    def load_models(self):
        """Load available ONNX models."""
        if not ONNX_AVAILABLE:
            print("ONNX Runtime not available, skipping model loading")
            return
        
        print("\n" + "=" * 60)
        print("Loading AI Models")
        print("=" * 60)
        
        model_files = {
            'html': ['html_parser_v1.onnx', 'html_parser_transformer_v2.onnx', 'html_parser_lstm_v2.onnx'],
            'css': ['css_optimizer_v1.onnx'],
            'js': ['js_analyzer_v1.onnx']
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
                else:
                    print(f"- {filename} not found")
    
    def tokenize(self, text: str, max_length: int = 512) -> np.ndarray:
        """Simple character-level tokenization."""
        tokens = []
        for ch in text[:max_length]:
            tokens.append(ord(ch) % 256)
        
        # Pad
        while len(tokens) < max_length:
            tokens.append(0)
        
        return np.array([tokens], dtype=np.int64)
    
    def benchmark_html_model(self, test_data: List[Dict], model_name: str) -> Dict:
        """Benchmark HTML parsing model."""
        if model_name not in self.sessions:
            return {'error': 'Model not loaded'}
        
        session = self.sessions[model_name]
        
        correct = 0
        total = len(test_data)
        inference_times = []
        
        for sample in test_data:
            # Prepare input
            input_data = self.tokenize(sample['input'])
            
            # Measure inference time
            start_time = time.perf_counter()
            outputs = session.run(None, {'input': input_data})
            inference_time = time.perf_counter() - start_time
            inference_times.append(inference_time)
            
            # Check prediction
            prediction = outputs[0][0][0] > 0.5
            expected = sample['label'] == 'valid'
            
            if prediction == expected:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        avg_inference_time = statistics.mean(inference_times) if inference_times else 0
        throughput = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'accuracy': accuracy,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'throughput_samples_per_sec': throughput,
            'total_samples': total
        }
    
    def benchmark_traditional_parsing(self, test_data: List[Dict], data_type: str) -> Dict:
        """Benchmark traditional parsing methods (baseline)."""
        correct = 0
        total = len(test_data)
        inference_times = []
        
        for sample in test_data:
            start_time = time.perf_counter()
            
            # Simple heuristic validation (baseline)
            text = sample['input']
            if data_type == 'html':
                valid = '<html>' in text.lower() or '<body>' in text.lower()
            elif data_type == 'css':
                valid = '{' in text and '}' in text
            else:  # js
                valid = 'function' in text or 'const' in text or 'var' in text
            
            inference_time = time.perf_counter() - start_time
            inference_times.append(inference_time)
            
            expected = sample['label'] == 'valid'
            if valid == expected:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        avg_inference_time = statistics.mean(inference_times) if inference_times else 0
        throughput = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'accuracy': accuracy,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'throughput_samples_per_sec': throughput,
            'total_samples': total
        }
    
    def run_comprehensive_benchmark(self, data_dir: Path):
        """Run complete benchmark suite."""
        print("\n" + "=" * 60)
        print("BrowerAI AI Model Benchmark Suite")
        print("=" * 60)
        
        results_summary = {}
        
        # Benchmark HTML models
        html_test_path = data_dir / 'html' / 'test.json'
        if html_test_path.exists():
            with open(html_test_path, 'r') as f:
                html_test_data = json.load(f)
            
            print(f"\n📊 HTML Model Benchmark ({len(html_test_data)} samples)")
            print("-" * 60)
            
            # Traditional baseline
            traditional_results = self.benchmark_traditional_parsing(html_test_data, 'html')
            print(f"\nTraditional Parsing (Baseline):")
            print(f"  Accuracy: {traditional_results['accuracy']:.2%}")
            print(f"  Avg Time: {traditional_results['avg_inference_time_ms']:.4f} ms")
            print(f"  Throughput: {traditional_results['throughput_samples_per_sec']:.1f} samples/sec")
            
            results_summary['html_traditional'] = traditional_results
            
            # AI models
            for model_name in self.sessions:
                if 'html' in model_name:
                    print(f"\n{model_name}:")
                    ai_results = self.benchmark_html_model(html_test_data, model_name)
                    
                    if 'error' not in ai_results:
                        print(f"  Accuracy: {ai_results['accuracy']:.2%}")
                        print(f"  Avg Time: {ai_results['avg_inference_time_ms']:.4f} ms")
                        print(f"  Throughput: {ai_results['throughput_samples_per_sec']:.1f} samples/sec")
                        
                        # Calculate improvement
                        acc_improvement = (ai_results['accuracy'] - traditional_results['accuracy']) / traditional_results['accuracy'] * 100
                        print(f"  Accuracy Improvement: {acc_improvement:+.1f}%")
                        
                        results_summary[f'html_{model_name}'] = ai_results
        
        # Save results
        results_path = Path('benchmark_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")
        
        return results_summary
    
    def generate_report(self, results: Dict):
        """Generate a comprehensive benchmark report."""
        print("\n" + "=" * 60)
        print("Benchmark Summary Report")
        print("=" * 60)
        
        # Compare all models
        for key, result in results.items():
            if 'error' not in result:
                print(f"\n{key}:")
                print(f"  Accuracy: {result.get('accuracy', 0):.2%}")
                print(f"  Inference Time: {result.get('avg_inference_time_ms', 0):.4f} ms")
                print(f"  Throughput: {result.get('throughput_samples_per_sec', 0):.1f} samples/sec")
        
        print("\n" + "=" * 60)
        print("Performance Metrics")
        print("=" * 60)
        
        # Calculate statistics
        accuracies = [r['accuracy'] for r in results.values() if 'accuracy' in r]
        times = [r['avg_inference_time_ms'] for r in results.values() if 'avg_inference_time_ms' in r]
        
        if accuracies:
            print(f"\nAccuracy Range: {min(accuracies):.2%} - {max(accuracies):.2%}")
            print(f"Average Accuracy: {statistics.mean(accuracies):.2%}")
        
        if times:
            print(f"\nInference Time Range: {min(times):.4f} - {max(times):.4f} ms")
            print(f"Average Inference Time: {statistics.mean(times):.4f} ms")


def main():
    parser = argparse.ArgumentParser(description='Benchmark BrowerAI AI models')
    parser.add_argument('--models-dir', type=str, default='../models',
                       help='Directory containing ONNX models')
    parser.add_argument('--data-dir', type=str, default='../data',
                       help='Directory containing test data')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir) / 'local'
    data_dir = Path(args.data_dir)
    
    benchmark = ModelBenchmark(models_dir)
    benchmark.load_models()
    
    if ONNX_AVAILABLE and benchmark.sessions:
        results = benchmark.run_comprehensive_benchmark(data_dir)
        benchmark.generate_report(results)
    else:
        print("\n⚠️  No models available for benchmarking")
        print("Next steps:")
        print("  1. Train models: python scripts/train_html_parser.py")
        print("  2. Copy models to ../models/local/")
        print("  3. Run benchmark again")


if __name__ == '__main__':
    main()
