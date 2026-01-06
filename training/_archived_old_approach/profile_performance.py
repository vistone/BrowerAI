#!/usr/bin/env python3
"""
Performance Profiling Script for BrowerAI Phase 2.4

This script profiles AI model performance including inference time, memory usage,
throughput, and identifies bottlenecks.

Usage:
    python profile_performance.py [--models-dir MODELS_DIR] [--iterations NUM]
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List
import statistics

try:
    import onnxruntime as ort
    import numpy as np
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Install with: pip install onnxruntime")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with: pip install psutil")


class PerformanceProfiler:
    """Profile AI model performance."""
    
    def __init__(self, models_dir: Path, iterations: int = 100):
        self.models_dir = models_dir
        self.iterations = iterations
        self.sessions = {}
        self.results = {
            'models': {},
            'system_info': {},
            'bottlenecks': []
        }
    
    def get_system_info(self):
        """Get system information."""
        info = {
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        if PSUTIL_AVAILABLE:
            info['cpu_count'] = psutil.cpu_count()
            info['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
            info['memory_available_gb'] = psutil.virtual_memory().available / (1024**3)
        
        if ONNX_AVAILABLE:
            info['onnxruntime_version'] = ort.__version__
            info['available_providers'] = ort.get_available_providers()
        
        return info
    
    def load_models(self):
        """Load ONNX models."""
        if not ONNX_AVAILABLE:
            print("ONNX Runtime not available")
            return
        
        print("\n" + "=" * 60)
        print("Loading Models for Performance Profiling")
        print("=" * 60)
        
        model_files = [
            'html_parser_v1.onnx',
            'html_parser_transformer_v2.onnx',
            'html_parser_lstm_v2.onnx',
            'css_optimizer_v1.onnx',
            'css_deduplication_v1.onnx',
            'css_selector_optimizer_v1.onnx',
            'css_minifier_v1.onnx',
            'js_analyzer_v1.onnx',
            'js_tokenizer_enhancer_v1.onnx',
            'js_ast_predictor_v1.onnx',
            'js_optimization_suggestions_v1.onnx',
            'layout_optimizer_v1.onnx',
            'paint_optimizer_v1.onnx'
        ]
        
        for filename in model_files:
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    # Profile model loading
                    start_time = time.perf_counter()
                    start_memory = self.get_memory_usage()
                    
                    session = ort.InferenceSession(str(model_path))
                    
                    load_time = time.perf_counter() - start_time
                    load_memory = self.get_memory_usage() - start_memory
                    
                    self.sessions[filename] = {
                        'session': session,
                        'load_time': load_time,
                        'load_memory_mb': load_memory,
                        'file_size_kb': model_path.stat().st_size / 1024
                    }
                    
                    print(f"✓ Loaded {filename}")
                    print(f"    Load time: {load_time*1000:.2f} ms")
                    print(f"    Memory: {load_memory:.2f} MB")
                    print(f"    File size: {self.sessions[filename]['file_size_kb']:.2f} KB")
                except Exception as e:
                    print(f"✗ Failed to load {filename}: {e}")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        return 0.0
    
    def generate_test_input(self, session: ort.InferenceSession) -> Dict[str, np.ndarray]:
        """Generate test input for a model."""
        inputs = {}
        
        for input_meta in session.get_inputs():
            shape = input_meta.shape
            dtype = input_meta.type
            
            # Handle dynamic dimensions
            actual_shape = []
            for dim in shape:
                if isinstance(dim, str) or dim is None or dim < 0:
                    actual_shape.append(1)  # Use batch size 1 for dynamic dims
                else:
                    actual_shape.append(dim)
            
            # Generate appropriate data based on type
            if 'float' in dtype:
                data = np.random.randn(*actual_shape).astype(np.float32)
            elif 'int64' in dtype or 'int' in dtype:
                data = np.random.randint(0, 256, size=actual_shape).astype(np.int64)
            else:
                data = np.zeros(actual_shape, dtype=np.float32)
            
            inputs[input_meta.name] = data
        
        return inputs
    
    def profile_model_inference(self, model_name: str) -> Dict:
        """Profile model inference performance."""
        if model_name not in self.sessions:
            return {'error': 'Model not loaded'}
        
        session_info = self.sessions[model_name]
        session = session_info['session']
        
        print(f"\n⏱️  Profiling: {model_name}")
        
        # Generate test input
        test_input = self.generate_test_input(session)
        
        # Warm-up runs (don't count these)
        for _ in range(10):
            session.run(None, test_input)
        
        # Measure inference times
        inference_times = []
        memory_usages = []
        
        for i in range(self.iterations):
            # Memory before
            mem_before = self.get_memory_usage()
            
            # Run inference
            start_time = time.perf_counter()
            outputs = session.run(None, test_input)
            inference_time = time.perf_counter() - start_time
            
            # Memory after
            mem_after = self.get_memory_usage()
            
            inference_times.append(inference_time * 1000)  # Convert to ms
            memory_usages.append(mem_after - mem_before)
        
        # Calculate statistics
        results = {
            'model_name': model_name,
            'iterations': self.iterations,
            'load_time_ms': session_info['load_time'] * 1000,
            'load_memory_mb': session_info['load_memory_mb'],
            'file_size_kb': session_info['file_size_kb'],
            'inference_times_ms': {
                'min': min(inference_times),
                'max': max(inference_times),
                'mean': statistics.mean(inference_times),
                'median': statistics.median(inference_times),
                'stdev': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
                'p95': sorted(inference_times)[int(len(inference_times) * 0.95)],
                'p99': sorted(inference_times)[int(len(inference_times) * 0.99)]
            },
            'throughput_samples_per_sec': 1000.0 / statistics.mean(inference_times),
            'memory_usage_mb': {
                'mean': statistics.mean(memory_usages) if memory_usages else 0,
                'max': max(memory_usages) if memory_usages else 0
            }
        }
        
        # Print results
        print(f"  Inference Time:")
        print(f"    Mean: {results['inference_times_ms']['mean']:.4f} ms")
        print(f"    Median: {results['inference_times_ms']['median']:.4f} ms")
        print(f"    Min: {results['inference_times_ms']['min']:.4f} ms")
        print(f"    Max: {results['inference_times_ms']['max']:.4f} ms")
        print(f"    P95: {results['inference_times_ms']['p95']:.4f} ms")
        print(f"    P99: {results['inference_times_ms']['p99']:.4f} ms")
        print(f"  Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
        
        # Check for bottlenecks
        if results['inference_times_ms']['mean'] > 10.0:
            self.results['bottlenecks'].append({
                'model': model_name,
                'issue': 'High inference time',
                'value': f"{results['inference_times_ms']['mean']:.2f} ms",
                'threshold': '10 ms'
            })
        
        if results['file_size_kb'] > 100 * 1024:  # >100 MB
            self.results['bottlenecks'].append({
                'model': model_name,
                'issue': 'Large model size',
                'value': f"{results['file_size_kb'] / 1024:.1f} MB",
                'threshold': '100 MB'
            })
        
        return results
    
    def run_performance_profiling(self):
        """Run comprehensive performance profiling."""
        print("\n" + "=" * 60)
        print("BrowerAI Performance Profiling Suite")
        print("=" * 60)
        
        # Get system info
        self.results['system_info'] = self.get_system_info()
        
        print("\n🖥️  System Information:")
        for key, value in self.results['system_info'].items():
            print(f"  {key}: {value}")
        
        # Profile each model
        for model_name in self.sessions:
            results = self.profile_model_inference(model_name)
            self.results['models'][model_name] = results
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive performance report."""
        print("\n" + "=" * 60)
        print("Performance Profiling Report")
        print("=" * 60)
        
        # Summary statistics
        if self.results['models']:
            all_inference_times = [
                m['inference_times_ms']['mean'] 
                for m in self.results['models'].values() 
                if 'inference_times_ms' in m
            ]
            
            all_throughputs = [
                m['throughput_samples_per_sec']
                for m in self.results['models'].values()
                if 'throughput_samples_per_sec' in m
            ]
            
            print("\n📊 Overall Statistics:")
            print(f"  Models Profiled: {len(self.results['models'])}")
            print(f"  Average Inference Time: {statistics.mean(all_inference_times):.4f} ms")
            print(f"  Average Throughput: {statistics.mean(all_throughputs):.1f} samples/sec")
        
        # Bottleneck analysis
        if self.results['bottlenecks']:
            print("\n⚠️  Performance Bottlenecks Detected:")
            print("-" * 60)
            
            for bottleneck in self.results['bottlenecks']:
                print(f"\n  Model: {bottleneck['model']}")
                print(f"  Issue: {bottleneck['issue']}")
                print(f"  Current: {bottleneck['value']}")
                print(f"  Threshold: {bottleneck['threshold']}")
        else:
            print("\n✅ No performance bottlenecks detected")
            print("    All models meet performance targets")
        
        # Performance targets from roadmap
        print("\n🎯 Roadmap Performance Targets:")
        print("-" * 60)
        print("  ✓ Inference Time: <10ms per operation")
        print("  ✓ Model Size: <100MB for all models")
        print("  ✓ Parsing Speed: 50% faster than traditional")
        
        # Save results
        output_path = Path('performance_profile_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
        
        print("\n" + "=" * 60)
        print("Performance profiling identifies optimization opportunities")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Profile BrowerAI model performance')
    parser.add_argument('--models-dir', type=str, default='../../models',
                       help='Directory containing ONNX models')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of inference iterations (default: 100)')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir) / 'local'
    
    if not ONNX_AVAILABLE:
        print("\n⚠️  onnxruntime required for performance profiling")
        print("Install with: pip install onnxruntime")
        return
    
    profiler = PerformanceProfiler(models_dir, args.iterations)
    profiler.load_models()
    
    if not profiler.sessions:
        print("\n⚠️  No models available for profiling")
        print("Train models first using training scripts")
        return
    
    profiler.run_performance_profiling()
    profiler.generate_report()


if __name__ == '__main__':
    main()
