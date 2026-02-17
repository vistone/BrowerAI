#!/usr/bin/env python3
"""
Performance Monitoring and Metrics Collection
Tracks latency, throughput, and resource utilization
"""

import json
import time
import statistics
import requests
from typing import Dict, List, Any
from datetime import datetime
import subprocess
import psutil
import os


class PerformanceMonitor:
    """Monitor API performance metrics"""

    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "endpoints": {},
            "overall": {},
            "resources": {},
            "test_summary": {}
        }

    def benchmark_endpoint(self, endpoint: str, method: str, data: Dict[str, Any], 
                          num_requests: int = 20) -> Dict[str, Any]:
        """Benchmark a specific endpoint"""
        print(f"\n📊 Benchmarking: {method.upper()} {endpoint}")
        print(f"   Requests: {num_requests}")

        latencies = []
        success_count = 0
        error_count = 0

        for i in range(num_requests):
            start = time.time()
            try:
                if method.upper() == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=30)
                else:
                    response = requests.post(f"{self.base_url}{endpoint}", json=data, timeout=30)
                
                latency = (time.time() - start) * 1000
                latencies.append(latency)

                if 200 <= response.status_code < 300:
                    success_count += 1
                else:
                    error_count += 1

                if (i + 1) % 5 == 0:
                    print(f"   Progress: {i + 1}/{num_requests} requests")

            except Exception as e:
                error_count += 1
                print(f"   ✗ Request {i + 1} failed: {str(e)}")

        # Calculate statistics
        if latencies:
            results = {
                "endpoint": endpoint,
                "method": method.upper(),
                "total_requests": num_requests,
                "successful": success_count,
                "failed": error_count,
                "success_rate": f"{(success_count / num_requests * 100):.1f}%",
                "latency_ms": {
                    "mean": f"{statistics.mean(latencies):.2f}",
                    "median": f"{statistics.median(latencies):.2f}",
                    "min": f"{min(latencies):.2f}",
                    "max": f"{max(latencies):.2f}",
                    "stdev": f"{statistics.stdev(latencies):.2f}" if len(latencies) > 1 else "0.00",
                    "p95": f"{sorted(latencies)[int(len(latencies) * 0.95)]:.2f}",
                    "p99": f"{sorted(latencies)[int(len(latencies) * 0.99)]:.2f}"
                },
                "throughput_rps": f"{num_requests / (sum(latencies) / 1000):.1f}"
            }
        else:
            results = {
                "endpoint": endpoint,
                "method": method.upper(),
                "total_requests": num_requests,
                "successful": 0,
                "failed": error_count,
                "success_rate": "0%",
                "latency_ms": {},
                "throughput_rps": "0"
            }

        return results

    def run_comprehensive_benchmark(self) -> None:
        """Run comprehensive performance benchmarks"""
        print("\n" + "=" * 70)
        print("  BrowserAI Week 6 - Performance Benchmark")
        print("=" * 70)

        # Test features
        test_features = [0.1 + (i * 0.01) % 0.8 for i in range(48)]

        # Benchmark 1: Health check (lightweight)
        health_results = self.benchmark_endpoint(
            "/api/v1/health",
            "GET",
            {},
            num_requests=50
        )
        self.metrics["endpoints"]["health"] = health_results

        # Benchmark 2: Code generation (compute-heavy)
        gen_data = {
            "url": "https://example.com",
            "features": test_features,
            "website_intent": "blog",
            "design_style": "modern",
            "session_id": "benchmark",
            "timestamp": int(time.time())
        }
        gen_results = self.benchmark_endpoint(
            "/api/v1/generate",
            "POST",
            gen_data,
            num_requests=20
        )
        self.metrics["endpoints"]["generate"] = gen_results

        # Benchmark 3: Feedback (light)
        feedback_data = {
            "url": "https://example.com",
            "overall_quality": 0.85,
            "html_similarity": 0.88,
            "css_accuracy": 0.82,
            "layout_similarity": 0.85,
            "matched_elements": 45,
            "mismatched_elements": 5,
            "session_id": "benchmark",
            "timestamp": int(time.time())
        }
        feedback_results = self.benchmark_endpoint(
            "/api/v1/feedback",
            "POST",
            feedback_data,
            num_requests=30
        )
        self.metrics["endpoints"]["feedback"] = feedback_results

        # Calculate overall statistics
        all_latencies = []
        for endpoint_data in self.metrics["endpoints"].values():
            if "latency_ms" in endpoint_data and endpoint_data["latency_ms"]:
                try:
                    all_latencies.append(float(endpoint_data["latency_ms"]["mean"]))
                except:
                    pass

        if all_latencies:
            self.metrics["overall"]["avg_latency_ms"] = f"{statistics.mean(all_latencies):.2f}"
            self.metrics["overall"]["max_latency_ms"] = f"{max(all_latencies):.2f}"

        # Collect resource metrics
        self.collect_resource_metrics()

        # Print report
        self.print_report()

        # Save report
        self.save_report()

    def collect_resource_metrics(self) -> None:
        """Collect system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics["resources"]["cpu_percent"] = f"{cpu_percent:.1f}%"

            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics["resources"]["memory_percent"] = f"{memory.percent:.1f}%"
            self.metrics["resources"]["memory_used_mb"] = f"{memory.used / 1024 / 1024:.1f}"
            self.metrics["resources"]["memory_available_mb"] = f"{memory.available / 1024 / 1024:.1f}"

            # Check Python process memory
            try:
                result = subprocess.run(
                    ["pgrep", "-f", "api_server.py"],
                    capture_output=True,
                    text=True
                )
                if result.stdout.strip():
                    pid = int(result.stdout.strip().split()[0])
                    process = psutil.Process(pid)
                    memory_info = process.memory_info()
                    self.metrics["resources"]["api_server_memory_mb"] = f"{memory_info.rss / 1024 / 1024:.1f}"
            except:
                pass

        except Exception as e:
            self.metrics["resources"]["error"] = str(e)

    def print_report(self) -> None:
        """Print performance report"""
        print("\n" + "=" * 70)
        print("  Performance Report")
        print("=" * 70)

        for endpoint_name, endpoint_data in self.metrics["endpoints"].items():
            print(f"\n📍 {endpoint_data['method']} {endpoint_data['endpoint']}")
            print(f"   Success Rate: {endpoint_data['success_rate']}")
            
            if endpoint_data["latency_ms"]:
                print(f"   Latency (ms):")
                print(f"     - Mean:   {endpoint_data['latency_ms']['mean']}")
                print(f"     - Median: {endpoint_data['latency_ms']['median']}")
                print(f"     - Min:    {endpoint_data['latency_ms']['min']}")
                print(f"     - Max:    {endpoint_data['latency_ms']['max']}")
                print(f"     - P95:    {endpoint_data['latency_ms']['p95']}")
                print(f"     - P99:    {endpoint_data['latency_ms']['p99']}")
                print(f"   Throughput: {endpoint_data['throughput_rps']} req/s")

        print("\n📊 Overall Metrics")
        if "avg_latency_ms" in self.metrics["overall"]:
            print(f"   Average Latency: {self.metrics['overall']['avg_latency_ms']} ms")
        if "max_latency_ms" in self.metrics["overall"]:
            print(f"   Max Latency:     {self.metrics['overall']['max_latency_ms']} ms")

        print("\n💾 System Resources")
        print(f"   CPU Usage: {self.metrics['resources'].get('cpu_percent', 'N/A')}")
        print(f"   Memory Usage: {self.metrics['resources'].get('memory_percent', 'N/A')}")
        if "api_server_memory_mb" in self.metrics["resources"]:
            print(f"   API Server Memory: {self.metrics['resources']['api_server_memory_mb']} MB")

        print("\n" + "=" * 70 + "\n")

    def save_report(self) -> None:
        """Save report to JSON file"""
        report_path = "/home/stone/BrowerAI/data/week6_performance_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"✓ Report saved to: {report_path}")


def main():
    """Run performance monitoring"""
    monitor = PerformanceMonitor()
    monitor.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()
