#!/usr/bin/env python3
"""
Week 8 Phase B - Stress Testing Framework

Tests system stability under high load with concurrent requests.
Monitors memory, CPU, and response times.
"""

import time
import json
import argparse
import threading
import queue
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import statistics
import psutil
import numpy as np

from http_client import RealHttpClient, HttpClientConfig


@dataclass
class LoadTestConfig:
    """Configuration for load testing"""
    concurrent_users: int = 10
    requests_per_user: int = 10
    ramp_up_time: float = 0  # seconds to gradually increase load
    test_duration: Optional[float] = None  # Max duration in seconds
    base_url: str = "http://127.0.0.1:5000"
    timeout: float = 10.0


@dataclass
class RequestResult:
    """Result of a single request"""
    timestamp: float
    duration_ms: float
    status_code: int
    success: bool
    error: Optional[str] = None
    thread_id: int = 0


@dataclass
class ResourceSnapshot:
    """System resource snapshot"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    thread_count: int
    connections: int


class ResourceMonitor:
    """Monitor system resources during stress test"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.snapshots: List[ResourceSnapshot] = []
        self.running = False
        self.thread = None
        self.process = psutil.Process()
        
    def start(self):
        """Start monitoring"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            
    def _monitor_loop(self):
        """Monitoring loop"""
        while self.running:
            try:
                snapshot = ResourceSnapshot(
                    timestamp=time.time(),
                    cpu_percent=self.process.cpu_percent(interval=0.1),
                    memory_mb=self.process.memory_info().rss / 1024 / 1024,
                    memory_percent=self.process.memory_percent(),
                    thread_count=self.process.num_threads(),
                    connections=len(self.process.connections())
                )
                self.snapshots.append(snapshot)
            except Exception as e:
                print(f"Resource monitoring error: {e}")
            
            time.sleep(self.interval)
            
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.snapshots:
            return {}
            
        cpu_values = [s.cpu_percent for s in self.snapshots]
        mem_values = [s.memory_mb for s in self.snapshots]
        
        return {
            "duration_seconds": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            "samples": len(self.snapshots),
            "cpu": {
                "mean": statistics.mean(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
                "stdev": statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            },
            "memory_mb": {
                "mean": statistics.mean(mem_values),
                "max": max(mem_values),
                "min": min(mem_values),
                "stdev": statistics.stdev(mem_values) if len(mem_values) > 1 else 0
            },
            "threads": {
                "mean": statistics.mean([s.thread_count for s in self.snapshots]),
                "max": max([s.thread_count for s in self.snapshots])
            },
            "connections": {
                "mean": statistics.mean([s.connections for s in self.snapshots]),
                "max": max([s.connections for s in self.snapshots])
            }
        }


class StressTestWorker:
    """Worker thread for generating load"""
    
    def __init__(self, worker_id: int, http_client: RealHttpClient, 
                 requests: int, results_queue: queue.Queue):
        self.worker_id = worker_id
        self.http_client = http_client
        self.requests = requests
        self.results_queue = results_queue
        
    def run(self):
        """Execute requests"""
        for i in range(self.requests):
            result = self._execute_request()
            self.results_queue.put(result)
            
    def _execute_request(self) -> RequestResult:
        """Execute a single request"""
        start = time.time()
        
        try:
            # Create test data
            features = list(np.random.rand(48))
            request_data = {
                "url": f"https://example-{self.worker_id}.com",
                "features": features,
                "website_intent": "blog",
                "design_style": "modern",
                "session_id": f"stress-{self.worker_id}-{int(time.time() * 1000)}",
                "timestamp": int(time.time())
            }
            
            response = self.http_client.post(
                "/api/v1/generate",
                json=request_data
            )
            
            duration = (time.time() - start) * 1000
            
            return RequestResult(
                timestamp=start,
                duration_ms=duration,
                status_code=response.status_code,
                success=response.status_code == 200,
                thread_id=self.worker_id
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            return RequestResult(
                timestamp=start,
                duration_ms=duration,
                status_code=0,
                success=False,
                error=str(e),
                thread_id=self.worker_id
            )


class StressTestRunner:
    """Main stress test runner"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[RequestResult] = []
        self.resource_monitor = ResourceMonitor()
        
    def run(self) -> Dict[str, Any]:
        """Run stress test"""
        print("\n" + "="*70)
        print("  Week 8 Phase B - Stress Testing")
        print("="*70)
        print(f"\n📊 Configuration:")
        print(f"   Concurrent Users: {self.config.concurrent_users}")
        print(f"   Requests per User: {self.config.requests_per_user}")
        print(f"   Total Requests: {self.config.concurrent_users * self.config.requests_per_user}")
        print(f"   Ramp-up Time: {self.config.ramp_up_time}s")
        print()
        
        # Start resource monitoring
        self.resource_monitor.start()
        
        # Create HTTP client
        http_config = HttpClientConfig(
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )
        
        # Results queue
        results_queue = queue.Queue()
        
        # Create and start worker threads
        threads = []
        start_time = time.time()
        
        print("🚀 Starting stress test...")
        print(f"   Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        for i in range(self.config.concurrent_users):
            http_client = RealHttpClient(http_config)
            worker = StressTestWorker(i, http_client, self.config.requests_per_user, results_queue)
            thread = threading.Thread(target=worker.run)
            threads.append(thread)
            thread.start()
            
            # Ramp-up delay
            if self.config.ramp_up_time > 0:
                time.sleep(self.config.ramp_up_time / self.config.concurrent_users)
        
        # Wait for all threads to complete
        print("⏳ Executing requests...")
        for thread in threads:
            thread.join()
            
        end_time = time.time()
        
        # Stop resource monitoring
        self.resource_monitor.stop()
        
        # Collect results
        while not results_queue.empty():
            self.results.append(results_queue.get())
            
        duration = end_time - start_time
        
        print(f"✅ Stress test completed in {duration:.2f}s")
        print()
        
        # Analyze results
        return self._analyze_results(duration)
        
    def _analyze_results(self, duration: float) -> Dict[str, Any]:
        """Analyze test results"""
        if not self.results:
            return {"error": "No results collected"}
            
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        latencies = [r.duration_ms for r in successful]
        
        analysis = {
            "summary": {
                "total_requests": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(self.results) * 100,
                "duration_seconds": duration,
                "throughput_rps": len(self.results) / duration
            },
            "latency_ms": {},
            "resources": self.resource_monitor.get_summary(),
            "config": asdict(self.config)
        }
        
        if latencies:
            analysis["latency_ms"] = {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99)
            }
            
        # Error analysis
        if failed:
            error_types = {}
            for result in failed:
                error = result.error or "Unknown"
                error_types[error] = error_types.get(error, 0) + 1
            analysis["errors"] = error_types
            
        return analysis
        
    def print_report(self, analysis: Dict[str, Any]):
        """Print test report"""
        print("\n" + "="*70)
        print("  Stress Test Report")
        print("="*70)
        
        summary = analysis["summary"]
        latency = analysis.get("latency_ms", {})
        resources = analysis.get("resources", {})
        
        print(f"\n📊 Summary:")
        print(f"   Total Requests:     {summary['total_requests']}")
        print(f"   Successful:         {summary['successful']} ({summary['success_rate']:.1f}%)")
        print(f"   Failed:             {summary['failed']}")
        print(f"   Duration:           {summary['duration_seconds']:.2f}s")
        print(f"   Throughput:         {summary['throughput_rps']:.2f} RPS")
        
        if latency:
            print(f"\n⚡ Latency (ms):")
            print(f"   Mean:               {latency['mean']:.2f}")
            print(f"   Median:             {latency['median']:.2f}")
            print(f"   Min:                {latency['min']:.2f}")
            print(f"   Max:                {latency['max']:.2f}")
            print(f"   Std Dev:            {latency['stdev']:.2f}")
            print(f"   P95:                {latency['p95']:.2f}")
            print(f"   P99:                {latency['p99']:.2f}")
            
        if resources:
            cpu = resources.get("cpu", {})
            mem = resources.get("memory_mb", {})
            
            print(f"\n💻 Resources:")
            print(f"   CPU Mean:           {cpu.get('mean', 0):.1f}%")
            print(f"   CPU Max:            {cpu.get('max', 0):.1f}%")
            print(f"   Memory Mean:        {mem.get('mean', 0):.1f} MB")
            print(f"   Memory Max:         {mem.get('max', 0):.1f} MB")
            
        if "errors" in analysis:
            print(f"\n❌ Errors:")
            for error, count in analysis["errors"].items():
                print(f"   {error[:50]}: {count}")
                
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Week 8 Phase B - Stress Testing")
    parser.add_argument("--users", type=int, default=10, help="Concurrent users")
    parser.add_argument("--requests", type=int, default=10, help="Requests per user")
    parser.add_argument("--ramp-up", type=float, default=0, help="Ramp-up time in seconds")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout")
    parser.add_argument("--output", type=str, default="stress_test_results.json", help="Output file")
    
    args = parser.parse_args()
    
    config = LoadTestConfig(
        concurrent_users=args.users,
        requests_per_user=args.requests,
        ramp_up_time=args.ramp_up,
        timeout=args.timeout
    )
    
    runner = StressTestRunner(config)
    analysis = runner.run()
    runner.print_report(analysis)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(analysis, f, indent=2)
        
    print(f"\n📁 Results saved to: {args.output}")
    
    # Return exit code based on success
    success_rate = analysis["summary"]["success_rate"]
    if success_rate < 95:
        print(f"\n⚠️  Warning: Success rate ({success_rate:.1f}%) below 95%")
        return 1
    
    print(f"\n✅ Stress test passed! Success rate: {success_rate:.1f}%")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
