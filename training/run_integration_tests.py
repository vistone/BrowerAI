#!/usr/bin/env python3
"""
Week 6 Integration Test Execution Script
Orchestrates Python API server and runs comprehensive test suite
"""

import subprocess
import time
import sys
import os
import signal
import requests
from pathlib import Path


class IntegrationTestOrchestrator:
    """Orchestrates test execution with API server"""

    def __init__(self, workspace_root: str = "/home/stone/BrowerAI"):
        self.workspace_root = workspace_root
        self.api_process = None
        self.api_url = "http://127.0.0.1:5000"
        self.max_startup_time = 30

    def check_python_dependencies(self) -> bool:
        """Check if required Python packages are installed"""
        print("🔍 Checking Python dependencies...")
        
        required_packages = [
            ("flask", "Flask"),
            ("numpy", "NumPy"),
            ("requests", "Requests"),
            ("psutil", "psutil"),
            ("pydantic", "Pydantic")
        ]

        all_ok = True
        for package, name in required_packages:
            try:
                __import__(package)
                print(f"  ✓ {name}")
            except ImportError:
                print(f"  ✗ {name} (not installed)")
                all_ok = False

        return all_ok

    def start_api_server(self) -> bool:
        """Start Python API server"""
        print("\n🚀 Starting Python API Server...")
        
        api_script = Path(self.workspace_root) / "training" / "api_server.py"
        if not api_script.exists():
            print(f"✗ API server script not found: {api_script}")
            return False

        try:
            self.api_process = subprocess.Popen(
                [sys.executable, str(api_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(self.workspace_root) / "training")
            )
            print(f"  PID: {self.api_process.pid}")

            # Wait for server to start
            print("  Waiting for server to be ready...", end="", flush=True)
            start_time = time.time()
            
            while time.time() - start_time < self.max_startup_time:
                try:
                    response = requests.get(f"{self.api_url}/api/v1/health", timeout=2)
                    if response.status_code == 200:
                        print(" ✓")
                        print(f"  Server ready at {self.api_url}")
                        return True
                except:
                    print(".", end="", flush=True)
                    time.sleep(1)

            print(" ✗")
            print("✗ Server failed to start within timeout")
            return False

        except Exception as e:
            print(f"✗ Failed to start API server: {str(e)}")
            return False

    def stop_api_server(self) -> None:
        """Stop API server"""
        if self.api_process:
            print("\n🛑 Stopping API Server...")
            try:
                self.api_process.terminate()
                self.api_process.wait(timeout=5)
                print("  ✓ Server stopped")
            except subprocess.TimeoutExpired:
                print("  ⚠ Force killing server...")
                self.api_process.kill()
                self.api_process.wait()

    def run_integration_tests(self) -> bool:
        """Run Python integration tests"""
        print("\n🧪 Running Integration Tests...\n")
        
        test_script = Path(self.workspace_root) / "training" / "integration_test_runner.py"
        if not test_script.exists():
            print(f"✗ Test script not found: {test_script}")
            return False

        try:
            result = subprocess.run(
                [sys.executable, str(test_script)],
                cwd=str(Path(self.workspace_root) / "training"),
                capture_output=False
            )
            return result.returncode == 0

        except Exception as e:
            print(f"✗ Failed to run tests: {str(e)}")
            return False

    def run_performance_benchmark(self) -> bool:
        """Run performance benchmark"""
        print("\n📊 Running Performance Benchmark...\n")
        
        monitor_script = Path(self.workspace_root) / "training" / "performance_monitor.py"
        if not monitor_script.exists():
            print(f"✗ Monitor script not found: {monitor_script}")
            return False

        try:
            result = subprocess.run(
                [sys.executable, str(monitor_script)],
                cwd=str(Path(self.workspace_root) / "training"),
                capture_output=False
            )
            return result.returncode == 0

        except Exception as e:
            print(f"✗ Failed to run benchmark: {str(e)}")
            return False

    def run_full_test_suite(self) -> bool:
        """Run complete test suite"""
        print("\n" + "=" * 70)
        print("  BrowserAI Week 6 - Full Integration Test Suite")
        print("=" * 70)

        # Check dependencies
        if not self.check_python_dependencies():
            print("\n✗ Missing dependencies. Install with:")
            print("  pip install flask numpy requests psutil pydantic")
            return False

        # Start server
        if not self.start_api_server():
            return False

        try:
            # Run integration tests
            tests_passed = self.run_integration_tests()

            # Run performance benchmark
            benchmark_passed = self.run_performance_benchmark()

            # Summary
            print("\n" + "=" * 70)
            print("  Test Suite Summary")
            print("=" * 70)
            print(f"Integration Tests: {'✓ PASSED' if tests_passed else '✗ FAILED'}")
            print(f"Performance Benchmark: {'✓ PASSED' if benchmark_passed else '✗ FAILED'}")
            print("=" * 70 + "\n")

            return tests_passed and benchmark_passed

        finally:
            # Always stop server
            self.stop_api_server()


def main():
    """Main entry point"""
    orchestrator = IntegrationTestOrchestrator()
    
    try:
        success = orchestrator.run_full_test_suite()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        orchestrator.stop_api_server()
        sys.exit(1)


if __name__ == "__main__":
    main()
