#!/usr/bin/env python3
"""
Week 6 Integration Test Runner
Tests Python API Server with simulated Rust client requests
"""

import json
import time
import requests
from typing import Dict, List, Any, Tuple
from datetime import datetime
import statistics


class IntegrationTestRunner:
    """Runs comprehensive integration tests against Python API server"""

    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url
        self.feature_dim = 48
        self.latent_dim = 256
        self.test_results: List[Dict[str, Any]] = []
        self.session = requests.Session()
        self.session.timeout = 30

    def run_all_tests(self) -> bool:
        """Run complete integration test suite"""
        print("\n" + "=" * 70)
        print("  BrowserAI Week 6 - Integration Test Runner")
        print("=" * 70)
        print()

        tests = [
            ("Health Check", self.test_health_check),
            ("Feature Generation", self.test_feature_to_code_generation),
            ("Feedback Processing", self.test_feedback_processing),
            ("Learning Loop (3x)", self.test_learning_loop),
            ("Throughput (10 req)", self.test_throughput),
            ("Error Handling", self.test_error_handling),
            ("Latency Consistency", self.test_latency_consistency),
        ]

        results = []
        for name, test_func in tests:
            try:
                result = test_func()
                results.append(result)
                status_symbol = "✓" if result["passed"] else "✗"
                print(f"{status_symbol} {result['name']:40} {result['status']:8} ({result['duration_ms']:.2f}ms)")
            except Exception as e:
                print(f"✗ {name:40} ERROR    (Test crashed)")
                print(f"    Error: {str(e)}")
                results.append({
                    "name": name,
                    "passed": False,
                    "status": "ERROR",
                    "duration_ms": 0.0,
                    "details": {"error": str(e)}
                })

        print("\n" + "=" * 70)
        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        print(f"  Summary: {passed}/{total} tests passed")
        print("=" * 70 + "\n")

        return all(r["passed"] for r in results)

    def test_health_check(self) -> Dict[str, Any]:
        """Test 1: Health check endpoint"""
        test_name = "Health Check - Server Readiness"
        start = time.time()

        try:
            response = self.session.get(f"{self.base_url}/api/v1/health")
            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                passed = (
                    data.get("status") == "healthy" and
                    data.get("models_loaded") == 3 and
                    "uptime_seconds" in data
                )
                details = {
                    "status": data.get("status"),
                    "models_loaded": str(data.get("models_loaded")),
                    "version": data.get("version", "unknown")
                }
            else:
                passed = False
                details = {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            passed = False
            duration = (time.time() - start) * 1000
            details = {"error": str(e)}

        return {
            "name": test_name,
            "passed": passed,
            "status": "PASSED" if passed else "FAILED",
            "duration_ms": duration,
            "details": details
        }

    def test_feature_to_code_generation(self) -> Dict[str, Any]:
        """Test 2: Generate code from features"""
        test_name = "Feature to Code Generation"
        start = time.time()

        try:
            features = self.create_test_features()
            request_data = {
                "url": "https://example.com",
                "features": features,
                "website_intent": "blog",
                "design_style": "modern",
                "session_id": f"test-{int(time.time())}",
                "timestamp": int(time.time())
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/generate",
                json=request_data
            )
            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                passed = (
                    "html" in data and
                    "css" in data and
                    "javascript" in data and
                    "confidence" in data
                )
                details = {
                    "html_length": str(len(data.get("html", ""))),
                    "css_length": str(len(data.get("css", ""))),
                    "js_length": str(len(data.get("javascript", ""))),
                    "confidence": f"{data.get('confidence', 0):.2f}",
                    "should_use": str(data.get("should_use", False))
                }
            else:
                passed = False
                details = {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            passed = False
            duration = (time.time() - start) * 1000
            details = {"error": str(e)}

        return {
            "name": test_name,
            "passed": passed,
            "status": "PASSED" if passed else "FAILED",
            "duration_ms": duration,
            "details": details
        }

    def test_feedback_processing(self) -> Dict[str, Any]:
        """Test 3: Process feedback"""
        test_name = "Feedback Processing"
        start = time.time()

        try:
            feedback_data = {
                "url": "https://example.com",
                "overall_quality": 0.85,
                "html_similarity": 0.88,
                "css_accuracy": 0.82,
                "layout_similarity": 0.85,
                "matched_elements": 45,
                "mismatched_elements": 5,
                "session_id": f"test-{int(time.time())}",
                "timestamp": int(time.time())
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/feedback",
                json=feedback_data
            )
            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                passed = (
                    data.get("status") == "ok" and
                    "learner_metrics" in data
                )
                metrics = data.get("learner_metrics", {})
                details = {
                    "quality_score": f"{data.get('quality_score', 0):.2f}",
                    "buffer_size": str(data.get("buffer_size", 0)),
                    "average_loss": f"{metrics.get('average_loss', 0):.4f}",
                    "convergence": f"{metrics.get('convergence', 0):.2f}",
                    "updates": str(metrics.get("update_count", 0))
                }
            else:
                passed = False
                details = {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            passed = False
            duration = (time.time() - start) * 1000
            details = {"error": str(e)}

        return {
            "name": test_name,
            "passed": passed,
            "status": "PASSED" if passed else "FAILED",
            "duration_ms": duration,
            "details": details
        }

    def test_learning_loop(self) -> Dict[str, Any]:
        """Test 4: Complete learning loop"""
        test_name = "Learning Loop (3 Iterations)"
        start = time.time()

        try:
            iterations_completed = 0
            qualities = []

            for i in range(3):
                # Generate code
                features = self.create_test_features()
                gen_request = {
                    "url": "https://example.com",
                    "features": features,
                    "website_intent": "blog",
                    "design_style": "modern",
                    "session_id": f"loop-{i}",
                    "timestamp": int(time.time())
                }

                gen_response = self.session.post(
                    f"{self.base_url}/api/v1/generate",
                    json=gen_request
                )

                if gen_response.status_code != 200:
                    continue

                # Simulate quality evaluation
                quality = 0.80 + (i * 0.03)
                qualities.append(quality)

                # Send feedback
                feedback = {
                    "url": "https://example.com",
                    "overall_quality": quality,
                    "html_similarity": quality + 0.03,
                    "css_accuracy": quality - 0.01,
                    "layout_similarity": quality,
                    "matched_elements": 40 + i * 2,
                    "mismatched_elements": 5 - i,
                    "session_id": f"loop-{i}",
                    "timestamp": int(time.time())
                }

                fb_response = self.session.post(
                    f"{self.base_url}/api/v1/feedback",
                    json=feedback
                )

                if fb_response.status_code == 200:
                    iterations_completed += 1

            duration = (time.time() - start) * 1000
            passed = iterations_completed == 3

            details = {
                "iterations": f"{iterations_completed}/3",
                "avg_quality": f"{statistics.mean(qualities):.2f}" if qualities else "0.00",
                "quality_trend": "improving" if len(qualities) > 1 and qualities[-1] > qualities[0] else "unknown"
            }
        except Exception as e:
            passed = False
            duration = (time.time() - start) * 1000
            details = {"error": str(e)}

        return {
            "name": test_name,
            "passed": passed,
            "status": "PASSED" if passed else "FAILED",
            "duration_ms": duration,
            "details": details
        }

    def test_throughput(self) -> Dict[str, Any]:
        """Test 5: Throughput (10 concurrent-like requests)"""
        test_name = "Throughput Test (10 Requests)"
        start = time.time()

        try:
            successful = 0
            total = 10
            latencies = []

            for _ in range(total):
                req_start = time.time()
                features = self.create_test_features()
                request_data = {
                    "url": "https://example.com",
                    "features": features,
                    "website_intent": "blog",
                    "design_style": "modern",
                    "session_id": f"throughput-{int(time.time() * 1000)}",
                    "timestamp": int(time.time())
                }

                response = self.session.post(
                    f"{self.base_url}/api/v1/generate",
                    json=request_data
                )

                latencies.append((time.time() - req_start) * 1000)

                if response.status_code == 200:
                    successful += 1

            duration = (time.time() - start) * 1000
            passed = successful == total

            details = {
                "successful": f"{successful}/{total}",
                "avg_latency_ms": f"{statistics.mean(latencies):.2f}",
                "throughput_rps": f"{total / (duration / 1000):.1f}",
                "min_latency_ms": f"{min(latencies):.2f}",
                "max_latency_ms": f"{max(latencies):.2f}"
            }
        except Exception as e:
            passed = False
            duration = (time.time() - start) * 1000
            details = {"error": str(e)}

        return {
            "name": test_name,
            "passed": passed,
            "status": "PASSED" if passed else "FAILED",
            "duration_ms": duration,
            "details": details
        }

    def test_error_handling(self) -> Dict[str, Any]:
        """Test 6: Error handling"""
        test_name = "Error Handling"
        start = time.time()

        try:
            validation_count = 0

            # Test 1: Invalid feature dimension
            bad_features = [0.1] * 47  # Wrong size
            gen_request = {
                "url": "https://example.com",
                "features": bad_features,
                "website_intent": "blog",
                "design_style": "modern",
                "session_id": "error-test-1",
                "timestamp": int(time.time())
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/generate",
                json=gen_request
            )

            if response.status_code == 400:
                validation_count += 1

            # Test 2: Invalid quality score
            bad_feedback = {
                "url": "https://example.com",
                "overall_quality": 1.5,  # Out of range
                "html_similarity": 0.88,
                "css_accuracy": 0.82,
                "layout_similarity": 0.85,
                "matched_elements": 45,
                "mismatched_elements": 5,
                "session_id": "error-test-2",
                "timestamp": int(time.time())
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/feedback",
                json=bad_feedback
            )

            if response.status_code == 400:
                validation_count += 1

            duration = (time.time() - start) * 1000
            passed = validation_count == 2

            details = {
                "validations_triggered": f"{validation_count}/2",
                "feature_validation": "✓" if validation_count >= 1 else "✗",
                "feedback_validation": "✓" if validation_count >= 2 else "✗"
            }
        except Exception as e:
            passed = False
            duration = (time.time() - start) * 1000
            details = {"error": str(e)}

        return {
            "name": test_name,
            "passed": passed,
            "status": "PASSED" if passed else "FAILED",
            "duration_ms": duration,
            "details": details
        }

    def test_latency_consistency(self) -> Dict[str, Any]:
        """Test 7: Latency consistency"""
        test_name = "Latency Consistency (5 Requests)"
        start = time.time()

        try:
            latencies = []

            for _ in range(5):
                req_start = time.time()
                features = self.create_test_features()
                request_data = {
                    "url": "https://example.com",
                    "features": features,
                    "website_intent": "blog",
                    "design_style": "modern",
                    "session_id": f"latency-{int(time.time() * 1000)}",
                    "timestamp": int(time.time())
                }

                self.session.post(
                    f"{self.base_url}/api/v1/generate",
                    json=request_data
                )

                latencies.append((time.time() - req_start) * 1000)

            duration = (time.time() - start) * 1000
            variance = max(latencies) - min(latencies)
            passed = variance < 5.0  # Accept 5ms variance

            details = {
                "avg_latency_ms": f"{statistics.mean(latencies):.2f}",
                "min_latency_ms": f"{min(latencies):.2f}",
                "max_latency_ms": f"{max(latencies):.2f}",
                "variance_ms": f"{variance:.2f}",
                "consistency": "good" if variance < 5 else "inconsistent"
            }
        except Exception as e:
            passed = False
            duration = (time.time() - start) * 1000
            details = {"error": str(e)}

        return {
            "name": test_name,
            "passed": passed,
            "status": "PASSED" if passed else "FAILED",
            "duration_ms": duration,
            "details": details
        }

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def create_test_features(self) -> List[float]:
        """Create 48-dimensional test feature vector"""
        return [0.1 + (i * 0.01) % 0.8 for i in range(self.feature_dim)]

    def create_test_feedback(self) -> Dict[str, Any]:
        """Create test feedback data"""
        return {
            "url": "https://example.com",
            "overall_quality": 0.85,
            "html_similarity": 0.88,
            "css_accuracy": 0.82,
            "layout_similarity": 0.85,
            "matched_elements": 45,
            "mismatched_elements": 5,
            "session_id": f"test-{int(time.time())}",
            "timestamp": int(time.time())
        }


def main():
    """Run integration tests"""
    runner = IntegrationTestRunner()
    success = runner.run_all_tests()

    if success:
        print("✓ All integration tests PASSED!")
        exit(0)
    else:
        print("✗ Some tests FAILED!")
        exit(1)


if __name__ == "__main__":
    main()
