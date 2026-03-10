#!/usr/bin/env python3
"""
Test suite for enhanced API server with security features
Validates authentication, rate limiting, audit logging, and error handling
"""

import unittest
import json
import time
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import after setting path
sys.path.insert(0, str(Path(__file__).parent))
from api_server_enhanced import (
    BrowserAIServer, Config, RateLimiter, AuditLogger,
    FeaturePacketRequest, FeedbackPacketRequest
)


class TestRateLimiter(unittest.TestCase):
    """Test rate limiting functionality"""
    
    def setUp(self):
        self.limiter = RateLimiter(max_requests=3, window_seconds=1)
    
    def test_rate_limit_allow(self):
        """Test that requests within limit are allowed"""
        ip = "192.168.1.1"
        
        self.assertTrue(self.limiter.is_allowed(ip))
        self.assertTrue(self.limiter.is_allowed(ip))
        self.assertTrue(self.limiter.is_allowed(ip))
    
    def test_rate_limit_exceed(self):
        """Test that requests exceeding limit are blocked"""
        ip = "192.168.1.1"
        
        # Make 3 allowed requests
        for _ in range(3):
            self.assertTrue(self.limiter.is_allowed(ip))
        
        # 4th request should be blocked
        self.assertFalse(self.limiter.is_allowed(ip))
    
    def test_rate_limit_reset(self):
        """Test that rate limit resets after window"""
        ip = "192.168.1.1"
        
        # Fill limit
        for _ in range(3):
            self.assertTrue(self.limiter.is_allowed(ip))
        
        # Should be blocked
        self.assertFalse(self.limiter.is_allowed(ip))
        
        # Wait for window to pass
        time.sleep(1.1)
        
        # Should be allowed again
        self.assertTrue(self.limiter.is_allowed(ip))
    
    def test_rate_limit_different_ips(self):
        """Test that rate limiting is per-IP"""
        ip1 = "192.168.1.1"
        ip2 = "192.168.1.2"
        
        # Fill limit for IP1
        for _ in range(3):
            self.assertTrue(self.limiter.is_allowed(ip1))
        
        # IP1 should be blocked
        self.assertFalse(self.limiter.is_allowed(ip1))
        
        # IP2 should still work
        self.assertTrue(self.limiter.is_allowed(ip2))


class TestAuditLogger(unittest.TestCase):
    """Test audit logging"""
    
    def setUp(self):
        self.audit_log = AuditLogger(log_file="/tmp/test_audit.log")
    
    def test_audit_log_creation(self):
        """Test that audit logger creates log file"""
        import os
        self.audit_log.log_request(
            "POST", "/api/v1/generate", "192.168.1.1",
            user_id="user123", status_code=200
        )
        
        # Check log file exists
        self.assertTrue(os.path.exists("/tmp/test_audit.log"))
    
    def test_audit_log_format(self):
        """Test audit log format"""
        self.audit_log.log_request(
            "POST", "/api/v1/feedback", "192.168.1.1",
            user_id="user456", status_code=200,
            details={"quality": 0.85}
        )
        
        # Read log file
        with open("/tmp/test_audit.log", "r") as f:
            content = f.read()
            self.assertIn("POST", content)
            self.assertIn("192.168.1.1", content)
            self.assertIn("200", content)


class TestDataValidation(unittest.TestCase):
    """Test request data validation"""
    
    def test_feature_packet_valid(self):
        """Test valid feature packet"""
        data = {
            "url": "https://example.com",
            "features": [0.1] * 48,
            "website_intent": "blog",
            "design_style": "modern",
            "timestamp": 1704067200,
            "session_id": "sess-123"
        }
        
        packet = FeaturePacketRequest(**data)
        self.assertEqual(len(packet.features), 48)
        self.assertEqual(packet.url, "https://example.com")
    
    def test_feature_packet_invalid_url(self):
        """Test invalid URL format"""
        data = {
            "url": "not-a-url",
            "features": [0.1] * 48,
            "website_intent": "blog",
            "design_style": "modern",
            "timestamp": 1704067200,
            "session_id": "sess-123"
        }
        
        with self.assertRaises(ValueError):
            FeaturePacketRequest(**data)
    
    def test_feature_packet_invalid_dimensions(self):
        """Test invalid feature dimensions"""
        data = {
            "url": "https://example.com",
            "features": [0.1] * 50,  # Wrong size
            "website_intent": "blog",
            "design_style": "modern",
            "timestamp": 1704067200,
            "session_id": "sess-123"
        }
        
        with self.assertRaises(ValueError):
            FeaturePacketRequest(**data)
    
    def test_feedback_packet_valid(self):
        """Test valid feedback packet"""
        data = {
            "url": "https://example.com",
            "overall_quality": 0.8,
            "html_similarity": 0.85,
            "css_accuracy": 0.75,
            "layout_similarity": 0.80,
            "matched_elements": 50,
            "mismatched_elements": 5,
            "session_id": "sess-123",
            "timestamp": 1704067200
        }
        
        feedback = FeedbackPacketRequest(**data)
        self.assertEqual(feedback.overall_quality, 0.8)
    
    def test_feedback_packet_invalid_quality(self):
        """Test invalid quality score"""
        data = {
            "url": "https://example.com",
            "overall_quality": 1.5,  # Out of range
            "html_similarity": 0.85,
            "css_accuracy": 0.75,
            "layout_similarity": 0.80,
            "matched_elements": 50,
            "mismatched_elements": 5,
            "session_id": "sess-123",
            "timestamp": 1704067200
        }
        
        with self.assertRaises(ValueError):
            FeedbackPacketRequest(**data)


class TestAPISecurityFeatures(unittest.TestCase):
    """Test API security features"""
    
    def setUp(self):
        self.config = Config()
        self.config.ENABLE_JWT_AUTH = False  # Disable for testing
        self.config.ENABLE_RATE_LIMITING = True
        self.config.ENABLE_AUDIT_LOG = True
        self.server = BrowserAIServer(self.config)
    
    def test_server_initialization(self):
        """Test server initializes with security components"""
        self.assertIsNotNone(self.server.rate_limiter)
        self.assertIsNotNone(self.server.audit_logger)
        self.assertEqual(self.server.requests_total, 0)
        self.assertEqual(self.server.blocked_requests, 0)
    
    def test_security_headers(self):
        """Test that security headers are set"""
        with self.server.app.test_client() as client:
            response = client.get('/')
            
            # Check security headers
            self.assertIn('X-Content-Type-Options', response.headers)
            self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
            
            self.assertIn('X-Frame-Options', response.headers)
            self.assertEqual(response.headers['X-Frame-Options'], 'DENY')
    
    def test_health_check_endpoint(self):
        """Test health check endpoint"""
        with self.server.app.test_client() as client:
            response = client.get('/api/v1/health')
            
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data['status'], 'healthy')
            self.assertIn('uptime_seconds', data)
    
    def test_root_endpoint(self):
        """Test root endpoint information"""
        with self.server.app.test_client() as client:
            response = client.get('/')
            
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data['name'], 'BrowserAI Learning System API')
            self.assertIn('endpoints', data)
            self.assertIn('security_info', data)
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        with self.server.app.test_client() as client:
            response = client.get('/api/v1/metrics')
            
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertIn('server', data)
            self.assertIn('security', data)
            self.assertIn('learner', data)
    
    def test_security_status_endpoint(self):
        """Test security status endpoint"""
        with self.server.app.test_client() as client:
            response = client.get('/api/v1/security')
            
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertIn('security_features', data)
            self.assertIn('rate_limiter_config', data)


class TestAPIErrorHandling(unittest.TestCase):
    """Test API error handling"""
    
    def setUp(self):
        self.config = Config()
        self.config.ENABLE_JWT_AUTH = False
        self.server = BrowserAIServer(self.config)
    
    def test_invalid_request_body(self):
        """Test handling of invalid JSON"""
        with self.server.app.test_client() as client:
            response = client.post(
                '/api/v1/generate',
                data='invalid json',
                content_type='application/json'
            )
            
            # Should return error (400 or 500)
            self.assertIn(response.status_code, [400, 500])
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        with self.server.app.test_client() as client:
            response = client.post(
                '/api/v1/generate',
                json={
                    "url": "https://example.com",
                    "features": [0.1] * 48,
                    # Missing website_intent and design_style
                },
                content_type='application/json'
            )
            
            self.assertEqual(response.status_code, 400)
            data = response.get_json()
            self.assertIn('error', data)
    
    def test_oversized_request(self):
        """Test handling of oversized requests"""
        self.server.config.MAX_REQUEST_SIZE = 1000  # Small limit
        
        large_data = {
            "url": "https://example.com",
            "features": [0.1] * 48,
            "website_intent": "blog",
            "design_style": "modern",
            "timestamp": 1704067200,
            "session_id": "x" * 10000  # Large payload
        }
        
        with self.server.app.test_client() as client:
            response = client.post(
                '/api/v1/generate',
                json=large_data,
                content_type='application/json'
            )
            
            # Should be rejected for being too large
            self.assertIn(response.status_code, [413, 400])


def run_tests():
    """Run all tests"""
    logger.info("=" * 70)
    logger.info("Enhanced API Server Security Tests")
    logger.info("=" * 70)
    logger.info("")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRateLimiter))
    suite.addTests(loader.loadTestsFromTestCase(TestAuditLogger))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestAPISecurityFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIErrorHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info("=" * 70)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_tests())
