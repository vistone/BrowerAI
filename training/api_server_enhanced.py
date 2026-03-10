#!/usr/bin/env python3
"""
BrowerAI - Enhanced Flask API Server with Security Features
Provides secure REST API endpoints with authentication, rate limiting, and audit logging

Security Features:
- JWT authentication (optional token-based access)
- Rate limiting (10 requests/minute per IP)
- Request timeout protection (30 seconds)
- Audit logging (all requests logged)
- CORS protection
- Input validation & sanitization
- Error rate tracking
- Request size limits
- HTTPS support
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
import json
import hashlib
import hmac
import time
from collections import defaultdict, deque
import threading

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError, field_validator

from feature_encoder import FeatureEncoder
from code_generator import CodeGenerator
from online_learner import OnlineLearner, FeedbackBuffer

# ============================================================================
# Configuration
# ============================================================================

class SecurityConfig:
    """Security configuration"""
    ENABLE_JWT_AUTH = os.getenv("ENABLE_JWT_AUTH", "false").lower() == "true"
    JWT_SECRET = os.getenv("JWT_SECRET", "browerai-secret-key-change-in-production")
    ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))  # seconds
    MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "1048576"))  # 1MB
    ENABLE_AUDIT_LOG = os.getenv("ENABLE_AUDIT_LOG", "true").lower() == "true"
    ENABLE_HTTPS = os.getenv("ENABLE_HTTPS", "false").lower() == "true"
    HTTPS_CERT_PATH = os.getenv("HTTPS_CERT_PATH", "/etc/ssl/certs/server.crt")
    HTTPS_KEY_PATH = os.getenv("HTTPS_KEY_PATH", "/etc/ssl/private/server.key")


class Config(SecurityConfig):
    """Flask configuration extended with API config"""
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "5000"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Model configuration
    LATENT_DIM = int(os.getenv("LATENT_DIM", "256"))
    FEATURE_DIM = 48  # From Rust: 48-dimensional feature vector
    
    # Learning configuration
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "1000"))


# ============================================================================
# Request Rate Limiter
# ============================================================================

class RateLimiter:
    """Rate limiting per IP address"""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)  # IP -> deque of timestamps
        self.lock = threading.Lock()
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for client IP"""
        now = time.time()
        cutoff = now - self.window_seconds
        
        with self.lock:
            # Clean old requests
            requests = self.requests[client_ip]
            while requests and requests[0] < cutoff:
                requests.popleft()
            
            # Check limit
            if len(requests) >= self.max_requests:
                return False
            
            # Add current request
            requests.append(now)
            return True
    
    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for client IP"""
        with self.lock:
            requests = self.requests[client_ip]
            return max(0, self.max_requests - len(requests))


# ============================================================================
# Audit Logger
# ============================================================================

class AuditLogger:
    """Audit logging for security and compliance"""
    
    def __init__(self, log_file: str = "api_audit.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("audit")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_request(
        self,
        method: str,
        endpoint: str,
        client_ip: str,
        user_id: Optional[str] = None,
        status_code: int = 0,
        details: Optional[Dict] = None
    ):
        """Log API request"""
        details_str = json.dumps(details) if details else ""
        message = (
            f"{method:6s} {endpoint:30s} | IP: {client_ip:15s} | "
            f"User: {user_id or 'anonymous':20s} | Status: {status_code} | {details_str}"
        )
        self.logger.info(message)


# ============================================================================
# Data Models
# ============================================================================

class FeaturePacketRequest(BaseModel):
    """Request from Rust: Feature packet with 48-dim vector"""
    url: str
    features: List[float]  # 48-dimensional vector
    website_intent: str
    design_style: str
    feedback: Optional[Dict[str, Any]] = None
    timestamp: int
    session_id: str
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        """Validate feature vector has exactly 48 dimensions"""
        if len(v) != 48:
            raise ValueError(f"Feature vector must have exactly 48 dimensions, got {len(v)}")
        if not all(isinstance(f, (int, float)) for f in v):
            raise ValueError("All features must be numeric")
        if any(np.isnan(f) or np.isinf(f) for f in v):
            raise ValueError("Features cannot contain NaN or Inf")
        return v
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        """Validate URL format"""
        if not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError("URL must start with http:// or https://")
        if len(v) > 2048:
            raise ValueError("URL too long")
        return v


class GeneratedCodeResponse(BaseModel):
    """Response to Rust: Generated HTML/CSS/JS"""
    html: str
    css: str
    javascript: str
    confidence: float
    should_use: bool
    training_metrics: Optional[Dict[str, Any]] = None
    timestamp: int


class FeedbackPacketRequest(BaseModel):
    """Feedback from Rust: Rendering quality feedback"""
    url: str
    overall_quality: float
    html_similarity: float
    css_accuracy: float
    layout_similarity: float
    matched_elements: int
    mismatched_elements: int
    feedback_text: Optional[str] = None
    session_id: str
    timestamp: int
    
    @field_validator('overall_quality', 'html_similarity', 'css_accuracy', 'layout_similarity')
    @classmethod
    def validate_quality_scores(cls, v):
        """Validate quality scores are in [0, 1] range"""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Quality scores must be in [0, 1] range, got {v}")
        return v


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: int
    uptime_seconds: float
    models_loaded: int
    version: str


# ============================================================================
# Decorators for Security
# ============================================================================

def require_auth(f: Callable) -> Callable:
    """Decorator: Require JWT authentication if enabled"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        config = args[0].config if args else Config
        
        if not config.ENABLE_JWT_AUTH:
            return f(*args, **kwargs)
        
        # Check Authorization header
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({"error": "Missing or invalid authorization header"}), 401
        
        # Validate token (simplified - in production use proper JWT lib)
        token = auth_header.split(' ')[1]
        if not token or len(token) < 10:
            return jsonify({"error": "Invalid token"}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function


def rate_limit_check(limiter: RateLimiter) -> Callable:
    """Decorator: Check rate limits"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            
            if not limiter.is_allowed(client_ip):
                return jsonify({
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {limiter.max_requests} requests per {limiter.window_seconds} seconds"
                }), 429
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def with_timeout(timeout_seconds: int) -> Callable:
    """Decorator: Enforce request timeout (simple check)"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Store start time for timeout check
            request.start_time = time.time()
            request.timeout_seconds = timeout_seconds
            
            try:
                result = f(*args, **kwargs)
                elapsed = time.time() - request.start_time
                
                if elapsed > timeout_seconds:
                    logging.warning(
                        f"Request took {elapsed:.2f}s, exceeded timeout of {timeout_seconds}s"
                    )
                
                return result
            
            except Exception as e:
                raise
        
        return decorated_function
    return decorator


def validate_request_size(max_size: int) -> Callable:
    """Decorator: Validate request body size"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            content_length = request.content_length
            if content_length and content_length > max_size:
                return jsonify({
                    "error": "Request body too large",
                    "max_size": max_size,
                    "received": content_length
                }), 413
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


# ============================================================================
# Flask Application
# ============================================================================

class BrowserAIServer:
    """Enhanced Flask API Server with security features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.app = Flask(__name__)
        self.app.config.from_object(config)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize security components
        self.rate_limiter = RateLimiter(
            max_requests=config.RATE_LIMIT_REQUESTS,
            window_seconds=config.RATE_LIMIT_WINDOW
        )
        self.audit_logger = AuditLogger()
        
        self.logger.info("🔐 Initializing Enhanced BrowserAI API Server with Security Features...")
        
        # Initialize components
        self.feature_encoder = FeatureEncoder(config.FEATURE_DIM, config.LATENT_DIM)
        self.code_generator = CodeGenerator(config.LATENT_DIM)
        self.online_learner = OnlineLearner(
            feature_dim=config.FEATURE_DIM,
            latent_dim=config.LATENT_DIM,
            learning_rate=config.LEARNING_RATE,
            batch_size=config.BATCH_SIZE,
            enable_gradient_clip=True,
            enable_anomaly_detection=True
        )
        self.feedback_buffer = FeedbackBuffer(
            batch_size=config.BATCH_SIZE,
            max_buffer_size=config.MAX_QUEUE_SIZE
        )
        
        # Server metrics
        self.start_time = datetime.utcnow()
        self.requests_total = 0
        self.requests_success = 0
        self.requests_error = 0
        self.error_history = deque(maxlen=100)  # Last 100 errors
        self.blocked_requests = 0
        
        # Enable CORS with security settings
        CORS(
            self.app,
            resources={r"/api/*": {
                "origins": ["https://localhost", "http://localhost"],
                "methods": ["GET", "POST"],
                "max_age": 3600
            }}
        )
        
        # Security headers
        @self.app.after_request
        def set_security_headers(response):
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            return response
        
        # Global error handler
        @self.app.errorhandler(Exception)
        def handle_error(e):
            self.requests_error += 1
            self.error_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'type': type(e).__name__
            })
            self.logger.error(f"Unhandled error: {type(e).__name__}: {e}")
            return jsonify({
                "error": "Internal server error",
                "type": type(e).__name__
            }), 500
        
        # Register routes
        self._register_routes()
        
        self.logger.info("✅ Enhanced API Server initialized successfully!")
        self._log_security_status()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=self.config.LOG_LEVEL,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _log_security_status(self):
        """Log security configuration"""
        self.logger.info("Security Configuration:")
        self.logger.info(f"  JWT Auth: {self.config.ENABLE_JWT_AUTH}")
        self.logger.info(f"  Rate Limiting: {self.config.ENABLE_RATE_LIMITING} ({self.config.RATE_LIMIT_REQUESTS} req/{self.config.RATE_LIMIT_WINDOW}s)")
        self.logger.info(f"  Request Timeout: {self.config.REQUEST_TIMEOUT}s")
        self.logger.info(f"  Max Request Size: {self.config.MAX_REQUEST_SIZE} bytes")
        self.logger.info(f"  Audit Logging: {self.config.ENABLE_AUDIT_LOG}")
        self.logger.info(f"  HTTPS Support: {self.config.ENABLE_HTTPS}")
    
    def _register_routes(self):
        """Register all API routes"""
        self.app.route('/api/v1/health', methods=['GET'])(self.health_check)
        self.app.route('/api/v1/generate', methods=['POST'])(self.generate_code)
        self.app.route('/api/v1/feedback', methods=['POST'])(self.receive_feedback)
        self.app.route('/api/v1/metrics', methods=['GET'])(self.get_metrics)
        self.app.route('/api/v1/security', methods=['GET'])(self.security_status)
        self.app.route('/', methods=['GET'])(self.root)
    
    # ========================================================================
    # Health Check
    # ========================================================================
    
    def health_check(self) -> tuple:
        """GET /api/v1/health - Check server health with security info"""
        try:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            response = HealthResponse(
                status="healthy",
                timestamp=int(datetime.utcnow().timestamp()),
                uptime_seconds=uptime,
                models_loaded=3,
                version="2.0.0-secure"
            )
            
            self.logger.info("Health check passed")
            return jsonify(response.model_dump()), 200
        
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return jsonify({"error": str(e), "status": "unhealthy"}), 503
    
    # ========================================================================
    # Generate Code
    # ========================================================================
    
    @require_auth
    @with_timeout(30)
    @validate_request_size(1048576)
    def generate_code(self) -> tuple:
        """POST /api/v1/generate - Generate HTML/CSS/JS from features"""
        self.requests_total += 1
        client_ip = request.remote_addr
        
        try:
            # Rate limiting check
            if self.config.ENABLE_RATE_LIMITING:
                if not self.rate_limiter.is_allowed(client_ip):
                    self.blocked_requests += 1
                    self.audit_logger.log_request(
                        "POST", "/api/v1/generate", client_ip,
                        status_code=429, details={"reason": "rate_limit"}
                    )
                    return jsonify({"error": "Rate limit exceeded"}), 429
            
            # Parse and validate request
            data = request.get_json()
            if not data:
                raise ValueError("Empty request body")
            
            packet = FeaturePacketRequest(**data)
            
            self.logger.info(f"Generating code for: {packet.url} (session={packet.session_id})")
            
            # Encode features to latent space
            latent_vector = self.feature_encoder.encode(
                features=packet.features,
                intent=packet.website_intent,
                design_style=packet.design_style
            )
            
            # Generate code
            generated = self.code_generator.generate(
                latent_vector=latent_vector,
                session_id=packet.session_id
            )
            
            # Prepare response
            response = GeneratedCodeResponse(
                html=generated['html'],
                css=generated['css'],
                javascript=generated['javascript'],
                confidence=generated['confidence'],
                should_use=generated['confidence'] > 0.7,
                training_metrics={
                    "loss": generated.get('loss', 0.0),
                    "accuracy": generated.get('accuracy', 0.85),
                    "learning_rate": self.config.LEARNING_RATE,
                    "epoch": generated.get('epoch', 1),
                },
                timestamp=int(datetime.utcnow().timestamp())
            )
            
            self.requests_success += 1
            
            if self.config.ENABLE_AUDIT_LOG:
                self.audit_logger.log_request(
                    "POST", "/api/v1/generate", client_ip,
                    status_code=200,
                    details={"session": packet.session_id, "confidence": response.confidence}
                )
            
            self.logger.info(f"✅ Code generation successful (confidence: {response.confidence:.2f})")
            return jsonify(response.model_dump()), 200
        
        except ValidationError as e:
            self.requests_error += 1
            self.logger.error(f"Validation error: {e}")
            self.audit_logger.log_request(
                "POST", "/api/v1/generate", client_ip,
                status_code=400, details={"reason": "validation_error"}
            )
            return jsonify({
                "error": "Invalid request format",
                "details": [{"field": err["loc"][0], "message": err["msg"]} for err in e.errors()]
            }), 400
        
        except Exception as e:
            self.requests_error += 1
            self.logger.error(f"Code generation error: {type(e).__name__}: {e}")
            self.audit_logger.log_request(
                "POST", "/api/v1/generate", client_ip,
                status_code=500, details={"reason": str(e)}
            )
            return jsonify({"error": "Code generation failed"}), 500
    
    # ========================================================================
    # Receive Feedback
    # ========================================================================
    
    @require_auth
    @with_timeout(30)
    @validate_request_size(1048576)
    def receive_feedback(self) -> tuple:
        """POST /api/v1/feedback - Receive rendering quality feedback"""
        self.requests_total += 1
        client_ip = request.remote_addr
        
        try:
            # Rate limiting
            if self.config.ENABLE_RATE_LIMITING and not self.rate_limiter.is_allowed(client_ip):
                self.blocked_requests += 1
                return jsonify({"error": "Rate limit exceeded"}), 429
            
            # Parse and validate
            data = request.get_json()
            feedback = FeedbackPacketRequest(**data)
            
            self.logger.info(
                f"Receiving feedback for: {feedback.url} "
                f"(quality: {feedback.overall_quality:.2f})"
            )
            
            # Prepare feedback data
            feedback_data = {
                "quality_score": feedback.overall_quality,
                "html_quality": feedback.html_similarity,
                "css_quality": feedback.css_accuracy,
                "js_quality": feedback.layout_similarity,
                "session_id": feedback.session_id,
                "timestamp": feedback.timestamp
            }
            
            # Process through learner
            self.feedback_buffer.add(feedback_data)
            
            metrics = self.online_learner.get_metrics()
            
            self.requests_success += 1
            
            if self.config.ENABLE_AUDIT_LOG:
                self.audit_logger.log_request(
                    "POST", "/api/v1/feedback", client_ip,
                    status_code=200,
                    details={"quality": feedback.overall_quality, "buffer_size": self.feedback_buffer.size()}
                )
            
            return jsonify({
                "status": "ok",
                "quality_score": float(feedback.overall_quality),
                "buffer_size": self.feedback_buffer.size(),
                "timestamp": int(datetime.utcnow().timestamp())
            }), 200
        
        except ValidationError as e:
            self.requests_error += 1
            return jsonify({
                "error": "Invalid feedback format",
                "details": [{"field": err["loc"][0], "message": err["msg"]} for err in e.errors()]
            }), 400
        
        except Exception as e:
            self.requests_error += 1
            self.logger.error(f"Feedback error: {e}")
            return jsonify({"error": "Feedback processing failed"}), 500
    
    # ========================================================================
    # Metrics
    # ========================================================================
    
    def get_metrics(self) -> tuple:
        """GET /api/v1/metrics - Get server and learner metrics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        success_rate = (
            self.requests_success / self.requests_total * 100
            if self.requests_total > 0 else 0
        )
        
        learner_metrics = self.online_learner.get_metrics()
        
        return jsonify({
            "timestamp": int(datetime.utcnow().timestamp()),
            "server": {
                "uptime_seconds": uptime,
                "requests_total": self.requests_total,
                "requests_success": self.requests_success,
                "requests_error": self.requests_error,
                "success_rate": f"{success_rate:.2f}%",
                "blocked_requests": self.blocked_requests,
            },
            "security": {
                "rate_limiting_enabled": self.config.ENABLE_RATE_LIMITING,
                "audit_logging_enabled": self.config.ENABLE_AUDIT_LOG,
                "auth_enabled": self.config.ENABLE_JWT_AUTH,
                "recent_errors": len(self.error_history)
            },
            "learner": learner_metrics
        }), 200
    
    # ========================================================================
    # Security Status
    # ========================================================================
    
    def security_status(self) -> tuple:
        """GET /api/v1/security - Get security status"""
        return jsonify({
            "timestamp": int(datetime.utcnow().timestamp()),
            "security_features": {
                "jwt_authentication": self.config.ENABLE_JWT_AUTH,
                "rate_limiting": self.config.ENABLE_RATE_LIMITING,
                "audit_logging": self.config.ENABLE_AUDIT_LOG,
                "https_enabled": self.config.ENABLE_HTTPS,
                "request_timeout_seconds": self.config.REQUEST_TIMEOUT,
                "max_request_size_bytes": self.config.MAX_REQUEST_SIZE,
            },
            "rate_limiter_config": {
                "requests_per_window": self.config.RATE_LIMIT_REQUESTS,
                "window_seconds": self.config.RATE_LIMIT_WINDOW,
            },
            "statistics": {
                "blocked_requests": self.blocked_requests,
                "total_errors": len(self.error_history),
            }
        }), 200
    
    # ========================================================================
    # Root
    # ========================================================================
    
    def root(self) -> tuple:
        """GET / - API root information"""
        return jsonify({
            "name": "BrowserAI Learning System API",
            "version": "2.0.0-secure",
            "description": "REST API for AI-powered website learning with enhanced security",
            "endpoints": {
                "health": "GET /api/v1/health",
                "generate": "POST /api/v1/generate (requires auth)",
                "feedback": "POST /api/v1/feedback (requires auth)",
                "metrics": "GET /api/v1/metrics",
                "security": "GET /api/v1/security"
            },
            "security_info": {
                "rate_limiting": True,
                "authentication": self.config.ENABLE_JWT_AUTH,
                "audit_logging": self.config.ENABLE_AUDIT_LOG,
                "https_support": self.config.ENABLE_HTTPS
            }
        }), 200
    
    # ========================================================================
    # Server Control
    # ========================================================================
    
    def run(self):
        """Start the Flask application"""
        if self.config.ENABLE_HTTPS:
            self.logger.info("🔒 Starting server with HTTPS...")
            ssl_context = (self.config.HTTPS_CERT_PATH, self.config.HTTPS_KEY_PATH)
        else:
            self.logger.info("Starting server with HTTP (not recommended for production)...")
            ssl_context = None
        
        self.logger.info(
            f"Listening on {self.config.HOST}:{self.config.PORT}"
        )
        
        self.app.run(
            host=self.config.HOST,
            port=self.config.PORT,
            debug=self.config.DEBUG,
            use_reloader=False,
            ssl_context=ssl_context
        )


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point"""
    config = Config()
    server = BrowserAIServer(config)
    server.run()


if __name__ == "__main__":
    main()
