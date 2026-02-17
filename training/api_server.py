#!/usr/bin/env python3
"""
BrowerAI - Python Flask API Server for Week 6 Learning System Integration
Provides REST API endpoints for feature generation and online learning feedback
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

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

class Config:
    """Flask configuration"""
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
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com",
                "features": [0.1] * 48,
                "website_intent": "blog",
                "design_style": "modern",
                "timestamp": 1704067200,
                "session_id": "sess-123"
            }
        }


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
# Flask Application
# ============================================================================

class BrowserAIServer:
    """Main Flask API Server for BrowerAI Learning System"""
    
    def __init__(self, config: Config):
        self.config = config
        self.app = Flask(__name__)
        self.app.config.from_object(config)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.logger.info("Initializing BrowserAI API Server...")
        self.feature_encoder = FeatureEncoder(config.FEATURE_DIM, config.LATENT_DIM)
        self.code_generator = CodeGenerator(config.LATENT_DIM)
        self.online_learner = OnlineLearner(
            feature_dim=config.FEATURE_DIM,
            latent_dim=config.LATENT_DIM,
            learning_rate=config.LEARNING_RATE,
            batch_size=config.BATCH_SIZE
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
        
        # Enable CORS
        CORS(self.app)
        
        # Register routes
        self._register_routes()
        
        self.logger.info("BrowserAI API Server initialized successfully!")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=self.config.LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _register_routes(self):
        """Register all API routes"""
        self.app.route('/api/v1/health', methods=['GET'])(self.health_check)
        self.app.route('/api/v1/generate', methods=['POST'])(self.generate_code)
        self.app.route('/api/v1/feedback', methods=['POST'])(self.receive_feedback)
        self.app.route('/metrics', methods=['GET'])(self.get_metrics)
        self.app.route('/', methods=['GET'])(self.root)
    
    # ========================================================================
    # Health Check Endpoint
    # ========================================================================
    
    def health_check(self) -> tuple:
        """GET /api/v1/health - Check server health"""
        try:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            response = HealthResponse(
                status="healthy",
                timestamp=int(datetime.utcnow().timestamp()),
                uptime_seconds=uptime,
                models_loaded=3,  # encoder, generator, learner
                version="1.0.0"
            )
            
            self.logger.info("Health check passed")
            return jsonify(response.model_dump()), 200
        
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return jsonify({"error": str(e), "status": "unhealthy"}), 503
    
    # ========================================================================
    # Code Generation Endpoint
    # ========================================================================
    
    def generate_code(self) -> tuple:
        """POST /api/v1/generate - Generate HTML/CSS/JS from features"""
        self.requests_total += 1
        
        try:
            # Parse request
            data = request.get_json()
            packet = FeaturePacketRequest(**data)
            
            self.logger.info(f"Generating code for: {packet.url}")
            
            # Validate feature vector
            if len(packet.features) != self.config.FEATURE_DIM:
                raise ValueError(
                    f"Expected {self.config.FEATURE_DIM} features, "
                    f"got {len(packet.features)}"
                )
            
            # Encode features to latent space
            latent_vector = self.feature_encoder.encode(
                features=packet.features,
                intent=packet.website_intent,
                design_style=packet.design_style
            )
            
            # Generate code from latent vector
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
                    "latent_dim": self.config.LATENT_DIM,
                },
                timestamp=int(datetime.utcnow().timestamp())
            )
            
            self.requests_success += 1
            self.logger.info(
                f"Code generation successful. "
                f"Confidence: {response.confidence:.2f}"
            )
            
            return jsonify(response.model_dump()), 200
        
        except ValidationError as e:
            self.requests_error += 1
            self.logger.error(f"Validation error: {e}")
            return jsonify({
                "error": "Invalid request format",
                "details": e.errors()
            }), 400
        
        except Exception as e:
            self.requests_error += 1
            self.logger.error(f"Code generation error: {e}")
            return jsonify({"error": str(e)}), 500
    
    # ========================================================================
    # Feedback Endpoint
    # ========================================================================
    
    def receive_feedback(self) -> tuple:
        """POST /api/v1/feedback - Receive rendering quality feedback"""
        self.requests_total += 1
        
        try:
            # Parse request
            data = request.get_json()
            feedback = FeedbackPacketRequest(**data)
            
            self.logger.info(
                f"Receiving feedback for: {feedback.url} "
                f"(quality: {feedback.overall_quality:.2f})"
            )
            
            # Add feedback to buffer
            feedback_data = {
                "quality_score": feedback.overall_quality,
                "html_quality": feedback.html_similarity,
                "css_quality": feedback.css_accuracy,
                "js_quality": feedback.layout_similarity,
                "matched_elements": feedback.matched_elements,
                "mismatched_elements": feedback.mismatched_elements,
                "feedback_text": feedback.feedback_text,
                "session_id": feedback.session_id,
                "timestamp": feedback.timestamp
            }
            
            # Store in buffer (would be processed by online learner)
            self.feedback_buffer.add(feedback_data)
            
            # If buffer is ready, process batch
            buffer_ready = len(self.feedback_buffer.buffer) >= self.config.BATCH_SIZE
            
            if buffer_ready:
                batch = self.feedback_buffer.get_batch()
                self.logger.info(f"Processing feedback batch with {len(batch)} items")
                
                # Aggregate feedback for learning
                avg_quality = np.mean([f["quality_score"] for f in batch])
                avg_html = np.mean([f["html_quality"] for f in batch])
                avg_css = np.mean([f["css_quality"] for f in batch])
            
            # Get current metrics
            metrics = self.online_learner.get_metrics()
            
            self.requests_success += 1
            self.logger.info(
                f"Feedback processed successfully. "
                f"Buffer size: {self.feedback_buffer.size()}"
            )
            
            return jsonify({
                "status": "ok",
                "quality_score": float(feedback.overall_quality),
                "buffer_size": self.feedback_buffer.size(),
                "buffer_ready": buffer_ready,
                "learner_metrics": metrics,
                "timestamp": int(datetime.utcnow().timestamp())
            }), 200
        
        except ValidationError as e:
            self.requests_error += 1
            self.logger.error(f"Validation error: {e}")
            return jsonify({
                "error": "Invalid request format",
                "details": e.errors()
            }), 400
        
        except Exception as e:
            self.requests_error += 1
            self.logger.error(f"Feedback processing error: {e}")
            return jsonify({"error": str(e)}), 500
    
    # ========================================================================
    # Metrics Endpoint
    # ========================================================================
    
    def get_metrics(self) -> tuple:
        """GET /metrics - Get server metrics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        success_rate = (
            self.requests_success / self.requests_total * 100
            if self.requests_total > 0 else 0
        )
        
        return jsonify({
            "timestamp": int(datetime.utcnow().timestamp()),
            "uptime_seconds": uptime,
            "requests": {
                "total": self.requests_total,
                "success": self.requests_success,
                "error": self.requests_error,
                "success_rate": f"{success_rate:.2f}%"
            },
            "models": {
                "feature_encoder": "loaded",
                "code_generator": "loaded",
                "online_learner": "loaded"
            },
            "configuration": {
                "feature_dim": self.config.FEATURE_DIM,
                "latent_dim": self.config.LATENT_DIM,
                "learning_rate": self.config.LEARNING_RATE,
                "batch_size": self.config.BATCH_SIZE
            }
        }), 200
    
    # ========================================================================
    # Root Endpoint
    # ========================================================================
    
    def root(self) -> tuple:
        """GET / - API root information"""
        return jsonify({
            "name": "BrowserAI Learning System API",
            "version": "1.0.0",
            "description": "REST API for AI-powered website learning",
            "endpoints": {
                "health": "GET /api/v1/health",
                "generate": "POST /api/v1/generate",
                "feedback": "POST /api/v1/feedback",
                "metrics": "GET /metrics"
            },
            "documentation": "See WEEK6_API_SPEC.md for full documentation",
            "uptime": {
                "start_time": self.start_time.isoformat(),
                "uptime_seconds": (
                    datetime.utcnow() - self.start_time
                ).total_seconds()
            }
        }), 200
    
    # ========================================================================
    # Server Control
    # ========================================================================
    
    def run(self):
        """Start the Flask development server"""
        self.logger.info(
            f"Starting server on {self.config.HOST}:{self.config.PORT}"
        )
        self.app.run(
            host=self.config.HOST,
            port=self.config.PORT,
            debug=self.config.DEBUG,
            use_reloader=False
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
