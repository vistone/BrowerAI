"""
Online Learning Integration - P1 #2
Integrates feature encoder, framework detector, and crawler system
into a unified online learning pipeline

Features:
- 48D feature extraction from website data  
- 256D latent encoding via feature encoder
- Framework detection and code validation
- Complete feedback loop: crawl -> extract -> encode -> generate -> validate -> learn
"""

import sys
sys.path.insert(0, '/home/stone/BrowerAI/training')

import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from collections import defaultdict, deque
import hashlib
import os

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
logging.basicConfig(level=getattr(logging, _LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in ("1", "true", "yes", "on")


class OnlineLearningIntegration:
    """Unified integration system for online learning pipeline"""
    
    def __init__(self, feature_dim=48, latent_dim=256, learning_rate=0.001):
        """Initialize integration system
        
        Args:
            feature_dim: Feature dimension (48 standard)
            latent_dim: Latent vector dimension (256)
            learning_rate: Learning rate for weight updates
        """
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        learning_mode = _env_flag("BROWERAI_LEARNING_MODE")
        use_gpu = _env_flag("BROWERAI_USE_GPU")
        gpu_device = os.getenv("BROWERAI_GPU_DEVICE", "0").strip() or "0"
        logger.info(
            "OnlineLearningIntegration start: learning_mode=%s, use_gpu=%s, gpu_device=%s",
            learning_mode,
            use_gpu,
            gpu_device,
        )
        
        # Load components with graceful degradation
        self.online_learner = None
        self.feature_encoder = None
        self.framework_detector = None
        self.code_generator = None
        
        try:
            from online_learner import OnlineLearner
            self.online_learner = OnlineLearner(
                feature_dim=feature_dim,
                latent_dim=latent_dim,
                learning_rate=learning_rate
            )
            logger.info("✓ OnlineLearner loaded")
        except Exception as e:
            logger.warning(f"⚠ OnlineLearner: {e}")
        
        try:
            from feature_encoder_enhanced import EnhancedFeatureEncoder
            self.feature_encoder = EnhancedFeatureEncoder()
            logger.info("✓ FeatureEncoder loaded")
        except Exception as e:
            logger.warning(f"⚠ FeatureEncoder: {e}")
        
        try:
            from framework_detector_enhanced import EnsembleFrameworkDetector
            self.framework_detector = EnsembleFrameworkDetector()
            logger.info("✓ FrameworkDetector loaded")
        except Exception as e:
            logger.warning(f"⚠ FrameworkDetector: {e}")
        
        try:
            from code_generator import CodeGenerator
            self.code_generator = CodeGenerator(latent_dim=latent_dim)
            logger.info("✓ CodeGenerator loaded")
        except Exception as e:
            logger.warning(f"⚠ CodeGenerator: {e}")
        
        # Metrics storage
        self.metrics = {
            'total_processed': 0,
            'feature_encoder_activations': 0,
            'framework_detections': 0,
            'average_latency_ms': 0.0,
            'total_weight_updates': 0,
            'cache_hit_count': 0
        }
        
        # Cache and history
        self.feature_cache = {}
        self.sessions = deque(maxlen=100)
        self.sample_log = deque(maxlen=1000)
        self.framework_stats = defaultdict(int)
        self.total_processing_time_ms = 0.0
    
    def process_website(self, website_data: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        """Process a single website through the complete pipeline
        
        Args:
            website_data: Website information (html, css, scripts, etc)
            session_id: Session identifier for tracking
        
        Returns:
            Result dictionary with framework, quality_score, processing_time_ms
        """
        if session_id is None:
            session_id = f"proc_{datetime.now().strftime('%H%M%S')}"
        
        try:
            start_time = datetime.now()
            
            # Step 1: Framework detection
            html = website_data.get("html", "")
            detected_frameworks = website_data.get("detected_frameworks", {})
            framework_confidence = 0.5
            
            if self.framework_detector:
                try:
                    result = self.framework_detector.detect(html)
                    detected_frameworks = result.detected_frameworks
                    framework_confidence = result.confidence
                    self.metrics['framework_detections'] += 1
                except Exception as e:
                    logger.debug(f"Framework detection failed: {e}")
            
            best_framework = max(detected_frameworks, key=detected_frameworks.get, default="unknown")
            self.framework_stats[best_framework] += 1
            
            # Step 2: Extract features (48D)
            features = self._extract_features(website_data)
            
            # Step 3: Check cache
            feature_hash = hashlib.md5(features.tobytes()).hexdigest()
            if feature_hash in self.feature_cache:
                latent_vector = self.feature_cache[feature_hash]
                self.metrics['cache_hit_count'] += 1
                is_cached = True
            else:
                # Encode features
                if self.feature_encoder:
                    try:
                        result = self.feature_encoder.encode(features)
                        latent_vector = result.get("latent_vector", self._simple_encoding(features))
                        self.metrics['feature_encoder_activations'] += 1
                    except Exception as e:
                        logger.debug(f"Feature encoding failed: {e}")
                        latent_vector = self._simple_encoding(features)
                else:
                    latent_vector = self._simple_encoding(features)
                
                self.feature_cache[feature_hash] = latent_vector
                is_cached = False
            
            # Step 4: Generate code
            generation_result = {}
            if self.code_generator:
                try:
                    generation_result = self.code_generator.generate(latent_vector)
                except Exception as e:
                    logger.debug(f"Code generation failed: {e}")
            
            # Step 5: Validate code
            validation_score = self._validate_generated_code(generation_result)
            
            # Step 6: Online learning update
            if self.online_learner:
                try:
                    feedback = {
                        "quality_score": (framework_confidence + validation_score) / 2,
                        "framework": best_framework
                    }
                    self.online_learner.process_feedback(
                        features=features,
                        generated_latent=latent_vector,
                        feedback_data=feedback
                    )
                    self.metrics['total_weight_updates'] += 1
                except Exception as e:
                    logger.debug(f"Online learning update failed: {e}")
            
            # Compute metrics
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.total_processing_time_ms += elapsed_ms
            self.metrics['total_processed'] += 1
            self.metrics['average_latency_ms'] = (
                self.total_processing_time_ms / max(self.metrics['total_processed'], 1)
            )
            
            # Log sample
            self.sample_log.append({
                "session": session_id,
                "framework": best_framework,
                "quality": validation_score,
                "time_ms": elapsed_ms,
                "cached": is_cached
            })
            
            return {
                "success": True,
                "framework": best_framework,
                "quality_score": float((framework_confidence + validation_score) / 2),
                "processing_time_ms": elapsed_ms,
                "cached": is_cached
            }
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def batch_process(self, websites: List[Dict[str, Any]], session_id: str = None) -> Dict[str, Any]:
        """Process multiple websites in a batch
        
        Args:
            websites: List of website data dictionaries
            session_id: Batch session identifier
        
        Returns:
            Batch result with summary statistics
        """
        if session_id is None:
            session_id = f"batch_{datetime.now().strftime('%H%M%S')}"
        
        results = []
        successful = 0
        failed = 0
        total_time = 0.0
        frameworks = defaultdict(int)
        
        for i, website_data in enumerate(websites):
            result = self.process_website(website_data, f"{session_id}_{i}")
            results.append(result)
            
            if result.get("success"):
                successful += 1
                frameworks[result.get("framework", "unknown")] += 1
                total_time += result.get("processing_time_ms", 0)
            else:
                failed += 1
        
        self.sessions.append({
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(websites),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / max(len(websites), 1),
            "avg_time_ms": total_time / max(successful, 1),
            "frameworks": dict(frameworks)
        })
        
        return {
            "session_id": session_id,
            "total": len(websites),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / max(len(websites), 1),
            "frameworks": dict(frameworks),
            "results": results
        }
    
    def _extract_features(self, website_data: Dict[str, Any]) -> np.ndarray:
        """Extract 48D feature vector from website data"""
        features = np.zeros(48, dtype=np.float32)
        
        # HTML features (0-4)
        html = website_data.get("html", "")
        features[0] = len(html) / 100000
        features[1] = min(html.count("<div>"), 100)
        features[2] = min(html.count("<section>"), 100)
        features[3] = min(html.count("<article>"), 100)
        features[4] = 1.0 if "<!DOCTYPE" in html else 0.0
        
        # CSS features (5-9)
        css_data = website_data.get("css", [])
        features[5] = len(css_data)
        features[6] = sum(c.get("size", 0) for c in css_data) / 10000
        features[7] = sum(1 for c in css_data if c.get("type") == "external")
        features[8] = sum(1 for c in css_data if c.get("type") == "inline")
        features[9] = 0.0
        
        # JS features (10-19)
        scripts = website_data.get("scripts", [])
        features[10] = len(scripts)
        features[11] = sum(1 for s in scripts if s.get("type") == "external")
        features[12] = sum(1 for s in scripts if s.get("type") == "inline")
        features[13] = sum(1 for s in scripts if s.get("async"))
        features[14] = sum(1 for s in scripts if s.get("defer"))
        features[15] = sum(s.get("size", 0) for s in scripts) / 100000
        features[16] = min(html.count("async"), 50)
        features[17] = min(html.count("await"), 50)
        features[18] = min(html.count("Promise"), 50)
        features[19] = min(html.count("fetch("), 50)
        
        # Framework signals (20-27)
        frameworks = website_data.get("detected_frameworks", {})
        framework_map = {
            "React": 20, "Vue": 21, "Angular": 22, "Svelte": 23,
            "Next.js": 24, "Nuxt": 25, "Express": 26, "Fastify": 27
        }
        for fw, score in frameworks.items():
            idx = framework_map.get(fw)
            if idx is not None:
                features[idx] = score
        
        # Performance metrics (28-37)
        metadata = website_data.get("metadata", {})
        features[28] = metadata.get("script_count", 0) / 20
        features[29] = metadata.get("css_count", 0) / 10
        features[30] = metadata.get("html_size", 0) / 1000000
        features[31] = metadata.get("response_time_ms", 0) / 1000
        features[32] = 1.0 if metadata.get("from_cache") else 0.0
        features[33] = website_data.get("success", 1.0)
        features[34:38] = 0.0
        
        # Metadata labels (38-47)
        features[38] = 1.0 if website_data.get("title") else 0.0
        features[39] = 1.0 if website_data.get("description") else 0.0
        features[40] = 1.0 if website_data.get("service_worker") else 0.0
        features[41] = 1.0 if website_data.get("async_support") else 0.0
        features[42] = 1.0 if website_data.get("language") else 0.0
        features[43] = 1.0 if website_data.get("responsive") else 0.0
        features[44] = 1.0 if website_data.get("pwa") else 0.0
        features[45] = 1.0 if website_data.get("https") else 0.0
        features[46] = metadata.get("ttfb_ms", 0) / 1000
        features[47] = metadata.get("fcp_ms", 0) / 1000
        
        return features
    
    def _simple_encoding(self, features: np.ndarray) -> np.ndarray:
        """Simple linear encoding: 48D -> 256D"""
        np.random.seed(hash(features.tobytes()) % 2**32)
        encoding_matrix = np.random.randn(self.feature_dim, self.latent_dim) * 0.1
        return features @ encoding_matrix
    
    def _validate_generated_code(self, result: Dict[str, Any]) -> float:
        """Validate generated code quality (0-1 score)"""
        if not result:
            return 0.0
        
        # Simple heuristic: check if result has expected fields
        score = 0.0
        if result.get("html"):
            score += 0.3
        if result.get("css"):
            score += 0.3
        if result.get("js"):
            score += 0.4
        
        return min(score, 1.0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return self.metrics.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return {
            "components": {
                "online_learner": self.online_learner is not None,
                "feature_encoder": self.feature_encoder is not None,
                "framework_detector": self.framework_detector is not None,
                "code_generator": self.code_generator is not None
            },
            "metrics": self.get_metrics(),
            "cache_size": len(self.feature_cache),
            "sessions_count": len(self.sessions),
            "sample_log_count": len(self.sample_log)
        }


if __name__ == "__main__":
    logger.info("Starting OnlineLearningIntegration demo run")
    integration = OnlineLearningIntegration()
    status = integration.get_system_status()
    logger.info("System status: %s", status)

    demo_website = {
        "html": "<html><head><title>Demo</title></head><body><div>Hi</div></body></html>",
        "css": [{"type": "inline", "size": 120}],
        "scripts": [{"type": "inline", "size": 240, "async": False, "defer": False}],
        "metadata": {"html_size": 1200, "response_time_ms": 120, "from_cache": False},
        "success": 1.0,
        "title": "Demo",
        "description": "Demo page",
        "responsive": True,
        "https": True,
    }

    result = integration.process_website(demo_website, session_id="demo_start")
    logger.info("Demo result: %s", result)
