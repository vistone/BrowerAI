/// Rust-Python Communication Bridge
/// Implements HTTP API for sending features to Python OnlineLearningEngine
/// and receiving generated code back

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Feature vector sent to Python
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturePacket {
    /// Website URL for tracking
    pub url: String,
    
    /// 48-dimensional feature vector
    pub features: Vec<f32>,
    
    /// Website intent/type classification
    pub website_intent: String,
    
    /// Design characteristics
    pub design_style: String,
    
    /// Feedback from previous rendering (optional)
    pub feedback: Option<RenderingFeedback>,
    
    /// Timestamp for session tracking
    pub timestamp: i64,
    
    /// Session ID linking multiple visits
    pub session_id: String,
}

/// Feedback from rendering the generated code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderingFeedback {
    /// Quality score (0.0-1.0)
    pub quality_score: f32,
    
    /// Elements that matched original
    pub matched_elements: usize,
    
    /// Elements that didn't match
    pub mismatched_elements: usize,
    
    /// CSS accuracy percentage
    pub css_accuracy: f32,
    
    /// Layout similarity percentage
    pub layout_similarity: f32,
    
    /// Human feedback if available
    pub human_feedback: Option<String>,
}

/// Response from Python OnlineLearningEngine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedWebsitePacket {
    /// Generated HTML code
    pub html: String,
    
    /// Generated CSS code
    pub css: String,
    
    /// Generated JavaScript
    pub javascript: String,
    
    /// Confidence score for generation (0.0-1.0)
    pub confidence: f32,
    
    /// Whether to use this generation (vs fallback to original)
    pub should_use: bool,
    
    /// Training metrics from Python side
    pub training_metrics: Option<TrainingMetrics>,
    
    /// Timestamp of generation
    pub timestamp: i64,
}

/// Training metrics returned from Python
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Loss value from training
    pub loss: f32,
    
    /// Accuracy on validation set
    pub accuracy: f32,
    
    /// Learning rate used
    pub learning_rate: f32,
    
    /// Epoch number
    pub epoch: u32,
    
    /// Latent space dimension
    pub latent_dim: usize,
    
    /// Additional metrics
    pub additional: HashMap<String, f32>,
}

/// HTTP Client for Rust → Python communication
pub struct RustPythonBridge {
    /// Python server URL (e.g., "http://localhost:5000")
    pub python_server_url: String,
    
    /// HTTP client instance
    http_client: reqwest::Client,
    
    /// Request timeout in seconds
    timeout_seconds: u64,
    
    /// Retry attempts
    max_retries: usize,
}

impl RustPythonBridge {
    /// Create new bridge to Python server
    pub fn new(python_server_url: String) -> Self {
        Self {
            python_server_url,
            http_client: reqwest::Client::new(),
            timeout_seconds: 30,
            max_retries: 3,
        }
    }
    
    /// Send feature packet to Python and receive generated code
    pub async fn send_features_get_generation(
        &self,
        packet: &FeaturePacket,
    ) -> Result<GeneratedWebsitePacket> {
        let endpoint = format!("{}/api/v1/generate", self.python_server_url);
        
        let mut attempt = 0;
        loop {
            match self.try_request(&endpoint, packet).await {
                Ok(response) => return Ok(response),
                Err(e) if attempt < self.max_retries => {
                    attempt += 1;
                    log::warn!(
                        "Request attempt {} failed: {}. Retrying...",
                        attempt,
                        e
                    );
                    tokio::time::sleep(tokio::time::Duration::from_secs(2_u64.pow(attempt as u32)))
                        .await;
                }
                Err(e) => return Err(e).context("Failed to send features to Python after retries"),
            }
        }
    }
    
    /// Send training feedback to Python for online learning
    pub async fn send_feedback(&self, packet: &FeaturePacket) -> Result<()> {
        let endpoint = format!("{}/api/v1/feedback", self.python_server_url);
        
        self.http_client
            .post(&endpoint)
            .json(packet)
            .timeout(tokio::time::Duration::from_secs(self.timeout_seconds))
            .send()
            .await
            .context("Failed to send feedback to Python")?
            .error_for_status()
            .context("Python server returned error for feedback")?;
        
        Ok(())
    }
    
    /// Check if Python server is healthy
    pub async fn health_check(&self) -> Result<bool> {
        let endpoint = format!("{}/api/v1/health", self.python_server_url);
        
        match self
            .http_client
            .get(&endpoint)
            .timeout(tokio::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false),
        }
    }
    
    async fn try_request(
        &self,
        endpoint: &str,
        packet: &FeaturePacket,
    ) -> Result<GeneratedWebsitePacket> {
        let response = self
            .http_client
            .post(endpoint)
            .json(packet)
            .timeout(tokio::time::Duration::from_secs(self.timeout_seconds))
            .send()
            .await
            .context("HTTP request failed")?;
        
        response
            .error_for_status()
            .context("Python server returned error")?
            .json::<GeneratedWebsitePacket>()
            .await
            .context("Failed to parse response from Python")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_packet_serialization() {
        let packet = FeaturePacket {
            url: "https://example.com".to_string(),
            features: vec![0.1; 48],
            website_intent: "ecommerce".to_string(),
            design_style: "modern".to_string(),
            feedback: None,
            timestamp: 1704067200,
            session_id: "sess123".to_string(),
        };
        
        let json = serde_json::to_string(&packet).unwrap();
        let restored: FeaturePacket = serde_json::from_str(&json).unwrap();
        
        assert_eq!(restored.url, packet.url);
        assert_eq!(restored.features.len(), 48);
    }
    
    #[test]
    fn test_generated_packet_serialization() {
        let packet = GeneratedWebsitePacket {
            html: "<html></html>".to_string(),
            css: "body { color: red; }".to_string(),
            javascript: "console.log('test');".to_string(),
            confidence: 0.95,
            should_use: true,
            training_metrics: None,
            timestamp: 1704067200,
        };
        
        let json = serde_json::to_string(&packet).unwrap();
        let restored: GeneratedWebsitePacket = serde_json::from_str(&json).unwrap();
        
        assert_eq!(restored.html, packet.html);
        assert!(restored.confidence > 0.9);
    }
    
    #[test]
    fn test_feedback_serialization() {
        let feedback = RenderingFeedback {
            quality_score: 0.85,
            matched_elements: 100,
            mismatched_elements: 15,
            css_accuracy: 0.90,
            layout_similarity: 0.88,
            human_feedback: Some("Good layout".to_string()),
        };
        
        let json = serde_json::to_string(&feedback).unwrap();
        let restored: RenderingFeedback = serde_json::from_str(&json).unwrap();
        
        assert_eq!(restored.quality_score, 0.85);
        assert!(restored.human_feedback.is_some());
    }
    
    #[test]
    fn test_training_metrics_serialization() {
        let mut metrics = TrainingMetrics {
            loss: 0.125,
            accuracy: 0.92,
            learning_rate: 0.001,
            epoch: 42,
            latent_dim: 256,
            additional: HashMap::new(),
        };
        
        metrics.additional.insert("precision".to_string(), 0.91);
        metrics.additional.insert("recall".to_string(), 0.93);
        
        let json = serde_json::to_string(&metrics).unwrap();
        let restored: TrainingMetrics = serde_json::from_str(&json).unwrap();
        
        assert_eq!(restored.epoch, 42);
        assert_eq!(restored.additional.len(), 2);
    }
}
