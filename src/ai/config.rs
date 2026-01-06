/// AI Configuration and Fallback Tracking
/// Implements M1 of AI-Centric Execution Refresh:
/// - Configurable AI enable/disable switch
/// - Fallback reason tracking
/// - Model selection logging
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Global AI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiConfig {
    /// Enable AI enhancement globally
    pub enable_ai: bool,
    /// Enable fallback to baseline when AI fails
    pub enable_fallback: bool,
    /// Log AI decisions and performance
    pub enable_logging: bool,
    /// Maximum inference time before fallback (milliseconds)
    pub max_inference_time_ms: u64,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            enable_ai: true,
            enable_fallback: true,
            enable_logging: true,
            max_inference_time_ms: 100,
        }
    }
}

/// Reasons why AI fallback occurred
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FallbackReason {
    /// AI is disabled in configuration
    AiDisabled,
    /// Model file not found
    ModelNotFound(String),
    /// Model failed to load
    ModelLoadFailed(String),
    /// Inference failed with error
    InferenceFailed(String),
    /// Inference exceeded time limit
    TimeoutExceeded { actual_ms: u64, limit_ms: u64 },
    /// Model health check failed
    ModelUnhealthy(String),
    /// No suitable model available
    NoModelAvailable,
}

impl std::fmt::Display for FallbackReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FallbackReason::AiDisabled => write!(f, "AI disabled"),
            FallbackReason::ModelNotFound(name) => write!(f, "Model not found: {}", name),
            FallbackReason::ModelLoadFailed(err) => write!(f, "Model load failed: {}", err),
            FallbackReason::InferenceFailed(err) => write!(f, "Inference failed: {}", err),
            FallbackReason::TimeoutExceeded {
                actual_ms,
                limit_ms,
            } => {
                write!(f, "Timeout exceeded: {}ms > {}ms", actual_ms, limit_ms)
            }
            FallbackReason::ModelUnhealthy(reason) => write!(f, "Model unhealthy: {}", reason),
            FallbackReason::NoModelAvailable => write!(f, "No model available"),
        }
    }
}

/// AI operation statistics
#[derive(Debug, Clone, Default)]
pub struct AiStats {
    /// Total AI operations attempted
    pub total_attempts: u64,
    /// Successful AI operations
    pub successful: u64,
    /// Operations that fell back to baseline
    pub fallback_count: u64,
    /// Total inference time (milliseconds)
    pub total_inference_ms: u64,
}

impl AiStats {
    /// Get success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_attempts == 0 {
            0.0
        } else {
            self.successful as f64 / self.total_attempts as f64
        }
    }

    /// Get fallback rate (0.0 to 1.0)
    pub fn fallback_rate(&self) -> f64 {
        if self.total_attempts == 0 {
            0.0
        } else {
            self.fallback_count as f64 / self.total_attempts as f64
        }
    }

    /// Get average inference time (milliseconds)
    pub fn avg_inference_ms(&self) -> f64 {
        if self.successful == 0 {
            0.0
        } else {
            self.total_inference_ms as f64 / self.successful as f64
        }
    }
}

/// Tracker for AI fallback events
#[derive(Debug, Clone)]
pub struct FallbackTracker {
    stats: Arc<RwLock<AiStats>>,
    recent_fallbacks: Arc<RwLock<Vec<(Instant, FallbackReason)>>>,
    max_recent: usize,
}

impl Default for FallbackTracker {
    fn default() -> Self {
        Self::new(100)
    }
}

impl FallbackTracker {
    /// Create a new fallback tracker
    pub fn new(max_recent: usize) -> Self {
        Self {
            stats: Arc::new(RwLock::new(AiStats::default())),
            recent_fallbacks: Arc::new(RwLock::new(Vec::new())),
            max_recent,
        }
    }

    /// Record an AI attempt
    pub fn record_attempt(&self) {
        if let Ok(mut stats) = self.stats.write() {
            stats.total_attempts += 1;
        }
    }

    /// Record a successful AI operation
    pub fn record_success(&self, inference_ms: u64) {
        if let Ok(mut stats) = self.stats.write() {
            stats.successful += 1;
            stats.total_inference_ms += inference_ms;
        }
    }

    /// Record a fallback event
    pub fn record_fallback(&self, reason: FallbackReason) {
        if let Ok(mut stats) = self.stats.write() {
            stats.fallback_count += 1;
        }

        if let Ok(mut fallbacks) = self.recent_fallbacks.write() {
            fallbacks.push((Instant::now(), reason));
            // Keep only the most recent fallbacks
            if fallbacks.len() > self.max_recent {
                fallbacks.remove(0);
            }
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> AiStats {
        self.stats.read().unwrap().clone()
    }

    /// Get recent fallback reasons
    pub fn get_recent_fallbacks(&self) -> Vec<(Instant, FallbackReason)> {
        self.recent_fallbacks.read().unwrap().clone()
    }

    /// Clear all statistics
    pub fn clear(&self) {
        if let Ok(mut stats) = self.stats.write() {
            *stats = AiStats::default();
        }
        if let Ok(mut fallbacks) = self.recent_fallbacks.write() {
            fallbacks.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_ai_config_default() {
        let config = AiConfig::default();
        assert!(config.enable_ai);
        assert!(config.enable_fallback);
        assert!(config.enable_logging);
        assert_eq!(config.max_inference_time_ms, 100);
    }

    #[test]
    fn test_fallback_reason_display() {
        let reason = FallbackReason::AiDisabled;
        assert_eq!(reason.to_string(), "AI disabled");

        let reason = FallbackReason::ModelNotFound("test.onnx".to_string());
        assert_eq!(reason.to_string(), "Model not found: test.onnx");

        let reason = FallbackReason::TimeoutExceeded {
            actual_ms: 150,
            limit_ms: 100,
        };
        assert_eq!(reason.to_string(), "Timeout exceeded: 150ms > 100ms");
    }

    #[test]
    fn test_ai_stats_empty() {
        let stats = AiStats::default();
        assert_eq!(stats.success_rate(), 0.0);
        assert_eq!(stats.fallback_rate(), 0.0);
        assert_eq!(stats.avg_inference_ms(), 0.0);
    }

    #[test]
    fn test_ai_stats_calculations() {
        let stats = AiStats {
            total_attempts: 100,
            successful: 80,
            fallback_count: 20,
            total_inference_ms: 800,
        };
        assert_eq!(stats.success_rate(), 0.8);
        assert_eq!(stats.fallback_rate(), 0.2);
        assert_eq!(stats.avg_inference_ms(), 10.0);
    }

    #[test]
    fn test_fallback_tracker_record_attempt() {
        let tracker = FallbackTracker::new(10);
        tracker.record_attempt();
        tracker.record_attempt();
        let stats = tracker.get_stats();
        assert_eq!(stats.total_attempts, 2);
    }

    #[test]
    fn test_fallback_tracker_record_success() {
        let tracker = FallbackTracker::new(10);
        tracker.record_attempt();
        tracker.record_success(10);
        let stats = tracker.get_stats();
        assert_eq!(stats.successful, 1);
        assert_eq!(stats.total_inference_ms, 10);
    }

    #[test]
    fn test_fallback_tracker_record_fallback() {
        let tracker = FallbackTracker::new(10);
        tracker.record_attempt();
        tracker.record_fallback(FallbackReason::AiDisabled);
        let stats = tracker.get_stats();
        assert_eq!(stats.fallback_count, 1);
        let fallbacks = tracker.get_recent_fallbacks();
        assert_eq!(fallbacks.len(), 1);
        assert_eq!(fallbacks[0].1, FallbackReason::AiDisabled);
    }

    #[test]
    fn test_fallback_tracker_max_recent() {
        let tracker = FallbackTracker::new(3);
        for i in 0..5 {
            tracker.record_fallback(FallbackReason::ModelNotFound(format!("model{}", i)));
        }
        let fallbacks = tracker.get_recent_fallbacks();
        assert_eq!(fallbacks.len(), 3);
        // Should keep the last 3
        match &fallbacks[0].1 {
            FallbackReason::ModelNotFound(name) => assert_eq!(name, "model2"),
            _ => panic!("Expected ModelNotFound"),
        }
    }

    #[test]
    fn test_fallback_tracker_clear() {
        let tracker = FallbackTracker::new(10);
        tracker.record_attempt();
        tracker.record_success(10);
        tracker.record_fallback(FallbackReason::AiDisabled);
        tracker.clear();
        let stats = tracker.get_stats();
        assert_eq!(stats.total_attempts, 0);
        assert_eq!(stats.successful, 0);
        assert_eq!(stats.fallback_count, 0);
        let fallbacks = tracker.get_recent_fallbacks();
        assert_eq!(fallbacks.len(), 0);
    }

    #[test]
    fn test_fallback_tracker_concurrent() {
        let tracker = Arc::new(FallbackTracker::new(100));
        let mut handles = vec![];

        for _ in 0..10 {
            let tracker_clone = Arc::clone(&tracker);
            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    tracker_clone.record_attempt();
                    thread::sleep(Duration::from_micros(1));
                    tracker_clone.record_success(5);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = tracker.get_stats();
        assert_eq!(stats.total_attempts, 100);
        assert_eq!(stats.successful, 100);
    }
}
