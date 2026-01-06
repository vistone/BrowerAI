/// Online learning system for continuous model improvement
///
/// Enables models to learn from new data without full retraining
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for online learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size for updates
    pub batch_size: usize,
    /// Maximum samples to keep in memory
    pub max_samples: usize,
    /// Minimum samples before update
    pub min_samples_for_update: usize,
    /// Enable automatic updates
    pub auto_update: bool,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            max_samples: 10000,
            min_samples_for_update: 100,
            auto_update: false,
        }
    }
}

/// Training sample for online learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Input features
    pub input: Vec<f32>,
    /// Expected output
    pub output: Vec<f32>,
    /// Sample weight (importance)
    pub weight: f32,
}

impl TrainingSample {
    pub fn new(input: Vec<f32>, output: Vec<f32>) -> Self {
        Self {
            input,
            output,
            weight: 1.0,
        }
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.max(0.0);
        self
    }
}

/// Online learner that updates models incrementally
pub struct OnlineLearner {
    /// Learning configuration
    config: LearningConfig,
    /// Buffer of training samples
    sample_buffer: VecDeque<TrainingSample>,
    /// Total samples processed
    total_samples: usize,
    /// Number of updates performed
    update_count: usize,
}

impl OnlineLearner {
    /// Create a new online learner
    pub fn new(config: LearningConfig) -> Self {
        Self {
            config,
            sample_buffer: VecDeque::new(),
            total_samples: 0,
            update_count: 0,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(LearningConfig::default())
    }

    /// Add a training sample
    pub fn add_sample(&mut self, sample: TrainingSample) {
        self.sample_buffer.push_back(sample);
        self.total_samples += 1;

        // Remove oldest samples if buffer is full
        while self.sample_buffer.len() > self.config.max_samples {
            self.sample_buffer.pop_front();
        }

        // Trigger update if auto_update is enabled
        if self.config.auto_update && self.should_update() {
            let _ = self.trigger_update();
        }
    }

    /// Check if update should be triggered
    pub fn should_update(&self) -> bool {
        self.sample_buffer.len() >= self.config.min_samples_for_update
    }

    /// Trigger a model update (stub implementation)
    pub fn trigger_update(&mut self) -> Result<(), String> {
        if !self.should_update() {
            return Err("Not enough samples for update".to_string());
        }

        // In a real implementation, this would:
        // 1. Sample a batch from the buffer
        // 2. Compute gradients
        // 3. Update model weights
        // 4. Validate the update

        self.update_count += 1;
        Ok(())
    }

    /// Get learning statistics
    pub fn get_stats(&self) -> LearningStats {
        LearningStats {
            total_samples: self.total_samples,
            buffered_samples: self.sample_buffer.len(),
            update_count: self.update_count,
            learning_rate: self.config.learning_rate,
        }
    }

    /// Clear the sample buffer
    pub fn clear_buffer(&mut self) {
        self.sample_buffer.clear();
    }

    /// Update learning rate
    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        self.config.learning_rate = learning_rate.max(0.0);
    }

    /// Get current learning rate
    pub fn get_learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    /// Get number of buffered samples
    pub fn buffer_size(&self) -> usize {
        self.sample_buffer.len()
    }
}

/// Statistics for online learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStats {
    pub total_samples: usize,
    pub buffered_samples: usize,
    pub update_count: usize,
    pub learning_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_config_default() {
        let config = LearningConfig::default();
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.batch_size, 32);
        assert!(!config.auto_update);
    }

    #[test]
    fn test_training_sample_creation() {
        let sample = TrainingSample::new(vec![1.0, 2.0], vec![3.0]);
        assert_eq!(sample.input, vec![1.0, 2.0]);
        assert_eq!(sample.output, vec![3.0]);
        assert_eq!(sample.weight, 1.0);
    }

    #[test]
    fn test_training_sample_with_weight() {
        let sample = TrainingSample::new(vec![1.0], vec![2.0]).with_weight(0.5);
        assert_eq!(sample.weight, 0.5);
    }

    #[test]
    fn test_online_learner_creation() {
        let learner = OnlineLearner::with_defaults();
        assert_eq!(learner.buffer_size(), 0);
        assert_eq!(learner.total_samples, 0);
    }

    #[test]
    fn test_online_learner_add_sample() {
        let mut learner = OnlineLearner::with_defaults();
        let sample = TrainingSample::new(vec![1.0], vec![2.0]);

        learner.add_sample(sample);
        assert_eq!(learner.buffer_size(), 1);
        assert_eq!(learner.total_samples, 1);
    }

    #[test]
    fn test_online_learner_buffer_limit() {
        let mut config = LearningConfig::default();
        config.max_samples = 5;
        let mut learner = OnlineLearner::new(config);

        for i in 0..10 {
            learner.add_sample(TrainingSample::new(vec![i as f32], vec![0.0]));
        }

        assert_eq!(learner.buffer_size(), 5);
        assert_eq!(learner.total_samples, 10);
    }

    #[test]
    fn test_online_learner_should_update() {
        let mut config = LearningConfig::default();
        config.min_samples_for_update = 5;
        let mut learner = OnlineLearner::new(config);

        assert!(!learner.should_update());

        for i in 0..5 {
            learner.add_sample(TrainingSample::new(vec![i as f32], vec![0.0]));
        }

        assert!(learner.should_update());
    }

    #[test]
    fn test_online_learner_trigger_update() {
        let mut config = LearningConfig::default();
        config.min_samples_for_update = 2;
        let mut learner = OnlineLearner::new(config);

        learner.add_sample(TrainingSample::new(vec![1.0], vec![2.0]));
        learner.add_sample(TrainingSample::new(vec![3.0], vec![4.0]));

        let result = learner.trigger_update();
        assert!(result.is_ok());
        assert_eq!(learner.update_count, 1);
    }

    #[test]
    fn test_online_learner_stats() {
        let mut learner = OnlineLearner::with_defaults();
        learner.add_sample(TrainingSample::new(vec![1.0], vec![2.0]));

        let stats = learner.get_stats();
        assert_eq!(stats.total_samples, 1);
        assert_eq!(stats.buffered_samples, 1);
        assert_eq!(stats.update_count, 0);
    }

    #[test]
    fn test_online_learner_learning_rate() {
        let mut learner = OnlineLearner::with_defaults();

        learner.set_learning_rate(0.01);
        assert_eq!(learner.get_learning_rate(), 0.01);

        // Test negative clamping
        learner.set_learning_rate(-0.5);
        assert_eq!(learner.get_learning_rate(), 0.0);
    }

    #[test]
    fn test_online_learner_clear_buffer() {
        let mut learner = OnlineLearner::with_defaults();
        learner.add_sample(TrainingSample::new(vec![1.0], vec![2.0]));

        assert_eq!(learner.buffer_size(), 1);
        learner.clear_buffer();
        assert_eq!(learner.buffer_size(), 0);
    }
}
