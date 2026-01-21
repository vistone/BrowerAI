/// Online learning system for continuous model improvement
///
/// Enables models to learn from new data without full retraining
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Model weights container for gradient descent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeights {
    /// Weight vector
    pub weights: Vec<f32>,
    /// Bias term
    pub bias: f32,
    /// Last update timestamp
    pub last_update: i64,
    /// Momentum buffer for accelerated gradient descent
    #[serde(skip)]
    momentum_buffer: Option<Vec<f32>>,
}

impl ModelWeights {
    /// Create new model weights with given dimension
    pub fn new(size: usize) -> Self {
        Self {
            weights: vec![0.0; size],
            bias: 0.0,
            last_update: chrono::Utc::now().timestamp(),
            momentum_buffer: None,
        }
    }

    /// Get a copy of current weights
    pub fn get_weights(&self) -> (Vec<f32>, f32) {
        (self.weights.clone(), self.bias)
    }

    /// Update weights using gradient descent with optional momentum
    pub fn update_with_momentum(
        &mut self,
        gradients: &[f32],
        gradient_bias: f32,
        learning_rate: f32,
        momentum: f32,
    ) -> Result<()> {
        if gradients.len() != self.weights.len() {
            return Err(anyhow::anyhow!(
                "Gradient size {} != weight size {}",
                gradients.len(),
                self.weights.len()
            ));
        }

        // Initialize or update momentum buffer
        if momentum > 0.0 {
            if self.momentum_buffer.is_none() {
                self.momentum_buffer = Some(vec![0.0; self.weights.len()]);
            }

            if let Some(buffer) = &mut self.momentum_buffer {
                for (i, (w, g)) in self.weights.iter_mut().zip(gradients.iter()).enumerate() {
                    // Momentum: v = momentum * v - learning_rate * g
                    buffer[i] = momentum * buffer[i] - learning_rate * g;
                    *w += buffer[i];
                }
            }
        } else {
            // Standard gradient descent
            for (w, g) in self.weights.iter_mut().zip(gradients.iter()) {
                *w -= learning_rate * g;
            }
        }

        self.bias -= learning_rate * gradient_bias;
        self.last_update = chrono::Utc::now().timestamp();
        Ok(())
    }

    /// Update weights using simple gradient descent (no momentum)
    pub fn update(
        &mut self,
        gradients: &[f32],
        gradient_bias: f32,
        learning_rate: f32,
    ) -> Result<()> {
        self.update_with_momentum(gradients, gradient_bias, learning_rate, 0.0)
    }

    /// Compute L2 norm of weights (for regularization)
    pub fn l2_norm(&self) -> f32 {
        self.weights.iter().map(|w| w * w).sum::<f32>().sqrt()
    }

    /// Apply L2 regularization to weights
    pub fn apply_l2_regularization(&mut self, l2_coeff: f32) {
        let scale = 1.0 - l2_coeff;
        for w in &mut self.weights {
            *w *= scale;
        }
    }
}

/// Configuration for online learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Learning rate (typical: 0.001)
    pub learning_rate: f32,
    /// Batch size for updates (typical: 32)
    pub batch_size: usize,
    /// Maximum samples to keep in memory
    pub max_samples: usize,
    /// Minimum samples before update (typical: 100)
    pub min_samples_for_update: usize,
    /// Enable automatic updates when threshold reached
    pub auto_update: bool,
    /// L2 regularization coefficient (typical: 0.0001)
    pub l2_regularization: f32,
    /// Momentum factor for gradient updates (0.0 = no momentum, 0.9 = high momentum)
    pub momentum: f32,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            max_samples: 10000,
            min_samples_for_update: 100,
            auto_update: false,
            l2_regularization: 0.0001,
            momentum: 0.9,
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
    /// Sample weight (importance, range: 0.0-1.0)
    pub weight: f32,
}

impl TrainingSample {
    /// Create new training sample with unit weight
    pub fn new(input: Vec<f32>, output: Vec<f32>) -> Self {
        Self {
            input,
            output,
            weight: 1.0,
        }
    }

    /// Set sample weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.max(0.0);
        self
    }

    /// Compute input dimension
    pub fn input_dim(&self) -> usize {
        self.input.len()
    }

    /// Compute output dimension
    pub fn output_dim(&self) -> usize {
        self.output.len()
    }
}

/// Gradient information from batch processing
#[derive(Debug, Clone)]
pub struct BatchGradients {
    /// Averaged gradients
    pub gradients: Vec<f32>,
    /// Averaged gradient for bias
    pub gradient_bias: f32,
    /// Batch loss (for monitoring)
    pub loss: f32,
    /// Number of samples in batch
    pub batch_size: usize,
}

impl BatchGradients {
    /// Create new batch gradients
    pub fn new(gradients: Vec<f32>, gradient_bias: f32, loss: f32, batch_size: usize) -> Self {
        Self {
            gradients,
            gradient_bias,
            loss,
            batch_size,
        }
    }
}

/// Online learner that updates models incrementally
pub struct OnlineLearner {
    /// Learning configuration
    config: LearningConfig,
    /// Buffer of training samples
    sample_buffer: VecDeque<TrainingSample>,
    /// Model weights
    model_weights: Option<ModelWeights>,
    /// Total samples processed
    total_samples: usize,
    /// Number of updates performed
    update_count: usize,
    /// Total loss accumulated
    total_loss: f32,
}

impl OnlineLearner {
    /// Create a new online learner
    pub fn new(config: LearningConfig) -> Self {
        Self {
            config,
            sample_buffer: VecDeque::new(),
            model_weights: None,
            total_samples: 0,
            update_count: 0,
            total_loss: 0.0,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(LearningConfig::default())
    }

    /// Initialize model weights with given dimension
    pub fn initialize_weights(&mut self, input_dim: usize) {
        self.model_weights = Some(ModelWeights::new(input_dim));
    }

    /// Add a training sample
    pub fn add_sample(&mut self, sample: TrainingSample) -> Result<()> {
        // Initialize weights if needed
        if self.model_weights.is_none() {
            self.initialize_weights(sample.input_dim());
        }

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

        Ok(())
    }

    /// Check if update should be triggered
    pub fn should_update(&self) -> bool {
        self.sample_buffer.len() >= self.config.min_samples_for_update
    }

    /// Compute gradients from a batch of samples
    pub fn compute_batch_gradients(&self, samples: &[TrainingSample]) -> Result<BatchGradients> {
        if samples.is_empty() {
            return Err(anyhow::anyhow!("Cannot compute gradients from empty batch"));
        }

        let input_dim = samples[0].input_dim();
        let mut gradients = vec![0.0; input_dim];
        let mut gradient_bias = 0.0;
        let mut total_loss = 0.0;

        for sample in samples {
            if sample.input_dim() != input_dim {
                return Err(anyhow::anyhow!("Inconsistent sample dimensions"));
            }

            // Compute prediction error (simplified linear regression)
            let predicted = if let Some(weights) = &self.model_weights {
                let mut pred = weights.bias;
                for (x, w) in sample.input.iter().zip(&weights.weights) {
                    pred += x * w;
                }
                pred
            } else {
                0.0
            };

            let error = predicted - sample.output[0]; // Assuming single output
            total_loss += error * error * sample.weight;

            // Accumulate gradients
            for (i, &x) in sample.input.iter().enumerate() {
                gradients[i] += error * x * sample.weight;
            }
            gradient_bias += error * sample.weight;
        }

        // Average gradients by batch size
        let batch_len = samples.len() as f32;
        for g in &mut gradients {
            *g /= batch_len;
        }
        gradient_bias /= batch_len;
        total_loss /= batch_len;

        Ok(BatchGradients::new(
            gradients,
            gradient_bias,
            total_loss,
            samples.len(),
        ))
    }

    /// Trigger a model update using batch gradient descent
    pub fn trigger_update(&mut self) -> Result<()> {
        if !self.should_update() {
            return Err(anyhow::anyhow!("Not enough samples for update"));
        }

        // Sample a batch from the buffer
        let batch_size = self.config.batch_size.min(self.sample_buffer.len());
        let batch: Vec<TrainingSample> = self
            .sample_buffer
            .iter()
            .rev()
            .take(batch_size)
            .cloned()
            .collect();

        // Compute gradients
        let batch_gradients = self.compute_batch_gradients(&batch)?;

        // Update model weights
        if let Some(weights) = &mut self.model_weights {
            weights.update_with_momentum(
                &batch_gradients.gradients,
                batch_gradients.gradient_bias,
                self.config.learning_rate,
                self.config.momentum,
            )?;

            // Apply L2 regularization
            if self.config.l2_regularization > 0.0 {
                weights.apply_l2_regularization(self.config.l2_regularization);
            }
        }

        self.update_count += 1;
        self.total_loss += batch_gradients.loss;

        log::debug!(
            "Online learning update #{}: loss={:.6}, lr={:.4}, batch_size={}",
            self.update_count,
            batch_gradients.loss,
            self.config.learning_rate,
            batch_size
        );

        Ok(())
    }

    /// Get learning statistics
    pub fn get_stats(&self) -> LearningStats {
        let avg_loss = if self.update_count > 0 {
            self.total_loss / self.update_count as f32
        } else {
            0.0
        };

        LearningStats {
            total_samples: self.total_samples,
            buffered_samples: self.sample_buffer.len(),
            update_count: self.update_count,
            learning_rate: self.config.learning_rate,
            average_loss: avg_loss,
            momentum: self.config.momentum,
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

    /// Get model weights (if initialized)
    pub fn get_model_weights(&self) -> Option<(Vec<f32>, f32)> {
        self.model_weights.as_ref().map(|w| w.get_weights())
    }

    /// Set model weights directly (for loading pre-trained models)
    pub fn set_model_weights(&mut self, weights: Vec<f32>, bias: f32) -> Result<()> {
        let mut model_weights = ModelWeights::new(weights.len());
        model_weights.weights = weights;
        model_weights.bias = bias;
        self.model_weights = Some(model_weights);
        Ok(())
    }
}

/// Statistics for online learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStats {
    /// Total samples seen
    pub total_samples: usize,
    /// Currently buffered samples
    pub buffered_samples: usize,
    /// Number of update steps
    pub update_count: usize,
    /// Current learning rate
    pub learning_rate: f32,
    /// Average loss over recent updates
    pub average_loss: f32,
    /// Current momentum factor
    pub momentum: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn test_model_weights_creation() {
        let weights = ModelWeights::new(10);
        assert_eq!(weights.weights.len(), 10);
        assert_eq!(weights.bias, 0.0);
        assert!(weights.momentum_buffer.is_none());
    }

    #[test]
    fn test_model_weights_l2_norm() {
        let mut weights = ModelWeights::new(3);
        weights.weights = vec![3.0, 4.0, 0.0];
        assert_eq!(weights.l2_norm(), 5.0); // sqrt(9 + 16 + 0)
    }

    #[test]
    fn test_model_weights_update() {
        let mut weights = ModelWeights::new(2);
        let gradients = vec![1.0, 2.0];
        let _ = weights.update(&gradients, 0.5, 0.1);
        assert_eq!(weights.weights[0], -0.1);
        assert_eq!(weights.weights[1], -0.2);
        assert_eq!(weights.bias, -0.05);
    }

    #[test]
    fn test_learning_config_default() {
        let config = LearningConfig::default();
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.batch_size, 32);
        assert!(!config.auto_update);
        assert_eq!(config.momentum, 0.9);
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
    fn test_training_sample_negative_weight_clamped() {
        let sample = TrainingSample::new(vec![1.0], vec![2.0]).with_weight(-0.5);
        assert_eq!(sample.weight, 0.0);
    }

    #[test]
    fn test_online_learner_creation() {
        let learner = OnlineLearner::with_defaults();
        assert_eq!(learner.buffer_size(), 0);
        assert_eq!(learner.total_samples, 0);
        assert!(learner.model_weights.is_none());
    }

    #[test]
    fn test_online_learner_initialize_weights() {
        let mut learner = OnlineLearner::with_defaults();
        learner.initialize_weights(5);
        assert!(learner.model_weights.is_some());
        let (weights, bias) = learner.get_model_weights().unwrap();
        assert_eq!(weights.len(), 5);
        assert_eq!(bias, 0.0);
    }

    #[test]
    fn test_online_learner_add_sample() {
        let mut learner = OnlineLearner::with_defaults();
        let sample = TrainingSample::new(vec![1.0], vec![2.0]);
        let _ = learner.add_sample(sample);

        assert_eq!(learner.buffer_size(), 1);
        assert_eq!(learner.total_samples, 1);
        assert!(learner.model_weights.is_some());
    }

    #[rstest]
    #[case(5, 10)]
    #[case(10, 20)]
    #[case(100, 100)]
    fn test_online_learner_buffer_limit(#[case] max_samples: usize, #[case] num_samples: usize) {
        let mut config = LearningConfig::default();
        config.max_samples = max_samples;
        let mut learner = OnlineLearner::new(config);

        for i in 0..num_samples {
            let sample = TrainingSample::new(vec![i as f32], vec![0.0]);
            let _ = learner.add_sample(sample);
        }

        assert_eq!(learner.buffer_size(), max_samples.min(num_samples));
        assert_eq!(learner.total_samples, num_samples);
    }

    #[test]
    fn test_online_learner_should_update() {
        let mut config = LearningConfig::default();
        config.min_samples_for_update = 5;
        let mut learner = OnlineLearner::new(config);

        assert!(!learner.should_update());

        for i in 0..5 {
            let sample = TrainingSample::new(vec![i as f32], vec![0.0]);
            let _ = learner.add_sample(sample);
        }

        assert!(learner.should_update());
    }

    #[test]
    fn test_online_learner_trigger_update() {
        let mut config = LearningConfig::default();
        config.min_samples_for_update = 2;
        let mut learner = OnlineLearner::new(config);

        let _ = learner.add_sample(TrainingSample::new(vec![1.0], vec![2.0]));
        let _ = learner.add_sample(TrainingSample::new(vec![3.0], vec![4.0]));

        let result = learner.trigger_update();
        assert!(result.is_ok());
        assert_eq!(learner.update_count, 1);
    }

    #[test]
    fn test_online_learner_stats() {
        let mut learner = OnlineLearner::with_defaults();
        let _ = learner.add_sample(TrainingSample::new(vec![1.0], vec![2.0]));

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

        learner.set_learning_rate(-0.5);
        assert_eq!(learner.get_learning_rate(), 0.0);
    }

    #[test]
    fn test_online_learner_clear_buffer() {
        let mut learner = OnlineLearner::with_defaults();
        let _ = learner.add_sample(TrainingSample::new(vec![1.0], vec![2.0]));

        assert_eq!(learner.buffer_size(), 1);
        learner.clear_buffer();
        assert_eq!(learner.buffer_size(), 0);
    }

    #[test]
    fn test_compute_batch_gradients() {
        let mut learner = OnlineLearner::with_defaults();
        learner.initialize_weights(2);

        let samples = vec![
            TrainingSample::new(vec![1.0, 2.0], vec![3.0]),
            TrainingSample::new(vec![2.0, 3.0], vec![5.0]),
        ];

        let result = learner.compute_batch_gradients(&samples);
        assert!(result.is_ok());

        let gradients = result.unwrap();
        assert_eq!(gradients.batch_size, 2);
        assert!(gradients.loss >= 0.0);
    }

    #[test]
    fn test_batch_gradients_empty_error() {
        let learner = OnlineLearner::with_defaults();
        let samples: Vec<TrainingSample> = vec![];
        let result = learner.compute_batch_gradients(&samples);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_weights_l2_regularization() {
        let mut weights = ModelWeights::new(3);
        weights.weights = vec![10.0, 20.0, 30.0];

        weights.apply_l2_regularization(0.1);
        assert!(weights.weights[0] < 10.0);
        assert!(weights.weights[1] < 20.0);
        assert!(weights.weights[2] < 30.0);
    }
}
