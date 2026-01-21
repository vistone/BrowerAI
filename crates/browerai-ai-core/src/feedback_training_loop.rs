/// Complete Feedback Training Loop
///
/// This module implements the full cycle:
/// 1. Collect user feedback on rendered pages
/// 2. Extract training signals
/// 3. Train the ONNX model
/// 4. Evaluate improvements
/// 5. Deploy updated model
///
/// Real-world usage: Continuously improve rendering based on user interactions
use crate::trainable_model::TrainableOnnxModel;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// User feedback on rendered layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutFeedback {
    /// Session ID
    pub session_id: String,
    /// Original page URL
    pub url: String,
    /// User satisfaction score (1-5)
    pub satisfaction_score: u8,
    /// Time spent on page (seconds)
    pub time_on_page: u64,
    /// Click events (element IDs)
    pub clicks: Vec<String>,
    /// Form submissions
    pub form_submissions: Vec<String>,
    /// Errors encountered
    pub errors: Vec<String>,
    /// Free-form comment
    pub comment: Option<String>,
    /// Timestamp
    pub timestamp: i64,
}

/// Training signal extracted from feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSignal {
    /// Input features
    pub features: Vec<f32>,
    /// Target label (success/failure)
    pub target: Vec<f32>,
    /// Weight (higher = more important)
    pub weight: f32,
    /// Source feedback ID
    pub feedback_id: String,
}

/// Training batch
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    /// Signals in this batch
    pub signals: Vec<TrainingSignal>,
    /// Average satisfaction
    pub avg_satisfaction: f32,
    /// Success rate
    pub success_rate: f32,
}

/// Feedback Training Loop
pub struct FeedbackTrainingLoop {
    model: TrainableOnnxModel,
    feedback_buffer: Vec<LayoutFeedback>,
    training_history: Vec<TrainingBatch>,
    config: FeedbackLoopConfig,
}

/// Configuration for feedback training loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackLoopConfig {
    /// Minimum feedback samples before training
    pub min_samples_before_training: usize,
    /// Maximum feedback buffer size
    pub max_buffer_size: usize,
    /// Training interval (number of batches)
    pub training_interval: usize,
    /// Auto-deploy threshold (improvement %)
    pub auto_deploy_threshold: f32,
}

impl Default for FeedbackLoopConfig {
    fn default() -> Self {
        Self {
            min_samples_before_training: 50,
            max_buffer_size: 10000,
            training_interval: 100,
            auto_deploy_threshold: 5.0, // 5% improvement
        }
    }
}

impl FeedbackTrainingLoop {
    /// Create new feedback training loop
    pub fn new(model: TrainableOnnxModel, config: FeedbackLoopConfig) -> Self {
        Self {
            model,
            feedback_buffer: Vec::new(),
            training_history: Vec::new(),
            config,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Result<Self> {
        let model = TrainableOnnxModel::with_defaults()?;
        Ok(Self::new(model, FeedbackLoopConfig::default()))
    }

    /// Add user feedback
    pub fn add_feedback(&mut self, feedback: LayoutFeedback) -> Result<()> {
        // Validate feedback
        if feedback.satisfaction_score < 1 || feedback.satisfaction_score > 5 {
            anyhow::bail!("Satisfaction score must be 1-5");
        }

        log::debug!(
            "Added feedback from session {}: satisfaction = {}",
            feedback.session_id,
            feedback.satisfaction_score
        );

        self.feedback_buffer.push(feedback);

        // Manage buffer size
        if self.feedback_buffer.len() > self.config.max_buffer_size {
            self.feedback_buffer.remove(0);
        }

        // Check if we should train
        if self.feedback_buffer.len() >= self.config.min_samples_before_training {
            self.maybe_train()?;
        }

        Ok(())
    }

    /// Extract training signals from feedback batch
    fn extract_signals(&self, feedback_batch: &[LayoutFeedback]) -> Vec<TrainingSignal> {
        let mut signals = Vec::new();

        for feedback in feedback_batch {
            // Feature engineering
            let features = self.extract_features(feedback);

            // Target: success (satisfaction >= 4) or failure (< 4)
            let target = if feedback.satisfaction_score >= 4 {
                vec![1.0, 0.0] // Success
            } else {
                vec![0.0, 1.0] // Failure
            };

            // Weight: higher satisfaction = higher weight
            let weight = (feedback.satisfaction_score as f32) / 5.0;

            signals.push(TrainingSignal {
                features,
                target,
                weight,
                feedback_id: feedback.session_id.clone(),
            });
        }

        signals
    }

    /// Extract features from feedback
    fn extract_features(&self, feedback: &LayoutFeedback) -> Vec<f32> {
        vec![
            feedback.satisfaction_score as f32 / 5.0, // Normalized satisfaction
            (feedback.time_on_page as f32).min(3600.0) / 3600.0, // Time (capped at 1 hour)
            feedback.clicks.len() as f32 / 50.0,      // Click density
            feedback.form_submissions.len() as f32,   // Form submissions
            feedback.errors.len() as f32,             // Error count
            if feedback.comment.is_some() { 1.0 } else { 0.0 }, // Has comment
            // Padding for fixed-size feature vector
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    }

    /// Maybe train if conditions are met
    fn maybe_train(&mut self) -> Result<()> {
        if self.feedback_buffer.len() < self.config.min_samples_before_training {
            return Ok(());
        }

        log::info!(
            "Training triggered: {} feedback samples accumulated",
            self.feedback_buffer.len()
        );

        // Extract batch
        let batch_size = self
            .config
            .min_samples_before_training
            .min(self.feedback_buffer.len());
        let batch: Vec<LayoutFeedback> = self.feedback_buffer.drain(0..batch_size).collect();

        self.train_from_batch(&batch)?;

        Ok(())
    }

    /// Train from feedback batch
    fn train_from_batch(&mut self, batch: &[LayoutFeedback]) -> Result<()> {
        log::info!("Starting training on {} feedback samples", batch.len());

        // Extract signals
        let signals = self.extract_signals(batch);

        // Prepare training data
        let inputs: Vec<Vec<f32>> = signals.iter().map(|s| s.features.clone()).collect();
        let targets: Vec<Vec<f32>> = signals.iter().map(|s| s.target.clone()).collect();

        // Train
        let loss = self.model.train_batch(&inputs, &targets)?;

        log::info!("Training completed. Loss: {:.4}", loss);

        // Calculate metrics
        let avg_satisfaction: f32 = batch
            .iter()
            .map(|f| f.satisfaction_score as f32)
            .sum::<f32>()
            / batch.len() as f32;
        let success_rate = (avg_satisfaction - 1.0) / 4.0; // Normalize to 0-1

        // Record batch
        let training_batch = TrainingBatch {
            signals,
            avg_satisfaction,
            success_rate,
        };

        self.training_history.push(training_batch);

        // Check if should auto-deploy
        self.maybe_auto_deploy()?;

        Ok(())
    }

    /// Check if model improvement warrants auto-deployment
    fn maybe_auto_deploy(&mut self) -> Result<()> {
        if self.training_history.len() < 2 {
            return Ok(());
        }

        let latest = &self.training_history[self.training_history.len() - 1];
        let previous = &self.training_history[self.training_history.len() - 2];

        let improvement_pct =
            ((latest.success_rate - previous.success_rate) / previous.success_rate) * 100.0;

        if improvement_pct > self.config.auto_deploy_threshold {
            log::info!(
                "Auto-deploying model: {:.2}% improvement detected",
                improvement_pct
            );
            self.model.save_checkpoint()?;
        }

        Ok(())
    }

    /// Get training statistics
    pub fn get_stats(&self) -> FeedbackLoopStats {
        let feedback_count = self.feedback_buffer.len() + self.training_history.len() * 50;

        let avg_satisfaction = if !self.training_history.is_empty() {
            self.training_history
                .iter()
                .map(|b| b.avg_satisfaction)
                .sum::<f32>()
                / self.training_history.len() as f32
        } else {
            0.0
        };

        let total_improvements = self.training_history.len() as u64;

        FeedbackLoopStats {
            feedback_collected: feedback_count,
            training_rounds: self.training_history.len(),
            avg_satisfaction,
            total_improvements,
            buffer_size: self.feedback_buffer.len(),
            model_stats: self.model.get_stats(),
        }
    }

    /// Clear training history
    pub fn reset(&mut self) {
        self.feedback_buffer.clear();
        self.training_history.clear();
        self.model.reset_stats();
    }
}

/// Statistics from feedback training loop
#[derive(Debug, Clone, Serialize)]
pub struct FeedbackLoopStats {
    pub feedback_collected: usize,
    pub training_rounds: usize,
    pub avg_satisfaction: f32,
    pub total_improvements: u64,
    pub buffer_size: usize,
    pub model_stats: crate::trainable_model::TrainingStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_creation() {
        let feedback = LayoutFeedback {
            session_id: "sess-001".to_string(),
            url: "https://example.com".to_string(),
            satisfaction_score: 5,
            time_on_page: 120,
            clicks: vec!["button-1".to_string()],
            form_submissions: vec![],
            errors: vec![],
            comment: Some("Great layout!".to_string()),
            timestamp: 0,
        };

        assert_eq!(feedback.satisfaction_score, 5);
        assert_eq!(feedback.clicks.len(), 1);
    }

    #[test]
    fn test_training_loop_creation() {
        let loop_result = FeedbackTrainingLoop::with_defaults();
        assert!(loop_result.is_ok());
    }

    #[test]
    fn test_add_feedback() {
        let mut loop_inst = FeedbackTrainingLoop::with_defaults().unwrap();

        let feedback = LayoutFeedback {
            session_id: "sess-001".to_string(),
            url: "https://example.com".to_string(),
            satisfaction_score: 4,
            time_on_page: 120,
            clicks: vec!["button-1".to_string()],
            form_submissions: vec!["form-1".to_string()],
            errors: vec![],
            comment: None,
            timestamp: chrono::Utc::now().timestamp(),
        };

        let result = loop_inst.add_feedback(feedback);
        assert!(result.is_ok());
        assert_eq!(loop_inst.feedback_buffer.len(), 1);
    }

    #[test]
    fn test_invalid_satisfaction_score() {
        let mut loop_inst = FeedbackTrainingLoop::with_defaults().unwrap();

        let feedback = LayoutFeedback {
            session_id: "sess-001".to_string(),
            url: "https://example.com".to_string(),
            satisfaction_score: 6, // Invalid: > 5
            time_on_page: 120,
            clicks: vec![],
            form_submissions: vec![],
            errors: vec![],
            comment: None,
            timestamp: 0,
        };

        let result = loop_inst.add_feedback(feedback);
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_extraction() {
        let loop_inst = FeedbackTrainingLoop::with_defaults().unwrap();

        let feedback = LayoutFeedback {
            session_id: "sess-001".to_string(),
            url: "https://example.com".to_string(),
            satisfaction_score: 5,
            time_on_page: 300,
            clicks: vec!["btn1".to_string(), "btn2".to_string()],
            form_submissions: vec!["form1".to_string()],
            errors: vec![],
            comment: Some("Good".to_string()),
            timestamp: 0,
        };

        let features = loop_inst.extract_features(&feedback);
        assert_eq!(features.len(), 14); // Fixed size
        assert!(features[0] > 0.9); // High satisfaction
    }

    #[test]
    fn test_signal_extraction() {
        let loop_inst = FeedbackTrainingLoop::with_defaults().unwrap();

        let feedback_batch = vec![LayoutFeedback {
            session_id: "sess-001".to_string(),
            url: "https://example.com".to_string(),
            satisfaction_score: 5,
            time_on_page: 120,
            clicks: vec![],
            form_submissions: vec![],
            errors: vec![],
            comment: None,
            timestamp: 0,
        }];

        let signals = loop_inst.extract_signals(&feedback_batch);
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].target, vec![1.0, 0.0]); // Success
        assert!(signals[0].weight > 0.9);
    }

    #[test]
    fn test_stats_generation() {
        let loop_inst = FeedbackTrainingLoop::with_defaults().unwrap();
        let stats = loop_inst.get_stats();

        assert_eq!(stats.feedback_collected, 0);
        assert_eq!(stats.training_rounds, 0);
        assert_eq!(stats.buffer_size, 0);
    }
}
