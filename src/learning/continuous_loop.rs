/// Continuous learning loop for autonomous improvement
/// 
/// Implements the "learn-infer-generate" cycle for continuous improvement

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::{
    CodeGenerator, CodeType, GeneratedCode, JsDeobfuscator, OnlineLearner,
    FeedbackCollector, LearningConfig,
};

/// Continuous learning loop configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousLearningConfig {
    /// Learning rate adjustment
    pub learning_rate: f32,
    /// Batch size for updates
    pub batch_size: usize,
    /// Update interval in seconds
    pub update_interval_secs: u64,
    /// Auto-generate samples
    pub auto_generate: bool,
    /// Maximum learning iterations
    pub max_iterations: Option<usize>,
}

impl Default for ContinuousLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            update_interval_secs: 60,
            auto_generate: true,
            max_iterations: None,
        }
    }
}

/// Continuous learning loop statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningLoopStats {
    /// Total iterations completed
    pub iterations: usize,
    /// Total samples processed
    pub samples_processed: usize,
    /// Model updates performed
    pub updates_performed: usize,
    /// Codes generated
    pub codes_generated: usize,
    /// Average learning time per iteration (ms)
    pub avg_iteration_time_ms: f64,
    /// Success rate (0.0-1.0)
    pub success_rate: f32,
}

/// Learning loop event
#[derive(Debug, Clone)]
pub enum LearningEvent {
    /// Iteration started
    IterationStarted(usize),
    /// Iteration completed
    IterationCompleted(usize, Duration),
    /// Sample collected
    SampleCollected(String),
    /// Model updated
    ModelUpdated(usize),
    /// Code generated
    CodeGenerated(CodeType),
    /// Error occurred
    Error(String),
}

/// Continuous learning loop
pub struct ContinuousLearningLoop {
    config: ContinuousLearningConfig,
    learner: OnlineLearner,
    generator: CodeGenerator,
    deobfuscator: JsDeobfuscator,
    feedback: FeedbackCollector,
    stats: LearningLoopStats,
    running: bool,
}

impl ContinuousLearningLoop {
    /// Create a new continuous learning loop
    pub fn new(config: ContinuousLearningConfig) -> Self {
        let learning_config = LearningConfig {
            learning_rate: config.learning_rate,
            batch_size: config.batch_size,
            max_samples: 10000,
            min_samples_for_update: config.batch_size,
            auto_update: true,
        };

        Self {
            config,
            learner: OnlineLearner::new(learning_config),
            generator: CodeGenerator::with_defaults(),
            deobfuscator: JsDeobfuscator::new(),
            feedback: FeedbackCollector::new(),
            stats: LearningLoopStats {
                iterations: 0,
                samples_processed: 0,
                updates_performed: 0,
                codes_generated: 0,
                avg_iteration_time_ms: 0.0,
                success_rate: 1.0,
            },
            running: false,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ContinuousLearningConfig::default())
    }

    /// Run one iteration of the learning loop
    pub fn run_iteration(&mut self) -> Result<Vec<LearningEvent>> {
        let start = Instant::now();
        let mut events = Vec::new();

        events.push(LearningEvent::IterationStarted(self.stats.iterations));

        // Phase 1: Inference - analyze existing code
        if let Some(feedback) = self.feedback.get_recent_feedback(10).first() {
            if let Some(ref comment) = feedback.comment {
                // Analyze obfuscation if it's JavaScript
                if comment.contains("function") || comment.contains("var") {
                    let analysis = self.deobfuscator.analyze_obfuscation(comment);
                    log::debug!(
                        "Obfuscation analysis: score={:.2}, techniques={}",
                        analysis.obfuscation_score,
                        analysis.techniques.len()
                    );
                }
            }
        }

        // Phase 2: Learning - collect and process samples
        let samples_collected = self.collect_learning_samples()?;
        self.stats.samples_processed += samples_collected;

        for _ in 0..samples_collected {
            events.push(LearningEvent::SampleCollected("training_sample".to_string()));
        }

        // Phase 3: Model update if enough samples
        if self.learner.should_update() {
            match self.learner.trigger_update() {
                Ok(_) => {
                    self.stats.updates_performed += 1;
                    events.push(LearningEvent::ModelUpdated(self.stats.updates_performed));
                    log::info!("Model updated (update #{})", self.stats.updates_performed);
                }
                Err(e) => {
                    events.push(LearningEvent::Error(format!("Update failed: {}", e)));
                    log::warn!("Model update failed: {}", e);
                }
            }
        }

        // Phase 4: Generation - generate new code if enabled
        if self.config.auto_generate {
            match self.generate_sample_code() {
                Ok(generated) => {
                    self.stats.codes_generated += 1;
                    events.push(LearningEvent::CodeGenerated(generated.code_type));
                    log::debug!("Generated code: {} tokens", generated.metadata.token_count);
                }
                Err(e) => {
                    events.push(LearningEvent::Error(format!("Generation failed: {}", e)));
                }
            }
        }

        // Update statistics
        self.stats.iterations += 1;
        let iteration_time = start.elapsed();
        self.update_avg_time(iteration_time);

        events.push(LearningEvent::IterationCompleted(
            self.stats.iterations,
            iteration_time,
        ));

        Ok(events)
    }

    /// Run the continuous learning loop
    pub fn run(&mut self, callback: Option<Box<dyn Fn(&LearningEvent)>>) -> Result<()> {
        self.running = true;
        log::info!("Starting continuous learning loop");

        let mut iteration = 0;
        loop {
            if !self.running {
                log::info!("Learning loop stopped");
                break;
            }

            // Check iteration limit
            if let Some(max_iter) = self.config.max_iterations {
                if iteration >= max_iter {
                    log::info!("Reached maximum iterations ({})", max_iter);
                    break;
                }
            }

            // Run one iteration
            match self.run_iteration() {
                Ok(events) => {
                    for event in events {
                        if let Some(ref cb) = callback {
                            cb(&event);
                        }
                    }
                }
                Err(e) => {
                    log::error!("Iteration error: {}", e);
                    if let Some(ref cb) = callback {
                        cb(&LearningEvent::Error(e.to_string()));
                    }
                }
            }

            iteration += 1;

            // Sleep between iterations
            std::thread::sleep(Duration::from_secs(self.config.update_interval_secs));
        }

        Ok(())
    }

    /// Stop the learning loop
    pub fn stop(&mut self) {
        self.running = false;
        log::info!("Stopping continuous learning loop");
    }

    /// Collect learning samples from feedback
    fn collect_learning_samples(&mut self) -> Result<usize> {
        let mut count = 0;

        // Get recent feedback items
        let recent_feedback = self.feedback.get_recent_feedback(10);

        for feedback in recent_feedback {
            // Use comment field if available
            if let Some(ref comment) = feedback.comment {
                // Create training sample from feedback comment
                let input_features = self.extract_features(comment);
                let output_features = input_features.clone(); // Placeholder

                let sample = super::online_learning::TrainingSample::new(
                    input_features,
                    output_features,
                );

                self.learner.add_sample(sample);
                count += 1;
            }
        }

        Ok(count)
    }

    /// Extract features from code content
    fn extract_features(&self, content: &str) -> Vec<f32> {
        // Simple feature extraction (placeholder)
        vec![
            content.len() as f32,
            content.lines().count() as f32,
            content.matches('{').count() as f32,
            content.matches('}').count() as f32,
            content.matches("function").count() as f32,
        ]
    }

    /// Generate sample code for testing/learning
    fn generate_sample_code(&self) -> Result<GeneratedCode> {
        use std::collections::HashMap;

        let code_types = vec![CodeType::Html, CodeType::Css, CodeType::JavaScript];
        let code_type = code_types[self.stats.iterations % code_types.len()].clone();

        let mut constraints = HashMap::new();

        let description = match code_type {
            CodeType::Html => {
                constraints.insert("title".to_string(), "Generated Page".to_string());
                constraints.insert("heading".to_string(), "Test Heading".to_string());
                constraints.insert("content".to_string(), "Test content".to_string());
                "basic page"
            }
            CodeType::Css => {
                constraints.insert("font".to_string(), "Arial, sans-serif".to_string());
                constraints.insert("padding".to_string(), "20px".to_string());
                constraints.insert("background".to_string(), "#f5f5f5".to_string());
                "basic styling"
            }
            CodeType::JavaScript => {
                constraints.insert("name".to_string(), "testFunction".to_string());
                constraints.insert("params".to_string(), "".to_string());
                constraints.insert("body".to_string(), "console.log('test');".to_string());
                "function"
            }
        };

        let request = super::code_generator::GenerationRequest {
            code_type,
            description: description.to_string(),
            constraints,
        };

        self.generator.generate(&request)
    }

    /// Update average iteration time
    fn update_avg_time(&mut self, duration: Duration) {
        let new_time_ms = duration.as_secs_f64() * 1000.0;
        let n = self.stats.iterations as f64;

        if n == 1.0 {
            self.stats.avg_iteration_time_ms = new_time_ms;
        } else {
            // Running average
            self.stats.avg_iteration_time_ms =
                (self.stats.avg_iteration_time_ms * (n - 1.0) + new_time_ms) / n;
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &LearningLoopStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = LearningLoopStats {
            iterations: 0,
            samples_processed: 0,
            updates_performed: 0,
            codes_generated: 0,
            avg_iteration_time_ms: 0.0,
            success_rate: 1.0,
        };
    }

    /// Check if loop is running
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Get learner statistics
    pub fn get_learner_stats(&self) -> super::online_learning::LearningStats {
        self.learner.get_stats()
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ContinuousLearningConfig) {
        self.config = config;
        self.learner.set_learning_rate(self.config.learning_rate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_learning_creation() {
        let loop_instance = ContinuousLearningLoop::with_defaults();
        assert!(!loop_instance.is_running());
        assert_eq!(loop_instance.stats.iterations, 0);
    }

    #[test]
    fn test_single_iteration() {
        let mut loop_instance = ContinuousLearningLoop::with_defaults();
        let result = loop_instance.run_iteration();

        assert!(result.is_ok());
        let events = result.unwrap();
        assert!(!events.is_empty());
        assert_eq!(loop_instance.stats.iterations, 1);
    }

    #[test]
    fn test_multiple_iterations() {
        let mut config = ContinuousLearningConfig::default();
        config.max_iterations = Some(3);
        config.update_interval_secs = 0; // No delay for testing

        let mut loop_instance = ContinuousLearningLoop::new(config);

        for _ in 0..3 {
            let result = loop_instance.run_iteration();
            assert!(result.is_ok());
        }

        assert_eq!(loop_instance.stats.iterations, 3);
    }

    #[test]
    fn test_stats_tracking() {
        let mut loop_instance = ContinuousLearningLoop::with_defaults();

        loop_instance.run_iteration().unwrap();
        let stats = loop_instance.get_stats();

        assert!(stats.avg_iteration_time_ms >= 0.0);
        assert_eq!(stats.iterations, 1);
    }

    #[test]
    fn test_stop_functionality() {
        let mut loop_instance = ContinuousLearningLoop::with_defaults();
        loop_instance.running = true;
        assert!(loop_instance.is_running());

        loop_instance.stop();
        assert!(!loop_instance.is_running());
    }

    #[test]
    fn test_config_update() {
        let mut loop_instance = ContinuousLearningLoop::with_defaults();
        let mut new_config = ContinuousLearningConfig::default();
        new_config.learning_rate = 0.01;

        loop_instance.update_config(new_config);
        assert_eq!(loop_instance.learner.get_learning_rate(), 0.01);
    }
}
