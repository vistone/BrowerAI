/// Self-optimization system for autonomous improvement
/// 
/// Automatically adjusts system parameters and selects best-performing models

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Optimization strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizationStrategy {
    /// Optimize for speed
    Speed,
    /// Optimize for accuracy
    Accuracy,
    /// Optimize for memory usage
    Memory,
    /// Balance between speed and accuracy
    Balanced,
    /// Custom optimization target
    Custom(String),
}

/// Configuration for self-optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Primary optimization strategy
    pub strategy: OptimizationStrategy,
    /// Minimum improvement threshold (0.0 to 1.0)
    pub min_improvement_threshold: f32,
    /// Enable automatic model switching
    pub auto_switch_models: bool,
    /// Enable parameter tuning
    pub auto_tune_parameters: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            strategy: OptimizationStrategy::Balanced,
            min_improvement_threshold: 0.05, // 5% improvement
            auto_switch_models: false,
            auto_tune_parameters: false,
        }
    }
}

/// Performance measurement for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    /// Model or configuration ID
    pub id: String,
    /// Speed metric (lower is better)
    pub speed_ms: f32,
    /// Accuracy metric (higher is better)
    pub accuracy: f32,
    /// Memory usage in MB
    pub memory_mb: f32,
    /// Sample count
    pub sample_count: usize,
}

impl PerformanceMeasurement {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            speed_ms: 0.0,
            accuracy: 0.0,
            memory_mb: 0.0,
            sample_count: 0,
        }
    }

    /// Update metrics with new measurements
    pub fn update(&mut self, speed_ms: f32, accuracy: f32, memory_mb: f32) {
        let n = self.sample_count as f32;
        self.speed_ms = (self.speed_ms * n + speed_ms) / (n + 1.0);
        self.accuracy = (self.accuracy * n + accuracy) / (n + 1.0);
        self.memory_mb = (self.memory_mb * n + memory_mb) / (n + 1.0);
        self.sample_count += 1;
    }

    /// Calculate score based on strategy
    pub fn score(&self, strategy: &OptimizationStrategy) -> f32 {
        match strategy {
            OptimizationStrategy::Speed => 1000.0 / self.speed_ms.max(1.0),
            OptimizationStrategy::Accuracy => self.accuracy,
            OptimizationStrategy::Memory => 1000.0 / self.memory_mb.max(1.0),
            OptimizationStrategy::Balanced => {
                // Balanced: 40% accuracy, 40% speed, 20% memory
                0.4 * self.accuracy + 0.4 * (100.0 / self.speed_ms.max(1.0)) + 0.2 * (100.0 / self.memory_mb.max(10.0))
            }
            OptimizationStrategy::Custom(_) => self.accuracy,
        }
    }
}

/// Self-optimizer that improves system performance
pub struct SelfOptimizer {
    /// Optimization configuration
    config: OptimizationConfig,
    /// Performance measurements by model/config ID
    measurements: HashMap<String, PerformanceMeasurement>,
    /// Currently active model/config
    active_id: Option<String>,
    /// Optimization history
    optimization_count: usize,
}

impl SelfOptimizer {
    /// Create a new self-optimizer
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            measurements: HashMap::new(),
            active_id: None,
            optimization_count: 0,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(OptimizationConfig::default())
    }

    /// Record performance measurement
    pub fn record_performance(
        &mut self,
        id: impl Into<String>,
        speed_ms: f32,
        accuracy: f32,
        memory_mb: f32,
    ) {
        let id_str = id.into();
        self.measurements
            .entry(id_str.clone())
            .or_insert_with(|| PerformanceMeasurement::new(&id_str))
            .update(speed_ms, accuracy, memory_mb);
    }

    /// Get best performing model/config based on strategy
    pub fn get_best_performer(&self) -> Option<&PerformanceMeasurement> {
        self.measurements
            .values()
            .filter(|m| m.sample_count > 0)
            .max_by(|a, b| {
                let a_score = a.score(&self.config.strategy);
                let b_score = b.score(&self.config.strategy);
                a_score.partial_cmp(&b_score).unwrap()
            })
    }

    /// Check if optimization is needed
    pub fn should_optimize(&self) -> bool {
        if self.measurements.len() < 2 {
            return false;
        }

        let best = match self.get_best_performer() {
            Some(b) => b,
            None => return false,
        };

        // Check if current active is significantly worse than best
        if let Some(active_id) = &self.active_id {
            if let Some(active) = self.measurements.get(active_id) {
                let active_score = active.score(&self.config.strategy);
                let best_score = best.score(&self.config.strategy);
                
                if best_score > 0.0 {
                    let improvement = (best_score - active_score) / best_score;
                    return improvement >= self.config.min_improvement_threshold;
                }
            }
        }

        true
    }

    /// Perform optimization (switch to best model)
    pub fn optimize(&mut self) -> Option<String> {
        if !self.should_optimize() {
            return None;
        }

        let best = self.get_best_performer()?;
        let best_id = best.id.clone();

        if self.config.auto_switch_models {
            self.active_id = Some(best_id.clone());
            self.optimization_count += 1;
        }

        Some(best_id)
    }

    /// Set active model/config
    pub fn set_active(&mut self, id: impl Into<String>) {
        self.active_id = Some(id.into());
    }

    /// Get active model/config ID
    pub fn get_active(&self) -> Option<&str> {
        self.active_id.as_deref()
    }

    /// Get performance measurements
    pub fn get_measurements(&self) -> &HashMap<String, PerformanceMeasurement> {
        &self.measurements
    }

    /// Get optimization statistics
    pub fn get_stats(&self) -> OptimizationStats {
        OptimizationStats {
            model_count: self.measurements.len(),
            optimization_count: self.optimization_count,
            active_model: self.active_id.clone(),
            best_model: self.get_best_performer().map(|m| m.id.clone()),
        }
    }

    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
        self.active_id = None;
    }
}

/// Statistics for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStats {
    pub model_count: usize,
    pub optimization_count: usize,
    pub active_model: Option<String>,
    pub best_model: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();
        assert_eq!(config.strategy, OptimizationStrategy::Balanced);
        assert!(!config.auto_switch_models);
    }

    #[test]
    fn test_performance_measurement_creation() {
        let perf = PerformanceMeasurement::new("model_v1");
        assert_eq!(perf.id, "model_v1");
        assert_eq!(perf.sample_count, 0);
    }

    #[test]
    fn test_performance_measurement_update() {
        let mut perf = PerformanceMeasurement::new("model_v1");
        perf.update(100.0, 0.95, 50.0);
        
        assert_eq!(perf.speed_ms, 100.0);
        assert_eq!(perf.accuracy, 0.95);
        assert_eq!(perf.memory_mb, 50.0);
        assert_eq!(perf.sample_count, 1);
        
        perf.update(200.0, 0.97, 60.0);
        assert_eq!(perf.sample_count, 2);
        assert_eq!(perf.speed_ms, 150.0); // Average of 100 and 200
    }

    #[test]
    fn test_performance_measurement_score_speed() {
        let mut perf = PerformanceMeasurement::new("model_v1");
        perf.update(100.0, 0.95, 50.0);
        
        let score = perf.score(&OptimizationStrategy::Speed);
        assert_eq!(score, 10.0); // 1000 / 100
    }

    #[test]
    fn test_performance_measurement_score_accuracy() {
        let mut perf = PerformanceMeasurement::new("model_v1");
        perf.update(100.0, 0.95, 50.0);
        
        let score = perf.score(&OptimizationStrategy::Accuracy);
        assert_eq!(score, 0.95);
    }

    #[test]
    fn test_self_optimizer_creation() {
        let optimizer = SelfOptimizer::with_defaults();
        assert!(optimizer.get_active().is_none());
        assert_eq!(optimizer.optimization_count, 0);
    }

    #[test]
    fn test_self_optimizer_record_performance() {
        let mut optimizer = SelfOptimizer::with_defaults();
        optimizer.record_performance("model_v1", 100.0, 0.95, 50.0);
        
        assert_eq!(optimizer.measurements.len(), 1);
        assert!(optimizer.measurements.contains_key("model_v1"));
    }

    #[test]
    fn test_self_optimizer_get_best_performer() {
        let mut optimizer = SelfOptimizer::with_defaults();
        optimizer.record_performance("model_v1", 100.0, 0.90, 50.0);
        optimizer.record_performance("model_v2", 100.0, 0.95, 50.0);
        
        let best = optimizer.get_best_performer();
        assert!(best.is_some());
        assert_eq!(best.unwrap().id, "model_v2"); // Higher accuracy
    }

    #[test]
    fn test_self_optimizer_should_optimize() {
        let mut config = OptimizationConfig::default();
        config.strategy = OptimizationStrategy::Accuracy; // Use accuracy strategy for simpler testing
        let mut optimizer = SelfOptimizer::new(config);
        
        // Not enough data
        assert!(!optimizer.should_optimize());
        
        optimizer.record_performance("model_v1", 100.0, 0.80, 50.0);
        optimizer.set_active("model_v1");
        
        // Still not enough models to compare
        assert!(!optimizer.should_optimize());
        
        optimizer.record_performance("model_v2", 100.0, 0.99, 50.0);
        
        // Now should optimize (significant improvement available)
        assert!(optimizer.should_optimize());
    }

    #[test]
    fn test_self_optimizer_optimize() {
        let mut config = OptimizationConfig::default();
        config.auto_switch_models = true;
        config.strategy = OptimizationStrategy::Accuracy; // Use accuracy strategy for simpler testing
        let mut optimizer = SelfOptimizer::new(config);
        
        optimizer.record_performance("model_v1", 100.0, 0.80, 50.0);
        optimizer.set_active("model_v1");
        optimizer.record_performance("model_v2", 100.0, 0.99, 50.0);
        
        let result = optimizer.optimize();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "model_v2");
        assert_eq!(optimizer.get_active(), Some("model_v2"));
    }

    #[test]
    fn test_self_optimizer_stats() {
        let mut optimizer = SelfOptimizer::with_defaults();
        optimizer.record_performance("model_v1", 100.0, 0.95, 50.0);
        optimizer.set_active("model_v1");
        
        let stats = optimizer.get_stats();
        assert_eq!(stats.model_count, 1);
        assert_eq!(stats.active_model, Some("model_v1".to_string()));
    }

    #[test]
    fn test_optimization_strategy_balanced() {
        let mut perf = PerformanceMeasurement::new("model_v1");
        perf.update(100.0, 0.95, 50.0);
        
        let score = perf.score(&OptimizationStrategy::Balanced);
        assert!(score > 0.0);
    }
}
