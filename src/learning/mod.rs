/// Learning and adaptation module for BrowerAI
/// 
/// This module implements Phase 5 functionality:
/// - Feedback collection
/// - Online learning
/// - Model versioning
/// - A/B testing
/// - Metrics dashboard
/// - Self-optimization
/// - User personalization

pub mod feedback;
pub mod versioning;
pub mod metrics;
pub mod online_learning;
pub mod ab_testing;
pub mod optimization;
pub mod personalization;

pub use feedback::{FeedbackCollector, Feedback, FeedbackType};
pub use versioning::{ModelVersion, VersionManager};
pub use metrics::{MetricsDashboard, Metric, MetricType};
pub use online_learning::{OnlineLearner, LearningConfig};
pub use ab_testing::{ABTest, TestVariant, ABTestManager};
pub use optimization::{SelfOptimizer, OptimizationStrategy};
pub use personalization::{UserPreferences, PersonalizationEngine};
