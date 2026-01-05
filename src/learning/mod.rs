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
/// - Code generation
/// - JS deobfuscation
/// - Continuous learning loop

pub mod feedback;
pub mod versioning;
pub mod metrics;
pub mod online_learning;
pub mod ab_testing;
pub mod optimization;
pub mod personalization;
pub mod website_learner;
pub mod code_generator;
pub mod deobfuscation;
pub mod advanced_deobfuscation;
pub mod continuous_loop;

pub use feedback::{FeedbackCollector, Feedback, FeedbackType};
pub use versioning::{ModelVersion, VersionManager};
pub use metrics::{MetricsDashboard, Metric, MetricType};
pub use online_learning::{OnlineLearner, LearningConfig};
pub use ab_testing::{ABTest, TestVariant, ABTestManager};
pub use optimization::{SelfOptimizer, OptimizationStrategy};
pub use personalization::{UserPreferences, PersonalizationEngine};
pub use website_learner::WebsiteLearner;
pub use code_generator::{CodeGenerator, GenerationRequest, CodeType, GeneratedCode};
pub use deobfuscation::{JsDeobfuscator, ObfuscationAnalysis, DeobfuscationStrategy};
pub use advanced_deobfuscation::{AdvancedDeobfuscator, AdvancedObfuscationAnalysis, FrameworkObfuscation};
pub use continuous_loop::{ContinuousLearningLoop, ContinuousLearningConfig};
