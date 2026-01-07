pub mod ab_testing;
pub mod advanced_deobfuscation;
pub mod code_generator;
pub mod continuous_loop;
pub mod deobfuscation;
pub mod enhanced_deobfuscation;
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
pub mod metrics;
pub mod online_learning;
pub mod optimization;
pub mod personalization;
pub mod versioning;
pub mod website_learner;

pub use ab_testing::{ABTest, ABTestManager, TestVariant};
pub use advanced_deobfuscation::{
    AdvancedDeobfuscator, AdvancedObfuscationAnalysis, FrameworkObfuscation,
};
pub use code_generator::{CodeGenerator, CodeType, GeneratedCode, GenerationRequest};
pub use continuous_loop::{ContinuousLearningConfig, ContinuousLearningLoop};
pub use deobfuscation::{DeobfuscationStrategy, JsDeobfuscator, ObfuscationAnalysis};
pub use enhanced_deobfuscation::{
    EnhancedDeobfuscator, EnhancedDeobfuscationResult, DeobfuscationStats,
    SelfDefendingPattern, ProxyFunctionType,
};
pub use feedback::{Feedback, FeedbackCollector, FeedbackType};
pub use metrics::{Metric, MetricType, MetricsDashboard};
pub use online_learning::{LearningConfig, OnlineLearner};
pub use optimization::{OptimizationStrategy, SelfOptimizer};
pub use personalization::{PersonalizationEngine, UserPreferences};
pub use versioning::{ModelVersion, VersionManager};
pub use website_learner::WebsiteLearner;
