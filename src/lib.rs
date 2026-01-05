pub mod ai;
pub mod devtools;
pub mod dom;
pub mod learning;
pub mod network;
pub mod parser;
pub mod plugins;
pub mod renderer;
pub mod testing;

pub use ai::{AdvancedPerformanceMonitor, HotReloadManager, InferenceEngine, ModelManager};
pub use devtools::{DOMInspector, NetworkMonitor, PerformanceProfiler};
pub use dom::{Document, DomElement, DomNode, DomApiExtensions, ElementHandle, JsSandbox};
pub use learning::{
    FeedbackCollector, MetricsDashboard, OnlineLearner, PersonalizationEngine, SelfOptimizer,
    VersionManager, CodeGenerator, GenerationRequest, CodeType, GeneratedCode,
    JsDeobfuscator, ObfuscationAnalysis, DeobfuscationStrategy, ContinuousLearningLoop,
};
pub use network::{HttpClient, ResourceCache};
pub use parser::{CssParser, HtmlParser, JsParser};
pub use plugins::{Plugin, PluginLoader, PluginRegistry};
pub use renderer::{PredictiveRenderer, RenderEngine};
pub use testing::{WebsiteTestResult, WebsiteTestSuite, WebsiteTester};
