pub mod ai;
pub mod devtools;
pub mod dom;
pub mod intelligent_rendering;
pub mod learning;
pub mod network;
pub mod parser;
pub mod plugins;
pub mod renderer;
pub mod testing;

pub use ai::{
    AdvancedPerformanceMonitor, AiConfig, AiStats, FallbackReason, FallbackTracker, GpuConfig,
    GpuProvider, GpuStats, HotReloadManager, InferenceEngine, ModelHealth, ModelHealthSummary,
    ModelManager,
};
pub use devtools::{DOMInspector, NetworkMonitor, PerformanceProfiler};
pub use dom::{Document, DomApiExtensions, DomElement, DomNode, ElementHandle, JsSandbox};
pub use learning::{
    CodeGenerator, CodeType, ContinuousLearningLoop, DeobfuscationStrategy, FeedbackCollector,
    GeneratedCode, GenerationRequest, JsDeobfuscator, MetricsDashboard, ObfuscationAnalysis,
    OnlineLearner, PersonalizationEngine, SelfOptimizer, VersionManager,
};
pub use network::{HttpClient, ResourceCache};
pub use parser::{CssParser, HtmlParser, JsParser};
pub use plugins::{Plugin, PluginLoader, PluginRegistry};
pub use renderer::{
    AiLayoutHint, LayoutValidator, PredictiveRenderer, RenderEngine, ValidationReport,
};
pub use testing::{
    BenchmarkConfig, BenchmarkResult, BenchmarkRunner, ComparisonResult, WebsiteTestResult,
    WebsiteTestSuite, WebsiteTester,
};
