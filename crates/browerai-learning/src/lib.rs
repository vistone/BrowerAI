pub mod ab_testing;
pub mod async_client;
pub mod auth_handler;
pub mod browser_automation;
pub mod browser_tech_detector;
pub mod code_generator;
pub mod code_verifier;
pub mod complete_inference_pipeline;
pub mod continuous_loop;
pub mod data_models;
pub mod data_structure_inference;
pub mod deobfuscation;
pub mod external_resource_analyzer;
pub mod feedback;
pub mod framework_knowledge;
pub mod generators;
pub mod high_fidelity_generator;
pub mod improved_code_generator;
pub mod integration_test;
pub mod learning_quality;
pub mod learning_sandbox;
pub mod metrics;
pub mod online_learning;
pub mod optimization;
pub mod personalization;
pub mod pipeline;
pub mod real_website_learner;
pub mod safe_sandbox;
pub mod semantic_comparator;
pub mod v8_tracer;
pub mod validation;
pub mod variable_semantics;
pub mod versioning;
pub mod wasm_analyzer;
pub mod website_generator;
pub mod website_learner;
pub mod websocket_analyzer;
pub mod websocket_monitor;
pub mod workflow_extractor;

pub mod dual_sandbox_learner;
pub use ab_testing::{ABTest, ABTestManager, TestVariant};
pub use async_client::{BrowserAIClient, ClientConfig, ClientState, CompleteResult};
pub use auth_handler::{AuthConfig, AuthManager, AuthToken, AuthenticationType};
pub use browerai_core::CodeType;
pub use browser_automation::{
    BrowserAutomation, BrowserConfig, BrowserSession, NetworkMonitor, NetworkRequest,
};
pub use browser_tech_detector::{
    BrowserTechDetector, BrowserTechnology, TechnologyDetectionResult,
};
pub use code_generator::{CodeGenerator, GeneratedCode, GenerationRequest};
pub use code_verifier::{
    CodeVerificationResult, CodeVerifier, CssVerification, HtmlVerification, JsVerification,
    VerificationError,
};
pub use complete_inference_pipeline::{
    CodeGenerationHint, CompleteInferencePipeline, CompleteInferenceResult, HintType,
};
pub use continuous_loop::{ContinuousLearningConfig, ContinuousLearningLoop};
pub use data_models::*;
pub use data_structure_inference::{
    DataStructureInferenceEngine, Field, InferredStructure, StructureInferenceResult, StructureType,
};
pub use external_resource_analyzer::{
    ExternalResourceAnalyzer, ExternalResourceGraph, ResourceDependency, ResourceType,
};
pub use framework_knowledge::{
    ConfidenceWeights, DeobfuscationStrategy as FrameworkDeobfuscationStrategy, DetectionResult,
    FrameworkCategory, FrameworkKnowledge, FrameworkKnowledgeBase, KnowledgeBaseStats,
    ObfuscationPattern, ObfuscationSignature, ObfuscationTechnique, SignatureType,
};
pub use high_fidelity_generator::{
    GeneratedWebsite as HighFidelityWebsite, HighFidelityGenerator, WebsiteAnalysisComplete,
};
pub use improved_code_generator::{GeneratedModule, ImprovedCodeGenerator};
pub use learning_quality::{IssueType, LearningQuality, QualityIssue, Severity};
pub use learning_sandbox::{IntentAnalyzer, WebsiteIntent};
pub use metrics::MetricsDashboardExt;
pub use metrics::{Histogram, Metric, MetricStats, MetricType, MetricsDashboard};
pub use online_learning::{LearningConfig, OnlineLearner};
pub use optimization::{OptimizationStrategy, SelfOptimizer};
pub use personalization::{PersonalizationEngine, UserPreferences};
pub use pipeline::{LearningInput, LearningOutput, LearningPipeline};
pub use real_website_learner::{
    LearningSession, RealWebsiteLearner, SessionStatus, WebsiteLearningTask,
};
pub use safe_sandbox::{BehaviorRecorder, PageFetcher};
pub use semantic_comparator::{SemanticComparator, SemanticComparisonResult};
pub use v8_tracer::{
    CallRecord, DOMOperation, EventListener, ExecutionTrace, OperationChain, StateChange,
    UserEvent, V8Tracer,
};
pub use variable_semantics::{
    DataType, InferenceResult, VariableScope, VariableSemantics, VariableSemanticsAnalyzer,
};
pub use versioning::{ModelVersion, VersionManager};
pub use wasm_analyzer::{WasmAnalyzer, WasmCallGraph, WasmFunction, WasmModuleInfo};
pub use website_generator::{GeneratedWebsite, WebsiteConfig, WebsiteGenerator};
pub use website_learner::WebsiteLearner;
pub use websocket_analyzer::{
    ConnectionType, MessageFormatAnalysis, WebSocketAnalyzer, WebSocketInfo,
};
pub use workflow_extractor::{Workflow, WorkflowExtractionResult, WorkflowExtractor};

pub use dual_sandbox_learner::{DualSandboxLearner, DualSandboxLearningResult, LearningSummary};
// Re-export from new standalone crates
// pub use browerai_deobfuscation as deobfuscation;
// pub use browerai_feedback as feedback;
