pub mod advanced_deobfuscation;
pub mod advanced_orchestrator;   // NEW: 高级管道编排器
#[cfg(feature = "ml")]
pub mod ai_deobfuscator;
pub mod ast_deobfuscation;
pub mod control_flow_graph;      // NEW: 控制流图分析
pub mod data_flow_analyzer;     // NEW: 数据流分析
pub mod deobfuscation;
pub mod enhanced_deobfuscation;
pub mod jsunpack_deobfuscator;  // NEW: JSUnpack-inspired deobfuscator
pub mod obfuscation_pattern_library; // NEW: 混淆模式识别库
pub mod obfuscation_detector_week4; // Week 4: ONNX 集成混淆检测器
#[cfg(feature = "ai")]
pub mod onnx_inference;
pub mod python_integration;     // NEW: Python反混淆系统集成
pub mod semantic_model;
pub mod string_pool_extractor;  // NEW: 字符串池析取器
pub mod symbolic_executor;      // NEW: 符号执行引擎
pub mod type_inference;         // NEW: 类型推断系统

pub use advanced_deobfuscation::{
    AdvancedDeobfuscator, AdvancedObfuscationAnalysis, FrameworkObfuscation,
};
pub use advanced_orchestrator::{
    AdvancedDeobfuscationPipeline, PipelineAnalysisResult, PipelineStage,
    AnalysisSummary, Insight,
};
#[cfg(feature = "ml")]
pub use ai_deobfuscator::{AIDeobfuscator, TransformerConfig};
pub use ast_deobfuscation::{ASTDeobfuscationStats, ASTDeobfuscator, VariableUsage};
pub use control_flow_graph::{
    ControlFlowAnalyzer, ControlFlowGraph, CFGNodeType, ReachabilityAnalysis,
    LoopAnalysis, SCC, CFGStatistics,
};
pub use data_flow_analyzer::{
    DataFlowAnalyzer, DataFlowAnalysisResult, DefUseChain, TaintInfo,
};
pub use deobfuscation::{DeobfuscationStrategy, JsDeobfuscator, ObfuscationAnalysis};
pub use enhanced_deobfuscation::{
    DeobfuscationStats, EnhancedDeobfuscationResult, EnhancedDeobfuscator, ProxyFunctionType,
    SelfDefendingPattern,
};
pub use jsunpack_deobfuscator::{
    JSUnpackDeobfuscator, UnpackResult, PackerType, DecodingTechnique, 
    AnalysisReport, RiskLevel, SuspiciousPattern, Severity,
};
pub use obfuscation_pattern_library::{
    ObfuscationPatternLibrary, ObfuscationPatternType, DetectedPattern, 
    PatternLibraryStatistics,
};
pub use obfuscation_detector_week4::{
    OnnxObfuscationDetector, ObfuscationDetectionResult, ObfuscationTechnique,
    FeatureExtractor, ComplexityMetrics,
};
pub use python_integration::{
    PythonDeobfuscationSystem, PythonDeobfuscationResult, ObfuscatorInfo,
    KnowledgeBaseStatistics,
};
pub use semantic_model::{
    DeobfuscationResult, SemanticDeobfuscator, SemanticKnowledgeBase, SemanticPrediction,
};
pub use string_pool_extractor::{
    StringPoolExtractor, StringPool, StringPoolEntry, StringSource, 
    StringPoolStatistics,
};
pub use symbolic_executor::{
    SymbolicExecutor, SymbolicAnalysisResult, SymbolicValue,
};
pub use type_inference::{
    TypeInferencer, TypeInferenceResult, TypeInfo, JSType, FunctionSignature,
};

