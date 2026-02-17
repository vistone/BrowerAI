pub mod advanced_deobfuscation;
pub mod advanced_orchestrator; // NEW: 高级管道编排器
#[cfg(feature = "ml")]
pub mod ai_deobfuscator;
pub mod ast_deobfuscation;
pub mod control_flow_graph; // NEW: 控制流图分析
pub mod data_flow_analyzer; // NEW: 数据流分析
pub mod deobfuscation;
pub mod enhanced_deobfuscation;
pub mod jsunpack_deobfuscator; // NEW: JSUnpack-inspired deobfuscator
pub mod obfuscation_detector_week4; // Week 4: ONNX 集成混淆检测器
pub mod obfuscation_pattern_library; // NEW: 混淆模式识别库
#[cfg(feature = "ai")]
pub mod onnx_inference;
pub mod python_integration; // NEW: Python反混淆系统集成
pub mod semantic_model;
pub mod string_pool_extractor; // NEW: 字符串池析取器
pub mod symbolic_executor; // NEW: 符号执行引擎
pub mod type_inference; // NEW: 类型推断系统

pub use advanced_deobfuscation::{
    AdvancedDeobfuscator, AdvancedObfuscationAnalysis, FrameworkObfuscation,
};
pub use advanced_orchestrator::{
    AdvancedDeobfuscationPipeline, AnalysisSummary, Insight, PipelineAnalysisResult, PipelineStage,
};
#[cfg(feature = "ml")]
pub use ai_deobfuscator::{AIDeobfuscator, TransformerConfig};
pub use ast_deobfuscation::{ASTDeobfuscationStats, ASTDeobfuscator, VariableUsage};
pub use control_flow_graph::{
    CFGNodeType, CFGStatistics, ControlFlowAnalyzer, ControlFlowGraph, LoopAnalysis,
    ReachabilityAnalysis, SCC,
};
pub use data_flow_analyzer::{DataFlowAnalysisResult, DataFlowAnalyzer, DefUseChain, TaintInfo};
pub use deobfuscation::{DeobfuscationStrategy, JsDeobfuscator, ObfuscationAnalysis};
pub use enhanced_deobfuscation::{
    DeobfuscationStats, EnhancedDeobfuscationResult, EnhancedDeobfuscator, ProxyFunctionType,
    SelfDefendingPattern,
};
pub use jsunpack_deobfuscator::{
    AnalysisReport, DecodingTechnique, JSUnpackDeobfuscator, PackerType, RiskLevel, Severity,
    SuspiciousPattern, UnpackResult,
};
pub use obfuscation_detector_week4::{
    ComplexityMetrics, FeatureExtractor, ObfuscationDetectionResult, ObfuscationTechnique,
    OnnxObfuscationDetector,
};
pub use obfuscation_pattern_library::{
    DetectedPattern, ObfuscationPatternLibrary, ObfuscationPatternType, PatternLibraryStatistics,
};
pub use python_integration::{
    KnowledgeBaseStatistics, ObfuscatorInfo, PythonDeobfuscationResult, PythonDeobfuscationSystem,
};
pub use semantic_model::{
    DeobfuscationResult, SemanticDeobfuscator, SemanticKnowledgeBase, SemanticPrediction,
};
pub use string_pool_extractor::{
    StringPool, StringPoolEntry, StringPoolExtractor, StringPoolStatistics, StringSource,
};
pub use symbolic_executor::{SymbolicAnalysisResult, SymbolicExecutor, SymbolicValue};
pub use type_inference::{
    FunctionSignature, JSType, TypeInferenceResult, TypeInferencer, TypeInfo,
};
