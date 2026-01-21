pub mod advanced_deobfuscation;
#[cfg(feature = "ml")]
pub mod ai_deobfuscator;
pub mod ast_deobfuscation;
pub mod deobfuscation;
pub mod enhanced_deobfuscation;
#[cfg(feature = "ai")]
pub mod onnx_inference;
pub mod semantic_model;

pub use advanced_deobfuscation::{
    AdvancedDeobfuscator, AdvancedObfuscationAnalysis, FrameworkObfuscation,
};
#[cfg(feature = "ml")]
pub use ai_deobfuscator::{AIDeobfuscator, TransformerConfig};
pub use ast_deobfuscation::{ASTDeobfuscationStats, ASTDeobfuscator, VariableUsage};
pub use deobfuscation::{DeobfuscationStrategy, JsDeobfuscator, ObfuscationAnalysis};
pub use enhanced_deobfuscation::{
    DeobfuscationStats, EnhancedDeobfuscationResult, EnhancedDeobfuscator, ProxyFunctionType,
    SelfDefendingPattern,
};
pub use semantic_model::{
    DeobfuscationResult, SemanticDeobfuscator, SemanticKnowledgeBase, SemanticPrediction,
};
