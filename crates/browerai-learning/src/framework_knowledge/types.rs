use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FrameworkCategory {
    Bundler,
    FrontendFramework,
    MetaFramework,
    MobileCrossPlatform,
    StateManagement,
    UILibrary,
    ObfuscatorTool,
    MicroFrontend,
    TestingFramework,
    SSRFramework,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SignatureType {
    StringLiteral,
    Regex,
    ASTPattern,
    ImportStatement,
    FunctionCall,
    VariableDeclaration,
    GlobalObject,
    Comment,
    LicenseHeader,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObfuscationTechnique {
    NameMangling,
    StringEncoding,
    ControlFlowFlattening,
    DeadCodeInjection,
    OpaquePredicates,
    StringArrayRotation,
    ProxyFunctions,
    SelfDefending,
    AntiDebugging,
    DomainLocking,
    CodeSplitting,
    LazyLoading,
    ModuleWrapping,
    TemplateCompilation,
    SourceMapRemoval,
    Minification,
    TreeShaking,
    DynamicImports,
    WebpackChunking,
    ConstantFolding,
    FunctionInlining,
    PropertyMangling,
    UnicodeEscaping,
}
