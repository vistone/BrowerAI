use anyhow::Result;
/// Comprehensive Framework Knowledge Base for JavaScript Deobfuscation
///
/// This module contains extensive knowledge about JavaScript frameworks worldwide,
/// their bundling/compilation patterns, and specific obfuscation techniques.
///
/// Coverage includes:
/// - Global frameworks (React, Vue, Angular, etc.)
/// - Chinese frameworks (Taro, Uni-app, Rax, San, Omi, etc.)
/// - Build tools (Webpack, Vite, Rollup, esbuild, etc.)
/// - Obfuscator tools (javascript-obfuscator, terser, uglify-js, etc.)
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive framework knowledge entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkKnowledge {
    /// Framework unique identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Category (bundler, framework, obfuscator, etc.)
    pub category: FrameworkCategory,
    /// Country/region of origin
    pub origin: String,
    /// Primary maintainer/company
    pub maintainer: String,
    /// Detection signatures
    pub signatures: Vec<ObfuscationSignature>,
    /// Known obfuscation patterns
    pub obfuscation_patterns: Vec<ObfuscationPattern>,
    /// Deobfuscation strategies
    pub strategies: Vec<DeobfuscationStrategy>,
    /// Confidence scoring weights
    pub confidence_weights: ConfidenceWeights,
    /// Related frameworks (dependencies, alternatives)
    pub related_frameworks: Vec<String>,
    /// Last updated timestamp
    pub last_updated: String,
}

/// Framework category
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FrameworkCategory {
    /// Build tool/bundler (Webpack, Vite, etc.)
    Bundler,
    /// Frontend framework (React, Vue, etc.)
    FrontendFramework,
    /// Meta framework (Next.js, Nuxt.js, etc.)
    MetaFramework,
    /// Mobile/Cross-platform (React Native, Taro, etc.)
    MobileCrossPlatform,
    /// State management (Redux, MobX, etc.)
    StateManagement,
    /// UI library (Ant Design, Material-UI, etc.)
    UILibrary,
    /// Obfuscator tool
    ObfuscatorTool,
    /// Micro frontend framework
    MicroFrontend,
    /// Testing framework
    TestingFramework,
    /// SSR/Hydration framework
    SSRFramework,
}

/// Obfuscation signature for detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObfuscationSignature {
    /// Signature name/description
    pub name: String,
    /// Pattern type (string literal, regex, AST pattern, etc.)
    pub pattern_type: SignatureType,
    /// Pattern value
    pub pattern: String,
    /// Detection weight (0.0-1.0)
    pub weight: f32,
    /// Required for positive match
    pub required: bool,
    /// Context hints (where to find this pattern)
    pub context: String,
}

/// Signature type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SignatureType {
    /// Exact string match
    StringLiteral,
    /// Regular expression
    Regex,
    /// AST node pattern
    ASTPattern,
    /// Import/require statement
    ImportStatement,
    /// Function call pattern
    FunctionCall,
    /// Variable declaration pattern
    VariableDeclaration,
    /// Comment pattern
    Comment,
    /// License header
    LicenseHeader,
}

/// Obfuscation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObfuscationPattern {
    /// Pattern name
    pub name: String,
    /// Technique description
    pub technique: ObfuscationTechnique,
    /// Example obfuscated code
    pub example_obfuscated: String,
    /// Example deobfuscated code
    pub example_deobfuscated: String,
    /// Complexity score (1-10)
    pub complexity: u8,
    /// Common in this framework (0.0-1.0)
    pub prevalence: f32,
    /// Detection hints
    pub detection_hints: Vec<String>,
}

/// Obfuscation technique classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObfuscationTechnique {
    /// Variable/function name mangling
    NameMangling,
    /// String encoding (hex, base64, unicode)
    StringEncoding,
    /// Control flow flattening
    ControlFlowFlattening,
    /// Dead code injection
    DeadCodeInjection,
    /// Opaque predicates
    OpaquePredicates,
    /// String array rotation
    StringArrayRotation,
    /// Proxy functions
    ProxyFunctions,
    /// Self-defending code
    SelfDefending,
    /// Anti-debugging
    AntiDebugging,
    /// Domain locking
    DomainLocking,
    /// Code splitting
    CodeSplitting,
    /// Lazy loading
    LazyLoading,
    /// Module wrapping
    ModuleWrapping,
    /// JSX/Template compilation
    TemplateCompilation,
    /// Source map removal
    SourceMapRemoval,
    /// Minification
    Minification,
    /// Tree shaking artifacts
    TreeShaking,
    /// Dynamic imports
    DynamicImports,
    /// Webpack chunks
    WebpackChunking,
    /// Constant folding
    ConstantFolding,
    /// Function inlining
    FunctionInlining,
    /// Property mangling
    PropertyMangling,
    /// Unicode escaping
    UnicodeEscaping,
}

/// Deobfuscation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeobfuscationStrategy {
    /// Strategy name
    pub name: String,
    /// Target technique
    pub target: ObfuscationTechnique,
    /// Approach description
    pub approach: String,
    /// Success rate (0.0-1.0)
    pub success_rate: f32,
    /// Implementation priority (1-10)
    pub priority: u8,
    /// Required tools/dependencies
    pub requirements: Vec<String>,
    /// Known limitations
    pub limitations: Vec<String>,
}

/// Confidence scoring weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceWeights {
    /// Weight for signature matches
    pub signature_match: f32,
    /// Weight for pattern matches
    pub pattern_match: f32,
    /// Weight for contextual analysis
    pub contextual: f32,
    /// Weight for related framework presence
    pub related_frameworks: f32,
}

/// Framework knowledge base manager
pub struct FrameworkKnowledgeBase {
    /// All registered frameworks
    frameworks: HashMap<String, FrameworkKnowledge>,
    /// Category index for fast lookup
    category_index: HashMap<FrameworkCategory, Vec<String>>,
    /// Signature index for pattern matching
    signature_index: HashMap<String, Vec<String>>,
}

impl FrameworkKnowledgeBase {
    /// Create a new knowledge base with comprehensive framework data
    pub fn new() -> Self {
        let mut kb = Self {
            frameworks: HashMap::new(),
            category_index: HashMap::new(),
            signature_index: HashMap::new(),
        };

        // Initialize with comprehensive framework knowledge
        kb.init_global_frameworks();
        kb.init_chinese_frameworks();
        kb.init_bundlers();
        kb.init_obfuscators();
        kb.init_meta_frameworks();
        kb.init_state_management();
        kb.init_ui_libraries();
        kb.init_additional_frameworks();
        kb.build_indices();

        kb
    }

    /// Initialize global frameworks (US, Europe, etc.)
    fn init_global_frameworks(&mut self) {
        // React
        self.add_framework(FrameworkKnowledge {
            id: "react".to_string(),
            name: "React".to_string(),
            category: FrameworkCategory::FrontendFramework,
            origin: "USA".to_string(),
            maintainer: "Meta (Facebook)".to_string(),
            signatures: vec![
                ObfuscationSignature {
                    name: "React.createElement".to_string(),
                    pattern_type: SignatureType::FunctionCall,
                    pattern: r"React\.createElement".to_string(),
                    weight: 0.9,
                    required: false,
                    context: "JSX compilation".to_string(),
                },
                ObfuscationSignature {
                    name: "_jsx runtime".to_string(),
                    pattern_type: SignatureType::ImportStatement,
                    pattern: r"react/jsx-runtime".to_string(),
                    weight: 0.95,
                    required: false,
                    context: "Modern JSX transform".to_string(),
                },
                ObfuscationSignature {
                    name: "_jsxs function".to_string(),
                    pattern_type: SignatureType::FunctionCall,
                    pattern: r"_jsxs?\(".to_string(),
                    weight: 0.85,
                    required: false,
                    context: "Automatic JSX runtime".to_string(),
                },
            ],
            obfuscation_patterns: vec![
                ObfuscationPattern {
                    name: "JSX to createElement".to_string(),
                    technique: ObfuscationTechnique::TemplateCompilation,
                    example_obfuscated:
                        r#"React.createElement("div", {className: "test"}, "Hello")"#.to_string(),
                    example_deobfuscated: r#"<div className="test">Hello</div>"#.to_string(),
                    complexity: 3,
                    prevalence: 1.0,
                    detection_hints: vec!["React.createElement calls".to_string()],
                },
                ObfuscationPattern {
                    name: "Automatic JSX Runtime".to_string(),
                    technique: ObfuscationTechnique::TemplateCompilation,
                    example_obfuscated: r#"_jsx("div", {className: "test", children: "Hello"})"#
                        .to_string(),
                    example_deobfuscated: r#"<div className="test">Hello</div>"#.to_string(),
                    complexity: 3,
                    prevalence: 0.9,
                    detection_hints: vec!["_jsx or _jsxs function calls".to_string()],
                },
            ],
            strategies: vec![DeobfuscationStrategy {
                name: "JSX reconstruction".to_string(),
                target: ObfuscationTechnique::TemplateCompilation,
                approach: "Parse createElement/jsx calls and reconstruct JSX syntax".to_string(),
                success_rate: 0.85,
                priority: 8,
                requirements: vec!["AST parser".to_string(), "React knowledge".to_string()],
                limitations: vec!["Complex spread props may be lossy".to_string()],
            }],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.4,
                pattern_match: 0.3,
                contextual: 0.2,
                related_frameworks: 0.1,
            },
            related_frameworks: vec![
                "webpack".to_string(),
                "next-js".to_string(),
                "react-native".to_string(),
            ],
            last_updated: "2026-01-07".to_string(),
        });

        // Vue
        self.add_framework(FrameworkKnowledge {
            id: "vue".to_string(),
            name: "Vue.js".to_string(),
            category: FrameworkCategory::FrontendFramework,
            origin: "China/Global".to_string(),
            maintainer: "Evan You / Vuejs Team".to_string(),
            signatures: vec![
                ObfuscationSignature {
                    name: "Vue 3 createVNode".to_string(),
                    pattern_type: SignatureType::FunctionCall,
                    pattern: r"_createVNode|_createElementVNode".to_string(),
                    weight: 0.9,
                    required: false,
                    context: "Vue 3 template compilation".to_string(),
                },
                ObfuscationSignature {
                    name: "Vue hoisted variables".to_string(),
                    pattern_type: SignatureType::VariableDeclaration,
                    pattern: r"_hoisted_\d+".to_string(),
                    weight: 0.85,
                    required: false,
                    context: "Static node hoisting optimization".to_string(),
                },
                ObfuscationSignature {
                    name: "Vue context wrapper".to_string(),
                    pattern_type: SignatureType::FunctionCall,
                    pattern: r"_withCtx|_renderSlot".to_string(),
                    weight: 0.8,
                    required: false,
                    context: "Component context and slots".to_string(),
                },
            ],
            obfuscation_patterns: vec![
                ObfuscationPattern {
                    name: "Template to render function".to_string(),
                    technique: ObfuscationTechnique::TemplateCompilation,
                    example_obfuscated: r#"_createElementVNode("div", {class: "test"}, [_createTextVNode("Hello")])"#.to_string(),
                    example_deobfuscated: r#"<div class="test">Hello</div>"#.to_string(),
                    complexity: 4,
                    prevalence: 1.0,
                    detection_hints: vec!["_createVNode function calls".to_string()],
                },
                ObfuscationPattern {
                    name: "Static hoisting".to_string(),
                    technique: ObfuscationTechnique::ConstantFolding,
                    example_obfuscated: r#"const _hoisted_1 = {class: "static"}; _createVNode("div", _hoisted_1)"#.to_string(),
                    example_deobfuscated: r#"<div class="static"></div>"#.to_string(),
                    complexity: 3,
                    prevalence: 0.8,
                    detection_hints: vec!["_hoisted_ variables".to_string()],
                },
            ],
            strategies: vec![
                DeobfuscationStrategy {
                    name: "Vue template reconstruction".to_string(),
                    target: ObfuscationTechnique::TemplateCompilation,
                    approach: "Parse createVNode calls and rebuild template syntax".to_string(),
                    success_rate: 0.80,
                    priority: 7,
                    requirements: vec!["Vue compiler knowledge".to_string()],
                    limitations: vec!["Dynamic components may be complex".to_string()],
                },
            ],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.4,
                pattern_match: 0.3,
                contextual: 0.2,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["vite".to_string(), "nuxt-js".to_string(), "vuex".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // Angular
        self.add_framework(FrameworkKnowledge {
            id: "angular".to_string(),
            name: "Angular".to_string(),
            category: FrameworkCategory::FrontendFramework,
            origin: "USA".to_string(),
            maintainer: "Google".to_string(),
            signatures: vec![
                ObfuscationSignature {
                    name: "Ivy compilation markers".to_string(),
                    pattern_type: SignatureType::StringLiteral,
                    pattern: r"ɵɵ".to_string(),
                    weight: 0.95,
                    required: false,
                    context: "Angular Ivy compiler output".to_string(),
                },
                ObfuscationSignature {
                    name: "Angular decorators".to_string(),
                    pattern_type: SignatureType::ImportStatement,
                    pattern: r"@angular/core".to_string(),
                    weight: 0.9,
                    required: false,
                    context: "Angular imports".to_string(),
                },
            ],
            obfuscation_patterns: vec![ObfuscationPattern {
                name: "Ivy template instructions".to_string(),
                technique: ObfuscationTechnique::TemplateCompilation,
                example_obfuscated:
                    r#"ɵɵelementStart(0, "div"); ɵɵtext(1, "Hello"); ɵɵelementEnd();"#.to_string(),
                example_deobfuscated: r#"<div>Hello</div>"#.to_string(),
                complexity: 5,
                prevalence: 1.0,
                detection_hints: vec!["ɵɵ function calls".to_string()],
            }],
            strategies: vec![DeobfuscationStrategy {
                name: "Ivy instruction reversal".to_string(),
                target: ObfuscationTechnique::TemplateCompilation,
                approach: "Parse Ivy instructions and reconstruct template HTML".to_string(),
                success_rate: 0.75,
                priority: 6,
                requirements: vec!["Angular Ivy knowledge".to_string()],
                limitations: vec!["Complex directives may be difficult".to_string()],
            }],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["webpack".to_string(), "rxjs".to_string()],
            last_updated: "2026-01-07".to_string(),
        });
    }

    /// Initialize Chinese frameworks
    fn init_chinese_frameworks(&mut self) {
        // Taro (京东)
        self.add_framework(FrameworkKnowledge {
            id: "taro".to_string(),
            name: "Taro".to_string(),
            category: FrameworkCategory::MobileCrossPlatform,
            origin: "China".to_string(),
            maintainer: "JD.com (京东)".to_string(),
            signatures: vec![
                ObfuscationSignature {
                    name: "Taro imports".to_string(),
                    pattern_type: SignatureType::ImportStatement,
                    pattern: r"@tarojs".to_string(),
                    weight: 0.95,
                    required: true,
                    context: "Taro package imports".to_string(),
                },
                ObfuscationSignature {
                    name: "Taro components".to_string(),
                    pattern_type: SignatureType::StringLiteral,
                    pattern: r"Taro\.Component".to_string(),
                    weight: 0.9,
                    required: false,
                    context: "Taro class components".to_string(),
                },
            ],
            obfuscation_patterns: vec![ObfuscationPattern {
                name: "Mini-program API wrapping".to_string(),
                technique: ObfuscationTechnique::ProxyFunctions,
                example_obfuscated: r#"Taro.request({url: "api", success: fn})"#.to_string(),
                example_deobfuscated: r#"fetch("api").then(fn)"#.to_string(),
                complexity: 4,
                prevalence: 0.9,
                detection_hints: vec!["Taro.* API calls".to_string()],
            }],
            strategies: vec![DeobfuscationStrategy {
                name: "Taro API conversion".to_string(),
                target: ObfuscationTechnique::ProxyFunctions,
                approach: "Convert Taro APIs to standard web APIs".to_string(),
                success_rate: 0.85,
                priority: 7,
                requirements: vec!["Taro API mapping".to_string()],
                limitations: vec!["Platform-specific features may not translate".to_string()],
            }],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["react".to_string(), "webpack".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // Uni-app (DCloud)
        self.add_framework(FrameworkKnowledge {
            id: "uni-app".to_string(),
            name: "Uni-app".to_string(),
            category: FrameworkCategory::MobileCrossPlatform,
            origin: "China".to_string(),
            maintainer: "DCloud (数字天堂)".to_string(),
            signatures: vec![
                ObfuscationSignature {
                    name: "uni API calls".to_string(),
                    pattern_type: SignatureType::FunctionCall,
                    pattern: r"uni\.(request|navigateTo|showToast)".to_string(),
                    weight: 0.95,
                    required: true,
                    context: "Uni-app unified API".to_string(),
                },
                ObfuscationSignature {
                    name: "DCloud imports".to_string(),
                    pattern_type: SignatureType::ImportStatement,
                    pattern: r"@dcloudio".to_string(),
                    weight: 0.9,
                    required: false,
                    context: "DCloud package imports".to_string(),
                },
            ],
            obfuscation_patterns: vec![ObfuscationPattern {
                name: "Uni API to web API".to_string(),
                technique: ObfuscationTechnique::ProxyFunctions,
                example_obfuscated: r#"uni.request({url: "/api", method: "GET", success: cb})"#
                    .to_string(),
                example_deobfuscated: r#"fetch("/api", {method: "GET"}).then(cb)"#.to_string(),
                complexity: 4,
                prevalence: 1.0,
                detection_hints: vec!["uni.* function calls".to_string()],
            }],
            strategies: vec![DeobfuscationStrategy {
                name: "Uni-app to web standard".to_string(),
                target: ObfuscationTechnique::ProxyFunctions,
                approach: "Map uni.* APIs to web standards".to_string(),
                success_rate: 0.80,
                priority: 8,
                requirements: vec!["Uni-app API reference".to_string()],
                limitations: vec!["Native platform features cannot be converted".to_string()],
            }],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["vue".to_string(), "vite".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // Rax (阿里巴巴)
        self.add_framework(FrameworkKnowledge {
            id: "rax".to_string(),
            name: "Rax".to_string(),
            category: FrameworkCategory::FrontendFramework,
            origin: "China".to_string(),
            maintainer: "Alibaba (阿里巴巴)".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Rax createElement".to_string(),
                pattern_type: SignatureType::ImportStatement,
                pattern: r#"from\s+['"]rax['"]"#.to_string(),
                weight: 0.95,
                required: true,
                context: "Rax framework import".to_string(),
            }],
            obfuscation_patterns: vec![ObfuscationPattern {
                name: "Rax JSX compilation".to_string(),
                technique: ObfuscationTechnique::TemplateCompilation,
                example_obfuscated: r#"createElement("div", {style: styles.container}, "Hello")"#
                    .to_string(),
                example_deobfuscated: r#"<div style={styles.container}>Hello</div>"#.to_string(),
                complexity: 3,
                prevalence: 1.0,
                detection_hints: vec!["Similar to React patterns".to_string()],
            }],
            strategies: vec![DeobfuscationStrategy {
                name: "Rax to JSX".to_string(),
                target: ObfuscationTechnique::TemplateCompilation,
                approach: "Similar to React, convert createElement to JSX".to_string(),
                success_rate: 0.85,
                priority: 6,
                requirements: vec!["React-like parsing".to_string()],
                limitations: vec!["Rax-specific hooks may differ".to_string()],
            }],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["react".to_string(), "webpack".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // San (百度)
        self.add_framework(FrameworkKnowledge {
            id: "san".to_string(),
            name: "San".to_string(),
            category: FrameworkCategory::FrontendFramework,
            origin: "China".to_string(),
            maintainer: "Baidu (百度)".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "San Component".to_string(),
                pattern_type: SignatureType::StringLiteral,
                pattern: r"san\.Component|san\.defineComponent".to_string(),
                weight: 0.95,
                required: true,
                context: "San component definition".to_string(),
            }],
            obfuscation_patterns: vec![ObfuscationPattern {
                name: "San template compilation".to_string(),
                technique: ObfuscationTechnique::TemplateCompilation,
                example_obfuscated: r#"template: "<div>{{message}}</div>""#.to_string(),
                example_deobfuscated: r#"<div>{{message}}</div>"#.to_string(),
                complexity: 3,
                prevalence: 1.0,
                detection_hints: vec!["Template strings with {{}}".to_string()],
            }],
            strategies: vec![DeobfuscationStrategy {
                name: "San template extraction".to_string(),
                target: ObfuscationTechnique::TemplateCompilation,
                approach: "Extract template strings from component definitions".to_string(),
                success_rate: 0.90,
                priority: 5,
                requirements: vec!["San template syntax knowledge".to_string()],
                limitations: vec!["Complex filters may be challenging".to_string()],
            }],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec![],
            last_updated: "2026-01-07".to_string(),
        });

        // Omi (腾讯)
        self.add_framework(FrameworkKnowledge {
            id: "omi".to_string(),
            name: "Omi".to_string(),
            category: FrameworkCategory::FrontendFramework,
            origin: "China".to_string(),
            maintainer: "Tencent (腾讯)".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Omi WeElement".to_string(),
                pattern_type: SignatureType::StringLiteral,
                pattern: r#"WeElement|define\(['"][\w-]+['"]\)"#.to_string(),
                weight: 0.95,
                required: true,
                context: "Omi Web Components".to_string(),
            }],
            obfuscation_patterns: vec![ObfuscationPattern {
                name: "Omi JSX to Web Components".to_string(),
                technique: ObfuscationTechnique::TemplateCompilation,
                example_obfuscated: r#"h("div", {class: "test"}, "Hello")"#.to_string(),
                example_deobfuscated: r#"<div class="test">Hello</div>"#.to_string(),
                complexity: 3,
                prevalence: 0.9,
                detection_hints: vec!["h() function calls for JSX".to_string()],
            }],
            strategies: vec![DeobfuscationStrategy {
                name: "Omi component extraction".to_string(),
                target: ObfuscationTechnique::TemplateCompilation,
                approach: "Extract templates from h() calls and render methods".to_string(),
                success_rate: 0.80,
                priority: 5,
                requirements: vec!["Web Components knowledge".to_string()],
                limitations: vec!["Shadow DOM specifics may be lost".to_string()],
            }],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["webpack".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // Qiankun (阿里乾坤)
        self.add_framework(FrameworkKnowledge {
            id: "qiankun".to_string(),
            name: "Qiankun (乾坤)".to_string(),
            category: FrameworkCategory::MicroFrontend,
            origin: "China".to_string(),
            maintainer: "Alibaba (阿里巴巴)".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Qiankun micro app registration".to_string(),
                pattern_type: SignatureType::FunctionCall,
                pattern: r"registerMicroApps|start".to_string(),
                weight: 0.95,
                required: true,
                context: "Micro-frontend registration".to_string(),
            }],
            obfuscation_patterns: vec![ObfuscationPattern {
                name: "Micro app lifecycle wrapping".to_string(),
                technique: ObfuscationTechnique::ModuleWrapping,
                example_obfuscated: r#"__INJECTED_PUBLIC_PATH_BY_QIANKUN__"#.to_string(),
                example_deobfuscated: r#"Original app entry"#.to_string(),
                complexity: 5,
                prevalence: 0.9,
                detection_hints: vec!["__INJECTED_*_BY_QIANKUN__".to_string()],
            }],
            strategies: vec![DeobfuscationStrategy {
                name: "Qiankun wrapper removal".to_string(),
                target: ObfuscationTechnique::ModuleWrapping,
                approach: "Remove Qiankun injection markers and extract original app".to_string(),
                success_rate: 0.75,
                priority: 6,
                requirements: vec!["Qiankun lifecycle knowledge".to_string()],
                limitations: vec!["Sandbox isolation may complicate extraction".to_string()],
            }],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.6,
                pattern_match: 0.2,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["single-spa".to_string()],
            last_updated: "2026-01-07".to_string(),
        });
    }

    /// Initialize bundlers and build tools
    fn init_bundlers(&mut self) {
        // Webpack
        self.add_framework(FrameworkKnowledge {
            id: "webpack".to_string(),
            name: "Webpack".to_string(),
            category: FrameworkCategory::Bundler,
            origin: "Global".to_string(),
            maintainer: "Webpack Team".to_string(),
            signatures: vec![
                ObfuscationSignature {
                    name: "Webpack runtime".to_string(),
                    pattern_type: SignatureType::FunctionCall,
                    pattern: r"__webpack_require__".to_string(),
                    weight: 0.95,
                    required: true,
                    context: "Webpack module system".to_string(),
                },
                ObfuscationSignature {
                    name: "Webpack chunks".to_string(),
                    pattern_type: SignatureType::VariableDeclaration,
                    pattern: r"webpackChunk|webpackJsonp".to_string(),
                    weight: 0.9,
                    required: false,
                    context: "Code splitting".to_string(),
                },
            ],
            obfuscation_patterns: vec![
                ObfuscationPattern {
                    name: "Module wrapping".to_string(),
                    technique: ObfuscationTechnique::ModuleWrapping,
                    example_obfuscated: r#"(function(modules) { /* runtime */ })({ 0: function(module, exports, __webpack_require__) { /* code */ } })"#.to_string(),
                    example_deobfuscated: r#"// Module code directly"#.to_string(),
                    complexity: 6,
                    prevalence: 1.0,
                    detection_hints: vec!["IIFE with modules object".to_string()],
                },
                ObfuscationPattern {
                    name: "Webpack 5 module federation".to_string(),
                    technique: ObfuscationTechnique::DynamicImports,
                    example_obfuscated: r#"__webpack_require__.f.remoteEntry"#.to_string(),
                    example_deobfuscated: r#"import("remote/Component")"#.to_string(),
                    complexity: 7,
                    prevalence: 0.3,
                    detection_hints: vec!["Module federation specific markers".to_string()],
                },
            ],
            strategies: vec![
                DeobfuscationStrategy {
                    name: "Webpack unwrapping".to_string(),
                    target: ObfuscationTechnique::ModuleWrapping,
                    approach: "Extract individual modules from bundle and resolve dependencies".to_string(),
                    success_rate: 0.85,
                    priority: 9,
                    requirements: vec!["Module graph analysis".to_string()],
                    limitations: vec!["Dynamic imports may be complex".to_string()],
                },
            ],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["babel".to_string(), "terser".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // Vite
        self.add_framework(FrameworkKnowledge {
            id: "vite".to_string(),
            name: "Vite".to_string(),
            category: FrameworkCategory::Bundler,
            origin: "Global".to_string(),
            maintainer: "Evan You / Vite Team".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Vite client".to_string(),
                pattern_type: SignatureType::StringLiteral,
                pattern: r"__vite|import\.meta\.hot".to_string(),
                weight: 0.9,
                required: false,
                context: "Vite HMR and dev features".to_string(),
            }],
            obfuscation_patterns: vec![ObfuscationPattern {
                name: "ES module native".to_string(),
                technique: ObfuscationTechnique::TreeShaking,
                example_obfuscated: r#"import { used } from './module'; // unused exports removed"#
                    .to_string(),
                example_deobfuscated: r#"// All exports visible"#.to_string(),
                complexity: 4,
                prevalence: 0.9,
                detection_hints: vec!["Clean ES modules with tree-shaking".to_string()],
            }],
            strategies: vec![DeobfuscationStrategy {
                name: "Vite module resolution".to_string(),
                target: ObfuscationTechnique::TreeShaking,
                approach: "Trace import statements to reconstruct module dependencies".to_string(),
                success_rate: 0.90,
                priority: 7,
                requirements: vec!["ES module parser".to_string()],
                limitations: vec!["Dynamic imports need runtime analysis".to_string()],
            }],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.4,
                pattern_match: 0.3,
                contextual: 0.2,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["rollup".to_string(), "esbuild".to_string()],
            last_updated: "2026-01-07".to_string(),
        });
    }

    /// Initialize obfuscator tools
    fn init_obfuscators(&mut self) {
        // javascript-obfuscator
        self.add_framework(FrameworkKnowledge {
            id: "javascript-obfuscator".to_string(),
            name: "javascript-obfuscator".to_string(),
            category: FrameworkCategory::ObfuscatorTool,
            origin: "Global".to_string(),
            maintainer: "sanex3339".to_string(),
            signatures: vec![
                ObfuscationSignature {
                    name: "String array variable".to_string(),
                    pattern_type: SignatureType::VariableDeclaration,
                    pattern: r"var _0x[a-f0-9]{4,}\s*=\s*\[".to_string(),
                    weight: 0.9,
                    required: false,
                    context: "String array obfuscation".to_string(),
                },
                ObfuscationSignature {
                    name: "Self-defending check".to_string(),
                    pattern_type: SignatureType::StringLiteral,
                    pattern: r"native code|Function\(.*toString".to_string(),
                    weight: 0.8,
                    required: false,
                    context: "Anti-tampering code".to_string(),
                },
            ],
            obfuscation_patterns: vec![
                ObfuscationPattern {
                    name: "String array with rotation".to_string(),
                    technique: ObfuscationTechnique::StringArrayRotation,
                    example_obfuscated: r#"var _0xabcd = ['str1', 'str2']; (function(_0x, _0x2) { var _0x3 = function(_0x4) { while (--_0x4) { _0x.push(_0x.shift()); } }; _0x3(++_0x2); }(_0xabcd, 0x123));"#.to_string(),
                    example_deobfuscated: r#"// Direct string usage"#.to_string(),
                    complexity: 8,
                    prevalence: 0.95,
                    detection_hints: vec!["_0x prefixed variables with array rotation".to_string()],
                },
                ObfuscationPattern {
                    name: "Control flow flattening".to_string(),
                    technique: ObfuscationTechnique::ControlFlowFlattening,
                    example_obfuscated: r#"while (true) { switch (state) { case 0: code1; state = 1; break; case 1: code2; return; } }"#.to_string(),
                    example_deobfuscated: r#"code1; code2;"#.to_string(),
                    complexity: 9,
                    prevalence: 0.7,
                    detection_hints: vec!["While-switch state machine pattern".to_string()],
                },
                ObfuscationPattern {
                    name: "Dead code injection".to_string(),
                    technique: ObfuscationTechnique::DeadCodeInjection,
                    example_obfuscated: r#"if (false) { fakecode(); } realcode();"#.to_string(),
                    example_deobfuscated: r#"realcode();"#.to_string(),
                    complexity: 5,
                    prevalence: 0.8,
                    detection_hints: vec!["if(false) blocks, opaque predicates".to_string()],
                },
            ],
            strategies: vec![
                DeobfuscationStrategy {
                    name: "String array unpacking".to_string(),
                    target: ObfuscationTechnique::StringArrayRotation,
                    approach: "Detect string array, apply rotation, replace references".to_string(),
                    success_rate: 0.90,
                    priority: 10,
                    requirements: vec!["Pattern matching".to_string(), "Array rotation logic".to_string()],
                    limitations: vec!["Complex rotation functions may vary".to_string()],
                },
                DeobfuscationStrategy {
                    name: "Control flow unflattening".to_string(),
                    target: ObfuscationTechnique::ControlFlowFlattening,
                    approach: "Trace state transitions and rebuild sequential code".to_string(),
                    success_rate: 0.70,
                    priority: 9,
                    requirements: vec!["Control flow analysis".to_string(), "State machine reversal".to_string()],
                    limitations: vec!["Complex nested switches challenging".to_string()],
                },
                DeobfuscationStrategy {
                    name: "Dead code elimination".to_string(),
                    target: ObfuscationTechnique::DeadCodeInjection,
                    approach: "Static analysis to identify and remove unreachable code".to_string(),
                    success_rate: 0.95,
                    priority: 8,
                    requirements: vec!["AST analysis".to_string()],
                    limitations: vec!["Dynamic conditions may be tricky".to_string()],
                },
            ],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.4,
                pattern_match: 0.4,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec![],
            last_updated: "2026-01-07".to_string(),
        });

        // Terser / UglifyJS
        self.add_framework(FrameworkKnowledge {
            id: "terser".to_string(),
            name: "Terser / UglifyJS".to_string(),
            category: FrameworkCategory::ObfuscatorTool,
            origin: "Global".to_string(),
            maintainer: "Terser Team".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "License banner".to_string(),
                pattern_type: SignatureType::Comment,
                pattern: r"/\*! (For license|Copyright)".to_string(),
                weight: 0.6,
                required: false,
                context: "Preserved comments".to_string(),
            }],
            obfuscation_patterns: vec![
                ObfuscationPattern {
                    name: "Variable name mangling".to_string(),
                    technique: ObfuscationTechnique::NameMangling,
                    example_obfuscated: r#"function a(b,c){return b+c}"#.to_string(),
                    example_deobfuscated: r#"function add(x, y) { return x + y; }"#.to_string(),
                    complexity: 4,
                    prevalence: 1.0,
                    detection_hints: vec!["Short single-letter identifiers".to_string()],
                },
                ObfuscationPattern {
                    name: "Whitespace removal".to_string(),
                    technique: ObfuscationTechnique::Minification,
                    example_obfuscated: r#"function x(){var a=1;return a}"#.to_string(),
                    example_deobfuscated: r#"function x() { var a = 1; return a; }"#.to_string(),
                    complexity: 2,
                    prevalence: 1.0,
                    detection_hints: vec!["No whitespace, compact code".to_string()],
                },
            ],
            strategies: vec![
                DeobfuscationStrategy {
                    name: "Name restoration (contextual)".to_string(),
                    target: ObfuscationTechnique::NameMangling,
                    approach: "Use context and patterns to suggest meaningful names".to_string(),
                    success_rate: 0.50,
                    priority: 5,
                    requirements: vec!["Semantic analysis".to_string()],
                    limitations: vec!["Cannot recover original names".to_string()],
                },
                DeobfuscationStrategy {
                    name: "Formatting/beautification".to_string(),
                    target: ObfuscationTechnique::Minification,
                    approach: "Add whitespace and indentation for readability".to_string(),
                    success_rate: 1.0,
                    priority: 3,
                    requirements: vec!["JavaScript parser".to_string()],
                    limitations: vec!["None".to_string()],
                },
            ],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.2,
                pattern_match: 0.5,
                contextual: 0.2,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["webpack".to_string()],
            last_updated: "2026-01-07".to_string(),
        });
    }

    /// Initialize meta frameworks (Next.js, Nuxt.js, etc.)
    fn init_meta_frameworks(&mut self) {
        // Next.js
        self.add_framework(FrameworkKnowledge {
            id: "next-js".to_string(),
            name: "Next.js".to_string(),
            category: FrameworkCategory::MetaFramework,
            origin: "USA".to_string(),
            maintainer: "Vercel".to_string(),
            signatures: vec![
                ObfuscationSignature {
                    name: "Next.js runtime".to_string(),
                    pattern_type: SignatureType::StringLiteral,
                    pattern: r"__next|_N_E|__NEXT_DATA__".to_string(),
                    weight: 0.95,
                    required: false,
                    context: "Next.js client runtime".to_string(),
                },
                ObfuscationSignature {
                    name: "Next.js data fetching".to_string(),
                    pattern_type: SignatureType::FunctionCall,
                    pattern: r"getServerSideProps|getStaticProps|getInitialProps".to_string(),
                    weight: 0.9,
                    required: false,
                    context: "Next.js data fetching methods".to_string(),
                },
            ],
            obfuscation_patterns: vec![ObfuscationPattern {
                name: "Hydration markers".to_string(),
                technique: ObfuscationTechnique::TemplateCompilation,
                example_obfuscated: r#"__NEXT_DATA__ = {"props": {...}}"#.to_string(),
                example_deobfuscated: r#"Server-side rendered props"#.to_string(),
                complexity: 5,
                prevalence: 1.0,
                detection_hints: vec!["__NEXT_DATA__ global variable".to_string()],
            }],
            strategies: vec![DeobfuscationStrategy {
                name: "Extract SSR data".to_string(),
                target: ObfuscationTechnique::TemplateCompilation,
                approach: "Parse __NEXT_DATA__ to extract server-rendered props".to_string(),
                success_rate: 0.95,
                priority: 8,
                requirements: vec!["JSON parsing".to_string()],
                limitations: vec!["Client-side only data not available".to_string()],
            }],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["react".to_string(), "webpack".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // Nuxt.js
        self.add_framework(FrameworkKnowledge {
            id: "nuxt-js".to_string(),
            name: "Nuxt.js".to_string(),
            category: FrameworkCategory::MetaFramework,
            origin: "France/Global".to_string(),
            maintainer: "Nuxt Team".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Nuxt context".to_string(),
                pattern_type: SignatureType::StringLiteral,
                pattern: r"\$nuxt|__NUXT__|nuxtServerInit".to_string(),
                weight: 0.95,
                required: false,
                context: "Nuxt.js runtime".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["vue".to_string(), "vite".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // Gatsby
        self.add_framework(FrameworkKnowledge {
            id: "gatsby".to_string(),
            name: "Gatsby".to_string(),
            category: FrameworkCategory::MetaFramework,
            origin: "USA".to_string(),
            maintainer: "Gatsby Team / Netlify".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Gatsby runtime".to_string(),
                pattern_type: SignatureType::StringLiteral,
                pattern: r"___gatsby|gatsby-browser|gatsby-ssr".to_string(),
                weight: 0.95,
                required: false,
                context: "Gatsby static site generation".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["react".to_string(), "webpack".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // SvelteKit
        self.add_framework(FrameworkKnowledge {
            id: "sveltekit".to_string(),
            name: "SvelteKit".to_string(),
            category: FrameworkCategory::MetaFramework,
            origin: "Global".to_string(),
            maintainer: "Svelte Team".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "SvelteKit imports".to_string(),
                pattern_type: SignatureType::ImportStatement,
                pattern: r"@sveltejs/kit|\$app/".to_string(),
                weight: 0.95,
                required: false,
                context: "SvelteKit framework".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["svelte".to_string(), "vite".to_string()],
            last_updated: "2026-01-07".to_string(),
        });
    }

    /// Initialize state management frameworks
    fn init_state_management(&mut self) {
        // Redux
        self.add_framework(FrameworkKnowledge {
            id: "redux".to_string(),
            name: "Redux".to_string(),
            category: FrameworkCategory::StateManagement,
            origin: "USA".to_string(),
            maintainer: "Redux Team".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Redux store".to_string(),
                pattern_type: SignatureType::FunctionCall,
                pattern: r"createStore|combineReducers".to_string(),
                weight: 0.9,
                required: false,
                context: "Redux state management".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["react".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // MobX
        self.add_framework(FrameworkKnowledge {
            id: "mobx".to_string(),
            name: "MobX".to_string(),
            category: FrameworkCategory::StateManagement,
            origin: "Global".to_string(),
            maintainer: "MobX Team".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "MobX observables".to_string(),
                pattern_type: SignatureType::FunctionCall,
                pattern: r"makeObservable|observable|action".to_string(),
                weight: 0.9,
                required: false,
                context: "MobX reactive programming".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["react".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // Zustand
        self.add_framework(FrameworkKnowledge {
            id: "zustand".to_string(),
            name: "Zustand".to_string(),
            category: FrameworkCategory::StateManagement,
            origin: "Global".to_string(),
            maintainer: "Poimandres".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Zustand create".to_string(),
                pattern_type: SignatureType::ImportStatement,
                pattern: r#"from\s+['"]zustand['"]"#.to_string(),
                weight: 0.9,
                required: false,
                context: "Zustand state library".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["react".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // Pinia (Vue)
        self.add_framework(FrameworkKnowledge {
            id: "pinia".to_string(),
            name: "Pinia".to_string(),
            category: FrameworkCategory::StateManagement,
            origin: "France/Global".to_string(),
            maintainer: "Eduardo San Martin Morote".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Pinia store".to_string(),
                pattern_type: SignatureType::FunctionCall,
                pattern: r"defineStore|createPinia".to_string(),
                weight: 0.95,
                required: false,
                context: "Pinia Vue state management".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["vue".to_string()],
            last_updated: "2026-01-07".to_string(),
        });
    }

    /// Initialize UI libraries
    fn init_ui_libraries(&mut self) {
        // Ant Design
        self.add_framework(FrameworkKnowledge {
            id: "antd".to_string(),
            name: "Ant Design".to_string(),
            category: FrameworkCategory::UILibrary,
            origin: "China".to_string(),
            maintainer: "Ant Group (蚂蚁集团)".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Ant Design imports".to_string(),
                pattern_type: SignatureType::ImportStatement,
                pattern: r#"from\s+['"]antd['"]|@ant-design"#.to_string(),
                weight: 0.95,
                required: false,
                context: "Ant Design UI library".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["react".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // Material-UI / MUI
        self.add_framework(FrameworkKnowledge {
            id: "mui".to_string(),
            name: "Material-UI (MUI)".to_string(),
            category: FrameworkCategory::UILibrary,
            origin: "Global".to_string(),
            maintainer: "MUI Team".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "MUI imports".to_string(),
                pattern_type: SignatureType::ImportStatement,
                pattern: r"@mui/|@material-ui/|makeStyles|createTheme".to_string(),
                weight: 0.9,
                required: false,
                context: "Material-UI library".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["react".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // Element UI / Element Plus (饿了么)
        self.add_framework(FrameworkKnowledge {
            id: "element-ui".to_string(),
            name: "Element UI / Element Plus".to_string(),
            category: FrameworkCategory::UILibrary,
            origin: "China".to_string(),
            maintainer: "Ele.me (饿了么)".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Element imports".to_string(),
                pattern_type: SignatureType::ImportStatement,
                pattern: r"element-ui|element-plus".to_string(),
                weight: 0.95,
                required: false,
                context: "Element UI library".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["vue".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // Vant (有赞)
        self.add_framework(FrameworkKnowledge {
            id: "vant".to_string(),
            name: "Vant".to_string(),
            category: FrameworkCategory::UILibrary,
            origin: "China".to_string(),
            maintainer: "Youzan (有赞)".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Vant imports".to_string(),
                pattern_type: SignatureType::ImportStatement,
                pattern: r#"from\s+['"]vant['"]|@vant"#.to_string(),
                weight: 0.95,
                required: false,
                context: "Vant mobile UI".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["vue".to_string()],
            last_updated: "2026-01-07".to_string(),
        });
    }

    /// Initialize additional frameworks (Svelte, Preact, etc.)
    fn init_additional_frameworks(&mut self) {
        // Svelte
        self.add_framework(FrameworkKnowledge {
            id: "svelte".to_string(),
            name: "Svelte".to_string(),
            category: FrameworkCategory::FrontendFramework,
            origin: "USA".to_string(),
            maintainer: "Rich Harris / Svelte Team".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Svelte component".to_string(),
                pattern_type: SignatureType::StringLiteral,
                pattern: r"SvelteComponent|svelte/internal|create_component".to_string(),
                weight: 0.9,
                required: false,
                context: "Svelte compilation".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["rollup".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // Preact
        self.add_framework(FrameworkKnowledge {
            id: "preact".to_string(),
            name: "Preact".to_string(),
            category: FrameworkCategory::FrontendFramework,
            origin: "USA".to_string(),
            maintainer: "Jason Miller / Preact Team".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Preact h function".to_string(),
                pattern_type: SignatureType::ImportStatement,
                pattern: r#"from\s+['"]preact['"]"#.to_string(),
                weight: 0.95,
                required: false,
                context: "Preact lightweight React alternative".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec!["react".to_string()],
            last_updated: "2026-01-07".to_string(),
        });

        // Solid.js
        self.add_framework(FrameworkKnowledge {
            id: "solid-js".to_string(),
            name: "Solid.js".to_string(),
            category: FrameworkCategory::FrontendFramework,
            origin: "USA".to_string(),
            maintainer: "Ryan Carniato / Solid Team".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Solid signals".to_string(),
                pattern_type: SignatureType::FunctionCall,
                pattern: r"createSignal|createEffect|createMemo".to_string(),
                weight: 0.9,
                required: false,
                context: "Solid.js reactive primitives".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec![],
            last_updated: "2026-01-07".to_string(),
        });

        // Alpine.js
        self.add_framework(FrameworkKnowledge {
            id: "alpine-js".to_string(),
            name: "Alpine.js".to_string(),
            category: FrameworkCategory::FrontendFramework,
            origin: "USA".to_string(),
            maintainer: "Caleb Porzio".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Alpine directives".to_string(),
                pattern_type: SignatureType::StringLiteral,
                pattern: r"x-data|x-show|x-if|Alpine\.start".to_string(),
                weight: 0.9,
                required: false,
                context: "Alpine.js lightweight framework".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec![],
            last_updated: "2026-01-07".to_string(),
        });

        // Lit
        self.add_framework(FrameworkKnowledge {
            id: "lit".to_string(),
            name: "Lit".to_string(),
            category: FrameworkCategory::FrontendFramework,
            origin: "USA".to_string(),
            maintainer: "Google".to_string(),
            signatures: vec![ObfuscationSignature {
                name: "Lit element".to_string(),
                pattern_type: SignatureType::StringLiteral,
                pattern: r"LitElement|lit-html|customElement".to_string(),
                weight: 0.9,
                required: false,
                context: "Lit web components".to_string(),
            }],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights {
                signature_match: 0.5,
                pattern_match: 0.3,
                contextual: 0.1,
                related_frameworks: 0.1,
            },
            related_frameworks: vec![],
            last_updated: "2026-01-07".to_string(),
        });
    }

    /// Add a framework to the knowledge base
    fn add_framework(&mut self, framework: FrameworkKnowledge) {
        let id = framework.id.clone();
        let category = framework.category.clone();

        // Add to main map
        self.frameworks.insert(id.clone(), framework);

        // Update category index
        self.category_index.entry(category).or_default().push(id);
    }

    /// Build search indices
    fn build_indices(&mut self) {
        // Build signature index for fast pattern matching
        for (id, framework) in &self.frameworks {
            for signature in &framework.signatures {
                self.signature_index
                    .entry(signature.pattern.clone())
                    .or_default()
                    .push(id.clone());
            }
        }
    }

    /// Get framework by ID
    pub fn get_framework(&self, id: &str) -> Option<&FrameworkKnowledge> {
        self.frameworks.get(id)
    }

    /// Get all frameworks in a category
    pub fn get_frameworks_by_category(
        &self,
        category: &FrameworkCategory,
    ) -> Vec<&FrameworkKnowledge> {
        self.category_index
            .get(category)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.frameworks.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Analyze code and return detected frameworks
    pub fn analyze_code(&self, code: &str) -> Result<Vec<DetectionResult>> {
        let mut results = Vec::new();

        for (id, framework) in &self.frameworks {
            let mut score = 0.0;
            let mut matched_signatures = Vec::new();

            // Check signatures
            for signature in &framework.signatures {
                if self.matches_signature(code, signature) {
                    score += signature.weight;
                    matched_signatures.push(signature.name.clone());
                }
            }

            // Calculate confidence
            let max_score: f32 = framework.signatures.iter().map(|s| s.weight).sum();
            let confidence = if max_score > 0.0 {
                (score / max_score) * framework.confidence_weights.signature_match
            } else {
                0.0
            };

            if confidence > 0.1 {
                results.push(DetectionResult {
                    framework_id: id.clone(),
                    framework_name: framework.name.clone(),
                    confidence,
                    matched_signatures,
                    applicable_strategies: framework.strategies.clone(),
                });
            }
        }

        // Sort by confidence
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(results)
    }

    /// Check if code matches a signature
    fn matches_signature(&self, code: &str, signature: &ObfuscationSignature) -> bool {
        match signature.pattern_type {
            SignatureType::StringLiteral => code.contains(&signature.pattern),
            SignatureType::Regex => {
                use regex::Regex;
                Regex::new(&signature.pattern)
                    .ok()
                    .map(|re| re.is_match(code))
                    .unwrap_or(false)
            }
            SignatureType::ImportStatement
            | SignatureType::FunctionCall
            | SignatureType::VariableDeclaration => {
                use regex::Regex;
                Regex::new(&signature.pattern)
                    .ok()
                    .map(|re| re.is_match(code))
                    .unwrap_or(false)
            }
            _ => false,
        }
    }

    /// Get all frameworks count
    pub fn framework_count(&self) -> usize {
        self.frameworks.len()
    }

    /// Get statistics
    pub fn get_statistics(&self) -> KnowledgeBaseStats {
        let total_frameworks = self.frameworks.len();
        let total_signatures: usize = self.frameworks.values().map(|f| f.signatures.len()).sum();
        let total_patterns: usize = self
            .frameworks
            .values()
            .map(|f| f.obfuscation_patterns.len())
            .sum();
        let total_strategies: usize = self.frameworks.values().map(|f| f.strategies.len()).sum();

        let mut category_counts = HashMap::new();
        for framework in self.frameworks.values() {
            *category_counts
                .entry(framework.category.clone())
                .or_insert(0) += 1;
        }

        KnowledgeBaseStats {
            total_frameworks,
            total_signatures,
            total_patterns,
            total_strategies,
            category_counts,
        }
    }
}

/// Detection result
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Framework ID
    pub framework_id: String,
    /// Framework name
    pub framework_name: String,
    /// Detection confidence (0.0-1.0)
    pub confidence: f32,
    /// Matched signatures
    pub matched_signatures: Vec<String>,
    /// Applicable deobfuscation strategies
    pub applicable_strategies: Vec<DeobfuscationStrategy>,
}

/// Knowledge base statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseStats {
    /// Total number of frameworks
    pub total_frameworks: usize,
    /// Total signatures across all frameworks
    pub total_signatures: usize,
    /// Total obfuscation patterns
    pub total_patterns: usize,
    /// Total deobfuscation strategies
    pub total_strategies: usize,
    /// Frameworks per category
    pub category_counts: HashMap<FrameworkCategory, usize>,
}

impl Default for FrameworkKnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_base_initialization() {
        let kb = FrameworkKnowledgeBase::new();
        assert!(kb.framework_count() > 0);
    }

    #[test]
    fn test_react_detection() {
        let kb = FrameworkKnowledgeBase::new();
        let code = r#"
            import React from 'react';
            function App() {
                return React.createElement("div", null, "Hello");
            }
        "#;

        let results = kb.analyze_code(code).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.framework_id == "react"));
    }

    #[test]
    fn test_vue_detection() {
        let kb = FrameworkKnowledgeBase::new();
        let code = r#"
            const _hoisted_1 = { class: "container" };
            function render() {
                return _createElementVNode("div", _hoisted_1, "Hello");
            }
        "#;

        let results = kb.analyze_code(code).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.framework_id == "vue"));
    }

    #[test]
    fn test_webpack_detection() {
        let kb = FrameworkKnowledgeBase::new();
        let code = r#"
            (function(modules) {
                function __webpack_require__(moduleId) {
                    return modules[moduleId].call();
                }
            })([function() { }]);
        "#;

        let results = kb.analyze_code(code).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.framework_id == "webpack"));
    }

    #[test]
    fn test_chinese_framework_detection() {
        let kb = FrameworkKnowledgeBase::new();

        // Test Taro
        let taro_code = r#"
            import Taro from '@tarojs/taro';
            Taro.request({ url: '/api' });
        "#;
        let results = kb.analyze_code(taro_code).unwrap();
        assert!(results.iter().any(|r| r.framework_id == "taro"));

        // Test Uni-app
        let uni_code = r#"
            uni.request({ url: '/api', success: () => {} });
        "#;
        let results = kb.analyze_code(uni_code).unwrap();
        assert!(results.iter().any(|r| r.framework_id == "uni-app"));
    }

    #[test]
    fn test_obfuscator_detection() {
        let kb = FrameworkKnowledgeBase::new();
        let code = r#"
            var _0xabcd = ['string1', 'string2', 'string3'];
            function test() {
                console.log(_0xabcd[0]);
            }
        "#;

        let results = kb.analyze_code(code).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_get_framework_by_category() {
        let kb = FrameworkKnowledgeBase::new();
        let bundlers = kb.get_frameworks_by_category(&FrameworkCategory::Bundler);
        assert!(!bundlers.is_empty());

        let chinese_frameworks =
            kb.get_frameworks_by_category(&FrameworkCategory::MobileCrossPlatform);
        assert!(!chinese_frameworks.is_empty());
    }

    #[test]
    fn test_statistics() {
        let kb = FrameworkKnowledgeBase::new();
        let stats = kb.get_statistics();

        assert!(stats.total_frameworks > 10);
        assert!(stats.total_signatures > 0);
        assert!(stats.total_patterns > 0);
        assert!(stats.total_strategies > 0);
        assert!(!stats.category_counts.is_empty());
    }
}
