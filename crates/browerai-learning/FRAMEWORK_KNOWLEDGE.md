# Framework Knowledge Base

## Overview

The Framework Knowledge Base is a comprehensive system for detecting and analyzing obfuscation patterns from JavaScript frameworks worldwide. It currently supports **30+ frameworks** with detailed knowledge about their compilation patterns, bundling strategies, and obfuscation techniques.

## Features

### Framework Detection
- **30 frameworks** across 7 categories
- **40 detection signatures** with pattern matching
- **Confidence scoring** based on matched signatures
- **Related framework tracking** for ecosystem understanding

### Obfuscation Analysis
- **20 documented patterns** with before/after examples
- **17 deobfuscation strategies** with success rates
- **Complexity scoring** (1-10 scale)
- **Prevalence tracking** (0.0-1.0) for each pattern

### Categories Covered
- **Bundlers**: Webpack, Vite
- **Frontend Frameworks**: React, Vue, Angular, Svelte, Preact, Solid.js, Alpine.js, Lit, etc.
- **Meta Frameworks**: Next.js, Nuxt.js, Gatsby, SvelteKit
- **State Management**: Redux, MobX, Zustand, Pinia
- **UI Libraries**: Material-UI, Ant Design, Element UI, Vant
- **Obfuscators**: javascript-obfuscator, Terser/UglifyJS
- **Mobile/Cross-platform**: Taro, Uni-app
- **Micro-frontends**: Qiankun

## Usage

### Basic Detection

```rust
use browerai_learning::FrameworkKnowledgeBase;

// Initialize knowledge base
let kb = FrameworkKnowledgeBase::new();

// Analyze JavaScript code
let code = r#"
    import React from 'react';
    function App() {
        return React.createElement("div", null, "Hello");
    }
"#;

let results = kb.analyze_code(code)?;
for result in results {
    println!("Detected: {} (confidence: {:.1}%)", 
        result.framework_name, 
        result.confidence * 100.0
    );
}
```

### Get Framework Information

```rust
// Get framework by ID
if let Some(framework) = kb.get_framework("react") {
    println!("Name: {}", framework.name);
    println!("Origin: {}", framework.origin);
    println!("Maintainer: {}", framework.maintainer);
    println!("Signatures: {}", framework.signatures.len());
    println!("Patterns: {}", framework.obfuscation_patterns.len());
}
```

### Browse by Category

```rust
use browerai_learning::FrameworkCategory;

// Get all Chinese mobile frameworks
let mobile = kb.get_frameworks_by_category(&FrameworkCategory::MobileCrossPlatform);
for framework in mobile {
    println!("{} - {}", framework.name, framework.maintainer);
}
```

### Get Statistics

```rust
let stats = kb.get_statistics();
println!("Total frameworks: {}", stats.total_frameworks);
println!("Total signatures: {}", stats.total_signatures);
println!("Total patterns: {}", stats.total_patterns);
println!("Total strategies: {}", stats.total_strategies);
```

## Supported Frameworks

### Global Frameworks

#### Frontend Frameworks
- **React** (Meta, USA) - JSX compilation, createElement patterns
- **Vue.js** (Evan You, China/Global) - Template compilation, hoisted variables
- **Angular** (Google, USA) - Ivy compilation with ɵɵ markers
- **Svelte** (Rich Harris, USA) - Component compilation
- **Preact** (Jason Miller, USA) - Lightweight React alternative
- **Solid.js** (Ryan Carniato, USA) - Fine-grained reactivity
- **Alpine.js** (Caleb Porzio, USA) - Lightweight framework with directives
- **Lit** (Google, USA) - Web Components with lit-html

#### Meta Frameworks
- **Next.js** (Vercel, USA) - React SSR/SSG with hydration
- **Nuxt.js** (Nuxt Team, France) - Vue SSR/SSG
- **Gatsby** (Gatsby/Netlify, USA) - React static site generation
- **SvelteKit** (Svelte Team, Global) - Svelte application framework

#### Bundlers & Build Tools
- **Webpack** (Webpack Team, Global) - Module bundler with chunking
- **Vite** (Evan You, Global) - Fast ES module bundler

#### State Management
- **Redux** (Redux Team, USA) - Predictable state container
- **MobX** (MobX Team, Global) - Simple reactive programming
- **Zustand** (Poimandres, Global) - Lightweight state library
- **Pinia** (Eduardo San Martin, France) - Vue state management

#### UI Libraries
- **Material-UI / MUI** (MUI Team, Global) - React Material Design
- **Ant Design** (Ant Group, China) - Enterprise UI library

### Chinese Frameworks

#### Mobile & Cross-platform
- **Taro** (JD.com 京东, China) - Multi-platform framework
- **Uni-app** (DCloud 数字天堂, China) - Cross-platform Vue framework

#### Frontend
- **Rax** (Alibaba 阿里, China) - Lightweight React-like framework
- **San** (Baidu 百度, China) - MVVM framework
- **Omi** (Tencent 腾讯, China) - Web Components framework

#### UI Libraries
- **Element UI / Element Plus** (Ele.me 饿了么, China) - Vue UI library
- **Vant** (Youzan 有赞, China) - Mobile UI library

#### Micro-frontends
- **Qiankun** (Alibaba 阿里, China) - Micro-frontend framework

### Obfuscators
- **javascript-obfuscator** (sanex3339, Global) - Advanced obfuscation
- **Terser / UglifyJS** (Terser Team, Global) - Minification and mangling

## Obfuscation Techniques

The system recognizes 23 obfuscation techniques:

1. **Name Mangling** - Variable/function name shortening
2. **String Encoding** - Hex, base64, unicode encoding
3. **Control Flow Flattening** - Switch-based state machines
4. **Dead Code Injection** - Unreachable code blocks
5. **Opaque Predicates** - Always true/false conditions
6. **String Array Rotation** - Rotated string arrays with decoder
7. **Proxy Functions** - Wrapper functions
8. **Self-Defending Code** - Anti-tampering checks
9. **Anti-Debugging** - Debugger detection
10. **Domain Locking** - Domain verification
11. **Code Splitting** - Dynamic imports
12. **Lazy Loading** - On-demand loading
13. **Module Wrapping** - IIFE/UMD patterns
14. **Template Compilation** - JSX/Vue template to JS
15. **Source Map Removal** - Debug info removal
16. **Minification** - Whitespace and comment removal
17. **Tree Shaking** - Unused code elimination
18. **Dynamic Imports** - Runtime module loading
19. **Webpack Chunking** - Code splitting
20. **Constant Folding** - Compile-time evaluation
21. **Function Inlining** - Function body substitution
22. **Property Mangling** - Object property renaming
23. **Unicode Escaping** - Character encoding

## Deobfuscation Strategies

Each framework has specific deobfuscation strategies with:
- **Success Rate**: 0.0-1.0 (expected effectiveness)
- **Priority**: 1-10 (implementation priority)
- **Requirements**: Tools/knowledge needed
- **Limitations**: Known constraints

Example strategies:
- **String Array Unpacking** (90% success, priority 10)
- **Control Flow Unflattening** (70% success, priority 9)
- **Dead Code Elimination** (95% success, priority 8)
- **JSX Reconstruction** (85% success, priority 8)
- **Vue Template Reconstruction** (80% success, priority 7)

## Architecture

### Data Structures

```rust
pub struct FrameworkKnowledge {
    pub id: String,
    pub name: String,
    pub category: FrameworkCategory,
    pub origin: String,
    pub maintainer: String,
    pub signatures: Vec<ObfuscationSignature>,
    pub obfuscation_patterns: Vec<ObfuscationPattern>,
    pub strategies: Vec<DeobfuscationStrategy>,
    pub confidence_weights: ConfidenceWeights,
    pub related_frameworks: Vec<String>,
    pub last_updated: String,
}

pub struct ObfuscationSignature {
    pub name: String,
    pub pattern_type: SignatureType,
    pub pattern: String,
    pub weight: f32,
    pub required: bool,
    pub context: String,
}

pub struct ObfuscationPattern {
    pub name: String,
    pub technique: ObfuscationTechnique,
    pub example_obfuscated: String,
    pub example_deobfuscated: String,
    pub complexity: u8,
    pub prevalence: f32,
    pub detection_hints: Vec<String>,
}

pub struct DeobfuscationStrategy {
    pub name: String,
    pub target: ObfuscationTechnique,
    pub approach: String,
    pub success_rate: f32,
    pub priority: u8,
    pub requirements: Vec<String>,
    pub limitations: Vec<String>,
}
```

### Detection Process

1. **Signature Matching**: Check code against framework signatures
2. **Pattern Recognition**: Identify obfuscation patterns
3. **Confidence Scoring**: Calculate detection confidence
4. **Strategy Selection**: Recommend deobfuscation approaches

## Extending the System

### Adding a New Framework

```rust
self.add_framework(FrameworkKnowledge {
    id: "my-framework".to_string(),
    name: "My Framework".to_string(),
    category: FrameworkCategory::FrontendFramework,
    origin: "Country".to_string(),
    maintainer: "Company/Developer".to_string(),
    signatures: vec![
        ObfuscationSignature {
            name: "Unique pattern".to_string(),
            pattern_type: SignatureType::StringLiteral,
            pattern: r"myFramework\.init".to_string(),
            weight: 0.9,
            required: true,
            context: "Framework initialization".to_string(),
        },
    ],
    obfuscation_patterns: vec![],
    strategies: vec![],
    confidence_weights: ConfidenceWeights::default(),
    related_frameworks: vec![],
    last_updated: "2026-01-07".to_string(),
});
```

### Adding Detection Patterns

1. Study framework compilation output
2. Identify unique signatures
3. Test against real-world code
4. Add to knowledge base with appropriate weights

## Examples

See `examples/framework_knowledge_demo.rs` for a comprehensive demonstration including:
- React detection
- Vue 3 detection
- Webpack bundle detection
- Chinese framework detection (Taro, Uni-app)
- Obfuscator detection (javascript-obfuscator)
- Category browsing

Run the demo:
```bash
cargo run --package browerai --example framework_knowledge_demo
```

## Performance

- **Fast initialization**: Knowledge base loads in milliseconds
- **Efficient matching**: Regex-based signature detection
- **Scalable**: Can handle hundreds of frameworks
- **Memory efficient**: Lazy evaluation of patterns

## Future Enhancements

1. **Machine Learning Integration**: Train models on detection patterns
2. **Real-time Updates**: Download framework updates from registry
3. **Custom Frameworks**: User-defined framework knowledge
4. **Advanced Patterns**: More complex obfuscation techniques
5. **Automated Testing**: Continuous validation against real codebases
6. **API Integration**: Query framework databases for latest patterns

## Contributing

To add a new framework:
1. Research framework compilation patterns
2. Collect real-world code samples
3. Identify unique signatures
4. Document obfuscation techniques
5. Define deobfuscation strategies
6. Add tests
7. Update documentation

## License

Part of the BrowerAI project, licensed under the same terms.
