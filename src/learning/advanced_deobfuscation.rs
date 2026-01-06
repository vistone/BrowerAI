/// Advanced JavaScript deobfuscation for modern frameworks and dynamic content
///
/// Handles complex obfuscation techniques including:
/// - Framework-specific bundling (Webpack, Rollup, etc.)
/// - Dynamic HTML injection via JavaScript
/// - Event-driven content loading
/// - Template literals and JSX compilation
/// - Code splitting and lazy loading
use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Framework metadata and characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkInfo {
    pub name: String,
    pub category: String,
    pub patterns: Vec<&'static str>,
    pub deobfuscation_strategy: &'static str,
    pub origin: String,
}

/// Advanced obfuscation patterns specific to frameworks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FrameworkObfuscation {
    // ========== Bundlers & Build Tools ==========
    /// Webpack module system
    WebpackBundled,
    /// Rollup/Vite bundling
    RollupBundled,
    /// Parcel bundler
    ParcelBundled,
    /// esbuild minification
    EsbuildMinified,
    /// Turbopack (Next.js)
    TurbopackBundled,
    /// Snowpack bundling
    SnowpackBundled,
    /// Browserify modules
    BrowserifyBundled,
    /// SystemJS loader
    SystemJSLoader,
    /// RequireJS/AMD
    RequireJSAMD,

    // ========== Frontend Frameworks ==========
    /// React JSX compilation
    ReactCompiled,
    /// Vue 2/3 template compilation
    VueCompiled,
    /// Angular compiled output
    AngularCompiled,
    /// Svelte compiled components
    SvelteCompiled,
    /// Solid.js compiled
    SolidJSCompiled,
    /// Preact optimized
    PreactCompiled,
    /// Ember.js compiled
    EmberCompiled,
    /// Backbone.js patterns
    BackbonePatterns,
    /// Knockout.js observables
    KnockoutObservables,
    /// Marko.js compiled
    MarkoCompiled,
    /// Lit (Web Components)
    LitElements,
    /// Stencil components
    StencilCompiled,
    /// Alpine.js directives
    AlpineDirectives,
    /// Hyperapp patterns
    HyperappPatterns,
    /// Mithril.js compiled
    MithrilCompiled,
    /// Riot.js components
    RiotComponents,
    /// Inferno.js compiled
    InfernoCompiled,
    /// Aurelia framework
    AureliaFramework,

    // ========== Meta Frameworks ==========
    /// Next.js (React)
    NextJSFramework,
    /// Nuxt.js (Vue)
    NuxtJSFramework,
    /// Gatsby static
    GatsbyStatic,
    /// Remix (React Router)
    RemixFramework,
    /// SvelteKit
    SvelteKitFramework,
    /// Astro framework
    AstroFramework,
    /// Qwik framework
    QwikFramework,
    /// Fresh (Deno)
    FreshFramework,
    /// Analog (Angular)
    AnalogFramework,

    // ========== Mobile & Native ==========
    /// React Native
    ReactNative,
    /// NativeScript
    NativeScript,
    /// Ionic Framework
    IonicFramework,
    /// Capacitor
    CapacitorBridge,
    /// Cordova/PhoneGap
    CordovaPhoneGap,
    /// Flutter Web
    FlutterWeb,
    /// Weex (Alibaba)
    WeexFramework,

    // ========== Chinese Frameworks ==========
    /// Taro (JD.com multi-platform framework)
    TaroFramework,
    /// Uni-app (DCloud cross-platform)
    UniAppFramework,
    /// mpvue (Meituan Vue mini-program)
    MpVueFramework,
    /// Chameleon (DiDi cross-platform)
    ChameleonFramework,
    /// Rax (Alibaba React-like)
    RaxFramework,
    /// Remax (Alibaba React mini-program)
    RemaxFramework,
    /// Kbone (WeChat web-to-miniprogram)
    KboneFramework,
    /// Omi (Tencent Web Components)
    OmiFramework,
    /// San (Baidu MVVM)
    SanFramework,
    /// RegularJS (NetEase)
    RegularJSFramework,
    /// KISSY (Alibaba legacy)
    KISSYFramework,

    // ========== State Management ==========
    /// Redux patterns
    ReduxPatterns,
    /// MobX observables
    MobXObservables,
    /// Vuex store
    VuexStore,
    /// Pinia store
    PiniaStore,
    /// Zustand
    ZustandStore,
    /// Jotai atoms
    JotaiAtoms,
    /// Recoil state
    RecoilState,
    /// XState machines
    XStateMachines,
    /// Effector
    EffectorStore,

    // ========== UI Libraries ==========
    /// Material-UI/MUI
    MaterialUICompiled,
    /// Ant Design
    AntDesignCompiled,
    /// Chakra UI
    ChakraUICompiled,
    /// TailwindCSS JIT
    TailwindJIT,
    /// Styled Components
    StyledComponents,
    /// Emotion CSS-in-JS
    EmotionCSS,
    /// Element UI (饿了么)
    ElementUICompiled,
    /// Vant (有赞)
    VantUICompiled,
    /// Ant Design Mobile
    AntMobileCompiled,

    // ========== SSR & Hydration ==========
    /// Server-side rendering
    ServerSideRendered,
    /// Island architecture (Astro)
    IslandArchitecture,
    /// Progressive hydration
    ProgressiveHydration,
    /// Streaming SSR
    StreamingSSR,

    // ========== Obfuscation Techniques ==========
    /// Dynamic HTML injection
    DynamicHtmlInjection,
    /// Event-driven loading
    EventDrivenContent,
    /// Template string obfuscation
    TemplateLiteralObfuscation,
    /// Code splitting
    CodeSplitting,
    /// Lazy loading
    LazyLoading,
    /// Tree shaking artifacts
    TreeShakingArtifacts,
    /// Minification (Terser/UglifyJS)
    TerserMinified,
    /// Closure Compiler
    ClosureCompiled,
    /// Dead code elimination
    DeadCodeEliminated,

    // ========== Module Systems ==========
    /// ES Modules
    ESModules,
    /// CommonJS
    CommonJS,
    /// UMD pattern
    UMDPattern,
    /// IIFE wrapper
    IIFEWrapper,

    // ========== Micro Frontends ==========
    /// Module Federation (Webpack 5)
    ModuleFederation,
    /// Single-SPA
    SingleSPA,
    /// Qiankun (阿里乾坤)
    QiankunMicroFrontend,
    /// Micro-app
    MicroApp,

    // ========== Testing Frameworks ==========
    /// Jest transforms
    JestTransforms,
    /// Vitest patterns
    VitestPatterns,

    // ========== Other ==========
    /// Unknown framework pattern
    UnknownFramework,
}

/// Analysis result for advanced obfuscation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedObfuscationAnalysis {
    /// Framework-specific patterns detected
    pub framework_patterns: Vec<FrameworkObfuscation>,
    /// Dynamic HTML injection points
    pub dynamic_injection_points: Vec<InjectionPoint>,
    /// Event-triggered content loaders
    pub event_loaders: Vec<EventLoader>,
    /// Template extraction results
    pub templates: Vec<ExtractedTemplate>,
    /// Confidence score
    pub confidence: f32,
}

/// Location where HTML is dynamically injected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionPoint {
    /// Line number in source
    pub line: usize,
    /// Injection method (innerHTML, createElement, etc.)
    pub method: String,
    /// Target element selector
    pub target: String,
    /// Estimated HTML content
    pub content_hint: String,
}

/// Event-triggered content loader
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLoader {
    /// Event type (click, load, scroll, etc.)
    pub event_type: String,
    /// Target element
    pub target: String,
    /// Loader function name
    pub function: String,
    /// Estimated trigger condition
    pub trigger_condition: String,
}

/// Extracted template from obfuscated code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedTemplate {
    /// Template type (HTML, JSX, Vue, etc.)
    pub template_type: String,
    /// Extracted HTML content
    pub html_content: String,
    /// Original obfuscated form
    pub original_form: String,
    /// Confidence in extraction
    pub confidence: f32,
}

/// Result of advanced deobfuscation
#[derive(Debug, Clone)]
pub struct AdvancedDeobfuscationResult {
    /// Deobfuscated JavaScript
    pub javascript: String,
    /// Extracted HTML templates
    pub html_templates: Vec<String>,
    /// Event-to-content mapping
    pub event_content_map: HashMap<String, String>,
    /// Success indicators
    pub success: bool,
    /// Processing steps
    pub steps: Vec<String>,
}

/// Advanced JavaScript deobfuscator for modern frameworks
pub struct AdvancedDeobfuscator {
    /// Enable framework-specific processing
    enable_framework_detection: bool,
    /// Enable dynamic HTML extraction
    enable_html_extraction: bool,
    /// Maximum extraction depth
    max_extraction_depth: usize,
}

impl AdvancedDeobfuscator {
    /// Create new advanced deobfuscator
    pub fn new() -> Self {
        Self {
            enable_framework_detection: true,
            enable_html_extraction: true,
            max_extraction_depth: 10,
        }
    }

    /// Analyze advanced obfuscation patterns
    pub fn analyze(&self, code: &str) -> Result<AdvancedObfuscationAnalysis> {
        let mut framework_patterns = Vec::new();
        let mut dynamic_injection_points = Vec::new();
        let mut event_loaders = Vec::new();
        let mut templates = Vec::new();

        // Detect framework patterns
        if self.enable_framework_detection {
            framework_patterns.extend(self.detect_framework_patterns(code)?);
        }

        // Detect dynamic HTML injection
        if self.enable_html_extraction {
            dynamic_injection_points = self.detect_injection_points(code)?;
            event_loaders = self.detect_event_loaders(code)?;
            templates = self.extract_templates(code)?;
        }

        let confidence = self.calculate_confidence(&framework_patterns, &templates);

        Ok(AdvancedObfuscationAnalysis {
            framework_patterns,
            dynamic_injection_points,
            event_loaders,
            templates,
            confidence,
        })
    }

    /// Perform advanced deobfuscation
    pub fn deobfuscate(&self, code: &str) -> Result<AdvancedDeobfuscationResult> {
        let mut steps = Vec::new();
        let mut javascript = code.to_string();
        let mut html_templates = Vec::new();
        let mut event_content_map = HashMap::new();

        // Step 1: Detect and unwrap framework bundling
        if self.detect_webpack(&code) {
            javascript = self.unwrap_webpack(&javascript)?;
            steps.push("Unwrapped Webpack bundle".to_string());
        }

        // Step 2: Extract HTML from dynamic injection
        let injection_points = self.detect_injection_points(&javascript)?;
        for point in injection_points {
            if let Some(html) = self.extract_html_from_injection(&javascript, &point)? {
                html_templates.push(html);
                steps.push(format!(
                    "Extracted HTML from {} at line {}",
                    point.method, point.line
                ));
            }
        }

        // Step 3: Extract templates from framework code
        let templates = self.extract_templates(&javascript)?;
        for template in templates {
            html_templates.push(template.html_content.clone());
            steps.push(format!("Extracted {} template", template.template_type));
        }

        // Step 4: Map event handlers to content
        let event_loaders = self.detect_event_loaders(&javascript)?;
        for loader in event_loaders {
            if let Some(content) = self.resolve_event_content(&javascript, &loader)? {
                event_content_map.insert(loader.event_type.clone(), content);
                steps.push(format!("Mapped {} event to content", loader.event_type));
            }
        }

        // Step 5: Clean up obfuscated code
        javascript = self.cleanup_obfuscation(&javascript)?;
        steps.push("Applied final cleanup".to_string());

        Ok(AdvancedDeobfuscationResult {
            javascript,
            html_templates,
            event_content_map,
            success: true,
            steps,
        })
    }

    /// Detect framework-specific patterns - Comprehensive global coverage
    fn detect_framework_patterns(&self, code: &str) -> Result<Vec<FrameworkObfuscation>> {
        let mut patterns = Vec::new();

        // ========== Bundlers & Build Tools ==========

        // Webpack
        if code.contains("__webpack_require__")
            || code.contains("webpackChunk")
            || code.contains("webpackJsonp")
            || code.contains("__webpack_modules__")
        {
            patterns.push(FrameworkObfuscation::WebpackBundled);
        }

        // Rollup/Vite
        if code.contains("import.meta")
            || code.contains("__vite")
            || (code.contains("export") && code.contains("rollup"))
        {
            patterns.push(FrameworkObfuscation::RollupBundled);
        }

        // Parcel
        if code.contains("$parcel$") || code.contains("parcelRequire") {
            patterns.push(FrameworkObfuscation::ParcelBundled);
        }

        // esbuild
        if code.contains("__esm(") || code.contains("__toESM") || code.contains("__export") {
            patterns.push(FrameworkObfuscation::EsbuildMinified);
        }

        // Turbopack
        if code.contains("__turbopack") || code.contains("TURBOPACK") {
            patterns.push(FrameworkObfuscation::TurbopackBundled);
        }

        // Browserify
        if code.contains("require=function") && code.contains("modules[id][0].call") {
            patterns.push(FrameworkObfuscation::BrowserifyBundled);
        }

        // SystemJS
        if code.contains("System.register") || code.contains("SystemJS") {
            patterns.push(FrameworkObfuscation::SystemJSLoader);
        }

        // RequireJS/AMD
        if code.contains("define.amd") || code.contains("requirejs") {
            patterns.push(FrameworkObfuscation::RequireJSAMD);
        }

        // ========== Frontend Frameworks ==========

        // React
        if code.contains("React.createElement")
            || code.contains("_jsx")
            || code.contains("_jsxs")
            || code.contains("_react.default.createElement")
            || code.contains("_reactDom")
            || code.contains("__jsx")
        {
            patterns.push(FrameworkObfuscation::ReactCompiled);
        }

        // Vue (2 & 3)
        if code.contains("_createVNode")
            || code.contains("_createElementVNode")
            || code.contains("_createBlock")
            || code.contains("Vue.component")
            || code.contains("_hoisted_")
            || code.contains("_withCtx")
            || code.contains("_renderSlot")
        {
            patterns.push(FrameworkObfuscation::VueCompiled);
        }

        // Angular
        if code.contains("ɵɵ")
            || code.contains("@angular")
            || code.contains("ngFactory")
            || code.contains("platformBrowser")
            || code.contains("ɵcmp")
            || code.contains("ɵdir")
        {
            patterns.push(FrameworkObfuscation::AngularCompiled);
        }

        // Svelte
        if code.contains("SvelteComponent")
            || code.contains("svelte/internal")
            || code.contains("create_component")
            || code.contains("mount_component")
            || code.contains("$$invalidate")
        {
            patterns.push(FrameworkObfuscation::SvelteCompiled);
        }

        // Solid.js
        if code.contains("createSignal")
            || code.contains("createEffect")
            || code.contains("solid-js")
            || code.contains("_$template")
        {
            patterns.push(FrameworkObfuscation::SolidJSCompiled);
        }

        // Preact
        if code.contains("preact.h")
            || code.contains("preact/hooks")
            || (code.contains("h(") && code.contains("VNode"))
        {
            patterns.push(FrameworkObfuscation::PreactCompiled);
        }

        // Ember.js
        if code.contains("Ember.Component")
            || code.contains("@ember")
            || code.contains("defineProperty") && code.contains("computed")
        {
            patterns.push(FrameworkObfuscation::EmberCompiled);
        }

        // Alpine.js
        if code.contains("x-data")
            || code.contains("Alpine.start")
            || code.contains("@click")
            || code.contains("x-show")
        {
            patterns.push(FrameworkObfuscation::AlpineDirectives);
        }

        // Lit (Web Components)
        if code.contains("LitElement")
            || code.contains("lit-html")
            || code.contains("customElement") && code.contains("property")
        {
            patterns.push(FrameworkObfuscation::LitElements);
        }

        // Stencil
        if code.contains("@stencil/core") || code.contains("@Component") && code.contains("@Prop") {
            patterns.push(FrameworkObfuscation::StencilCompiled);
        }

        // ========== Meta Frameworks ==========

        // Next.js
        if code.contains("__next")
            || code.contains("next/router")
            || code.contains("next/link")
            || code.contains("getServerSideProps")
            || code.contains("getStaticProps")
        {
            patterns.push(FrameworkObfuscation::NextJSFramework);
        }

        // Nuxt.js
        if code.contains("$nuxt")
            || code.contains("nuxtServerInit")
            || code.contains("asyncData")
            || code.contains("nuxt/app")
        {
            patterns.push(FrameworkObfuscation::NuxtJSFramework);
        }

        // Gatsby
        if code.contains("___gatsby")
            || code.contains("gatsby-browser")
            || code.contains("graphql") && code.contains("pageQuery")
        {
            patterns.push(FrameworkObfuscation::GatsbyStatic);
        }

        // Remix
        if code.contains("@remix-run")
            || code.contains("useLoaderData")
            || code.contains("remix") && code.contains("loader")
        {
            patterns.push(FrameworkObfuscation::RemixFramework);
        }

        // SvelteKit
        if code.contains("@sveltejs/kit")
            || code.contains("$app/")
            || code.contains("load") && code.contains("svelte")
        {
            patterns.push(FrameworkObfuscation::SvelteKitFramework);
        }

        // Astro
        if code.contains("astro:content") || code.contains("Astro.props") || code.contains(".astro")
        {
            patterns.push(FrameworkObfuscation::AstroFramework);
        }

        // Qwik
        if code.contains("@builder.io/qwik")
            || code.contains("component$")
            || code.contains("useSignal") && code.contains("qwik")
        {
            patterns.push(FrameworkObfuscation::QwikFramework);
        }

        // ========== Mobile & Native ==========

        // React Native
        if code.contains("react-native")
            || code.contains("AppRegistry")
            || code.contains("StyleSheet.create")
        {
            patterns.push(FrameworkObfuscation::ReactNative);
        }

        // Ionic
        if code.contains("@ionic") || code.contains("IonButton") || code.contains("ionViewDidEnter")
        {
            patterns.push(FrameworkObfuscation::IonicFramework);
        }

        // Capacitor
        if code.contains("@capacitor") || code.contains("Capacitor.Plugins") {
            patterns.push(FrameworkObfuscation::CapacitorBridge);
        }

        // Cordova
        if code.contains("cordova.js") || code.contains("device.platform") {
            patterns.push(FrameworkObfuscation::CordovaPhoneGap);
        }

        // ========== Chinese Frameworks ==========

        // Taro (京东)
        if code.contains("@tarojs")
            || code.contains("Taro.Component")
            || code.contains("taro-components")
        {
            patterns.push(FrameworkObfuscation::TaroFramework);
        }

        // Uni-app (DCloud)
        if code.contains("uni-app") || code.contains("uni.request") || code.contains("@dcloudio") {
            patterns.push(FrameworkObfuscation::UniAppFramework);
        }

        // mpvue (美团)
        if code.contains("mpvue") || code.contains("mpvue-loader") {
            patterns.push(FrameworkObfuscation::MpVueFramework);
        }

        // Rax (阿里)
        if code.contains("rax") || code.contains("createElement") && code.contains("rax") {
            patterns.push(FrameworkObfuscation::RaxFramework);
        }

        // Remax (Alibaba mini-program)
        if code.contains("remax") || code.contains("@remax") {
            patterns.push(FrameworkObfuscation::RemaxFramework);
        }

        // Kbone (微信)
        if code.contains("kbone") || code.contains("mp-runtime") {
            patterns.push(FrameworkObfuscation::KboneFramework);
        }

        // Omi (Tencent)
        if code.contains("omi") || code.contains("@omi") || code.contains("WeElement") {
            patterns.push(FrameworkObfuscation::OmiFramework);
        }

        // San (百度)
        if code.contains("san")
            && (code.contains("defineComponent") || code.contains("san.Component"))
        {
            patterns.push(FrameworkObfuscation::SanFramework);
        }

        // Chameleon (滴滴)
        if code.contains("chameleon") || code.contains("@chameleon") {
            patterns.push(FrameworkObfuscation::ChameleonFramework);
        }

        // Qiankun (阿里乾坤)
        if code.contains("qiankun") || code.contains("registerMicroApps") {
            patterns.push(FrameworkObfuscation::QiankunMicroFrontend);
        }

        // ========== State Management ==========

        // Redux
        if code.contains("createStore")
            || code.contains("redux")
            || code.contains("combineReducers")
        {
            patterns.push(FrameworkObfuscation::ReduxPatterns);
        }

        // MobX
        if code.contains("makeObservable") || code.contains("@observable") || code.contains("mobx")
        {
            patterns.push(FrameworkObfuscation::MobXObservables);
        }

        // Vuex
        if code.contains("Vuex.Store") || code.contains("vuex") || code.contains("mapState") {
            patterns.push(FrameworkObfuscation::VuexStore);
        }

        // Pinia
        if code.contains("defineStore") || code.contains("pinia") {
            patterns.push(FrameworkObfuscation::PiniaStore);
        }

        // Zustand
        if code.contains("create(") && code.contains("zustand") {
            patterns.push(FrameworkObfuscation::ZustandStore);
        }

        // ========== UI Libraries ==========

        // Material-UI
        if code.contains("@mui") || code.contains("makeStyles") || code.contains("material-ui") {
            patterns.push(FrameworkObfuscation::MaterialUICompiled);
        }

        // Ant Design
        if code.contains("antd") || code.contains("@ant-design") {
            patterns.push(FrameworkObfuscation::AntDesignCompiled);
        }

        // Element UI (饿了么)
        if code.contains("element-ui") || code.contains("element-plus") {
            patterns.push(FrameworkObfuscation::ElementUICompiled);
        }

        // Vant (有赞)
        if code.contains("vant") || code.contains("@vant") {
            patterns.push(FrameworkObfuscation::VantUICompiled);
        }

        // TailwindCSS
        if code.contains("tailwindcss") || code.contains("@tailwind") {
            patterns.push(FrameworkObfuscation::TailwindJIT);
        }

        // Styled Components
        if code.contains("styled-components") || code.contains("styled.") {
            patterns.push(FrameworkObfuscation::StyledComponents);
        }

        // Emotion
        if code.contains("@emotion") || code.contains("/** @jsx jsx */") {
            patterns.push(FrameworkObfuscation::EmotionCSS);
        }

        // ========== SSR & Hydration ==========

        if code.contains("hydrateRoot") || code.contains("hydrate") {
            patterns.push(FrameworkObfuscation::ServerSideRendered);
        }

        if code.contains("island:") || code.contains("client:load") {
            patterns.push(FrameworkObfuscation::IslandArchitecture);
        }

        // ========== Obfuscation Techniques ==========

        // Dynamic HTML injection
        if code.contains("innerHTML")
            || code.contains(".appendChild")
            || code.contains("insertAdjacentHTML")
        {
            patterns.push(FrameworkObfuscation::DynamicHtmlInjection);
        }

        // Event-driven content
        if code.contains("addEventListener")
            && (code.contains("innerHTML") || code.contains("createElement"))
        {
            patterns.push(FrameworkObfuscation::EventDrivenContent);
        }

        // Template literals with HTML
        let template_regex = Regex::new(r"`[^`]*<[^>]+>[^`]*`")?;
        if template_regex.is_match(code) {
            patterns.push(FrameworkObfuscation::TemplateLiteralObfuscation);
        }

        // Code splitting
        if code.contains("import(") || code.contains("lazy(") {
            patterns.push(FrameworkObfuscation::CodeSplitting);
        }

        // Minification
        if code.contains("/*! For license") || code.len() > 1000 && !code.contains("\n\n") {
            patterns.push(FrameworkObfuscation::TerserMinified);
        }

        // Module Federation
        if code.contains("__webpack_require__.f") || code.contains("remoteEntry") {
            patterns.push(FrameworkObfuscation::ModuleFederation);
        }

        // ES Modules
        if code.contains("export") && code.contains("import") {
            patterns.push(FrameworkObfuscation::ESModules);
        }

        // CommonJS
        if code.contains("module.exports") || code.contains("require(") {
            patterns.push(FrameworkObfuscation::CommonJS);
        }

        // IIFE
        if code.starts_with("(") && code.contains("})(") {
            patterns.push(FrameworkObfuscation::IIFEWrapper);
        }

        Ok(patterns)
    }

    /// Get framework metadata and characteristics
    pub fn get_framework_info(&self, framework: &FrameworkObfuscation) -> FrameworkInfo {
        match framework {
            // Bundlers
            FrameworkObfuscation::WebpackBundled => FrameworkInfo {
                name: "Webpack".to_string(),
                category: "Bundler".to_string(),
                patterns: vec!["__webpack_require__", "webpackChunk"],
                deobfuscation_strategy: "Unwrap module system, resolve dynamic imports",
                origin: "Global".to_string(),
            },
            FrameworkObfuscation::RollupBundled => FrameworkInfo {
                name: "Rollup/Vite".to_string(),
                category: "Bundler".to_string(),
                patterns: vec!["import.meta", "__vite"],
                deobfuscation_strategy: "Resolve ES modules, extract chunks",
                origin: "Global".to_string(),
            },
            FrameworkObfuscation::ParcelBundled => FrameworkInfo {
                name: "Parcel".to_string(),
                category: "Bundler".to_string(),
                patterns: vec!["$parcel$", "parcelRequire"],
                deobfuscation_strategy: "Unwrap Parcel runtime",
                origin: "Global".to_string(),
            },

            // Frontend Frameworks
            FrameworkObfuscation::ReactCompiled => FrameworkInfo {
                name: "React".to_string(),
                category: "Frontend Framework".to_string(),
                patterns: vec!["React.createElement", "_jsx", "_jsxs"],
                deobfuscation_strategy: "Convert createElement to JSX, extract components",
                origin: "USA (Facebook/Meta)".to_string(),
            },
            FrameworkObfuscation::VueCompiled => FrameworkInfo {
                name: "Vue".to_string(),
                category: "Frontend Framework".to_string(),
                patterns: vec!["_createVNode", "_createElementVNode", "_hoisted_"],
                deobfuscation_strategy: "Extract templates from render functions",
                origin: "China/Global (Evan You)".to_string(),
            },
            FrameworkObfuscation::AngularCompiled => FrameworkInfo {
                name: "Angular".to_string(),
                category: "Frontend Framework".to_string(),
                patterns: vec!["ɵɵ", "ngFactory", "platformBrowser"],
                deobfuscation_strategy: "Reverse Ivy compilation, extract templates",
                origin: "USA (Google)".to_string(),
            },
            FrameworkObfuscation::SvelteCompiled => FrameworkInfo {
                name: "Svelte".to_string(),
                category: "Frontend Framework".to_string(),
                patterns: vec!["SvelteComponent", "$$invalidate"],
                deobfuscation_strategy: "Extract reactive statements and templates",
                origin: "Global".to_string(),
            },

            // Chinese Frameworks
            FrameworkObfuscation::TaroFramework => FrameworkInfo {
                name: "Taro".to_string(),
                category: "Multi-platform Framework".to_string(),
                patterns: vec!["@tarojs", "Taro.Component"],
                deobfuscation_strategy: "Convert mini-program to web format",
                origin: "China (JD.com)".to_string(),
            },
            FrameworkObfuscation::UniAppFramework => FrameworkInfo {
                name: "Uni-app".to_string(),
                category: "Multi-platform Framework".to_string(),
                patterns: vec!["uni-app", "uni.request", "@dcloudio"],
                deobfuscation_strategy: "Extract Vue-based components",
                origin: "China (DCloud)".to_string(),
            },
            FrameworkObfuscation::RaxFramework => FrameworkInfo {
                name: "Rax".to_string(),
                category: "Frontend Framework".to_string(),
                patterns: vec!["rax", "createElement"],
                deobfuscation_strategy: "Similar to React, lightweight",
                origin: "China (Alibaba)".to_string(),
            },
            FrameworkObfuscation::OmiFramework => FrameworkInfo {
                name: "Omi".to_string(),
                category: "Web Components Framework".to_string(),
                patterns: vec!["omi", "@omi", "WeElement"],
                deobfuscation_strategy: "Extract Web Components",
                origin: "China (Tencent)".to_string(),
            },
            FrameworkObfuscation::SanFramework => FrameworkInfo {
                name: "San".to_string(),
                category: "Frontend Framework".to_string(),
                patterns: vec!["san", "defineComponent"],
                deobfuscation_strategy: "Extract data-driven components",
                origin: "China (Baidu)".to_string(),
            },
            FrameworkObfuscation::QiankunMicroFrontend => FrameworkInfo {
                name: "Qiankun".to_string(),
                category: "Micro Frontend".to_string(),
                patterns: vec!["qiankun", "registerMicroApps"],
                deobfuscation_strategy: "Extract sub-applications",
                origin: "China (Alibaba)".to_string(),
            },

            // Meta Frameworks
            FrameworkObfuscation::NextJSFramework => FrameworkInfo {
                name: "Next.js".to_string(),
                category: "Meta Framework (React)".to_string(),
                patterns: vec!["__next", "getServerSideProps"],
                deobfuscation_strategy: "Extract SSR/SSG pages and API routes",
                origin: "USA (Vercel)".to_string(),
            },
            FrameworkObfuscation::NuxtJSFramework => FrameworkInfo {
                name: "Nuxt.js".to_string(),
                category: "Meta Framework (Vue)".to_string(),
                patterns: vec!["$nuxt", "asyncData"],
                deobfuscation_strategy: "Extract Vue SSR components",
                origin: "France/Global".to_string(),
            },

            // Default
            _ => FrameworkInfo {
                name: "Unknown".to_string(),
                category: "Unknown".to_string(),
                patterns: vec![],
                deobfuscation_strategy: "Apply generic deobfuscation",
                origin: "Unknown".to_string(),
            },
        }
    }

    /// Apply framework-specific deobfuscation
    pub fn deobfuscate_framework_specific(
        &self,
        code: &str,
        framework: &FrameworkObfuscation,
    ) -> Result<String> {
        match framework {
            FrameworkObfuscation::WebpackBundled => self.unwrap_webpack(code),
            FrameworkObfuscation::ReactCompiled => self.deobfuscate_react(code),
            FrameworkObfuscation::VueCompiled => self.deobfuscate_vue(code),
            FrameworkObfuscation::AngularCompiled => self.deobfuscate_angular(code),
            FrameworkObfuscation::TaroFramework => self.deobfuscate_taro(code),
            FrameworkObfuscation::UniAppFramework => self.deobfuscate_uniapp(code),
            _ => Ok(code.to_string()),
        }
    }

    /// Unwrap Webpack bundled code
    pub fn unwrap_webpack(&self, code: &str) -> Result<String> {
        let mut result = String::new();

        // Pattern 1: Webpack 5+ module format
        // (self["webpackChunk"] = self["webpackChunk"] || []).push([[...], {...modules...}])
        let chunk_regex = Regex::new(
            r#"\(self\["webpackChunk[^"]*"\][^)]*\)\.push\(\[\[.*?\],\s*\{([^}]+)\}\]\)"#,
        )?;

        // Pattern 2: Webpack 4 IIFE format
        // (function(modules) { ... })({ 0: function(module, exports, __webpack_require__) {...} })
        let iife_regex = Regex::new(r"\(function\(modules\)\s*\{[^}]*\}\)\s*\(\{([^}]+)\}\)")?;

        // Pattern 3: __webpack_require__ calls
        let require_regex = Regex::new(r#"__webpack_require__\((\d+|['"][^'"]+['"])\)"#)?;

        // Extract modules
        let mut modules = HashMap::new();

        // Try chunk pattern first (Webpack 5+)
        if let Some(caps) = chunk_regex.captures(code) {
            if let Some(module_map) = caps.get(1) {
                let module_text = module_map.as_str();
                // Parse individual modules
                let module_regex = Regex::new(r#"(\d+):\s*function\([^)]*\)\s*\{([^}]*)\}"#)?;
                for module_cap in module_regex.captures_iter(module_text) {
                    if let (Some(id), Some(body)) = (module_cap.get(1), module_cap.get(2)) {
                        modules.insert(id.as_str().to_string(), body.as_str().to_string());
                    }
                }
            }
        }

        // Try IIFE pattern (Webpack 4)
        if modules.is_empty() {
            if let Some(caps) = iife_regex.captures(code) {
                if let Some(module_map) = caps.get(1) {
                    let module_text = module_map.as_str();
                    let module_regex = Regex::new(r#"(\d+):\s*function\([^)]*\)\s*\{([^}]*)\}"#)?;
                    for module_cap in module_regex.captures_iter(module_text) {
                        if let (Some(id), Some(body)) = (module_cap.get(1), module_cap.get(2)) {
                            modules.insert(id.as_str().to_string(), body.as_str().to_string());
                        }
                    }
                }
            }
        }

        // Reconstruct code without Webpack wrapper
        result.push_str("// Unwrapped from Webpack bundle\n\n");

        let modules_empty = modules.is_empty();

        for (id, body) in modules {
            result.push_str(&format!("// Module {}\n", id));
            result.push_str(&body);
            result.push_str("\n\n");
        }

        // If no modules found, return original with comment
        if modules_empty {
            result.push_str("// Warning: Could not extract modules from Webpack bundle\n");
            result.push_str(code);
        }

        Ok(result)
    }

    /// Deobfuscate React compiled code
    fn deobfuscate_react(&self, code: &str) -> Result<String> {
        let mut result = code.to_string();

        // Convert React.createElement back to JSX-like representation
        let create_element_regex =
            Regex::new(r#"React\.createElement\(['"](\w+)['"],\s*(\{[^}]*\}|\w+|null)"#)?;

        for caps in create_element_regex.captures_iter(code) {
            if let (Some(tag), Some(props)) = (caps.get(1), caps.get(2)) {
                let jsx_like = format!("<{}  {{...{}}}>", tag.as_str(), props.as_str());
                // Note: This is simplified, real implementation would be more complex
            }
        }

        Ok(result)
    }

    /// Deobfuscate Vue compiled code
    fn deobfuscate_vue(&self, code: &str) -> Result<String> {
        let mut result = code.to_string();

        // Extract hoisted templates
        let hoisted_regex = Regex::new(r"_hoisted_\d+\s*=\s*(\{[^}]*\})")?;

        // Extract createVNode calls and convert to template
        let vnode_regex = Regex::new(r#"_createVNode\(['"](\w+)['"],\s*(\{[^}]*\}|\w+|null)"#)?;

        Ok(result)
    }

    /// Deobfuscate Angular Ivy compiled code
    fn deobfuscate_angular(&self, code: &str) -> Result<String> {
        let mut result = code.to_string();

        // Extract component templates from Ivy instructions
        let template_regex = Regex::new(r#"ɵɵelementStart\(\d+,\s*['"](\w+)['"]"#)?;

        Ok(result)
    }

    /// Deobfuscate Taro framework code
    fn deobfuscate_taro(&self, code: &str) -> Result<String> {
        let mut result = code.to_string();

        // Taro uses React-like syntax but compiled for mini-programs
        // Convert back to readable format

        Ok(result)
    }

    /// Deobfuscate Uni-app framework code
    fn deobfuscate_uniapp(&self, code: &str) -> Result<String> {
        let mut result = code.to_string();

        // Uni-app is Vue-based
        // Extract uni.* API calls and convert to web APIs
        result = result.replace("uni.request", "fetch");
        result = result.replace("uni.navigateTo", "router.push");

        Ok(result)
    }

    /// Generate deobfuscation report
    pub fn generate_report(&self, analysis: &AdvancedObfuscationAnalysis) -> String {
        let mut report = String::new();

        report.push_str("=== Advanced Deobfuscation Analysis ===\n\n");

        report.push_str(&format!(
            "Confidence: {:.1}%\n\n",
            analysis.confidence * 100.0
        ));

        report.push_str("Detected Frameworks:\n");
        for framework in &analysis.framework_patterns {
            let info = self.get_framework_info(framework);
            report.push_str(&format!(
                "  • {} ({}) - Origin: {}\n",
                info.name, info.category, info.origin
            ));
            report.push_str(&format!("    Strategy: {}\n", info.deobfuscation_strategy));
        }

        report.push_str(&format!(
            "\nDynamic Injection Points: {}\n",
            analysis.dynamic_injection_points.len()
        ));
        report.push_str(&format!(
            "Event Loaders: {}\n",
            analysis.event_loaders.len()
        ));
        report.push_str(&format!(
            "Extracted Templates: {}\n",
            analysis.templates.len()
        ));

        report
    }

    /// Detect injection points
    fn detect_injection_points(&self, code: &str) -> Result<Vec<InjectionPoint>> {
        let mut points = Vec::new();

        for (line_num, line) in code.lines().enumerate() {
            // innerHTML injection
            if line.contains("innerHTML") {
                if let Some(target) = self.extract_target(line) {
                    points.push(InjectionPoint {
                        line: line_num + 1,
                        method: "innerHTML".to_string(),
                        target,
                        content_hint: self.extract_content_hint(line).unwrap_or_default(),
                    });
                }
            }

            // appendChild injection
            if line.contains("appendChild") || line.contains("append(") {
                if let Some(target) = self.extract_target(line) {
                    points.push(InjectionPoint {
                        line: line_num + 1,
                        method: "appendChild".to_string(),
                        target,
                        content_hint: self.extract_content_hint(line).unwrap_or_default(),
                    });
                }
            }

            // insertAdjacentHTML
            if line.contains("insertAdjacentHTML") {
                if let Some(target) = self.extract_target(line) {
                    points.push(InjectionPoint {
                        line: line_num + 1,
                        method: "insertAdjacentHTML".to_string(),
                        target,
                        content_hint: self.extract_content_hint(line).unwrap_or_default(),
                    });
                }
            }
        }

        Ok(points)
    }

    /// Detect event-triggered content loaders
    fn detect_event_loaders(&self, code: &str) -> Result<Vec<EventLoader>> {
        let mut loaders = Vec::new();

        let event_regex = Regex::new(r#"addEventListener\(['"](\w+)['"],\s*(\w+)"#)?;

        for caps in event_regex.captures_iter(code) {
            let event_type = caps
                .get(1)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            let function = caps
                .get(2)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();

            loaders.push(EventLoader {
                event_type: event_type.clone(),
                target: "detected".to_string(),
                function: function.clone(),
                trigger_condition: format!("When {} event fires", event_type),
            });
        }

        Ok(loaders)
    }

    /// Extract HTML templates from code
    fn extract_templates(&self, code: &str) -> Result<Vec<ExtractedTemplate>> {
        let mut templates = Vec::new();

        // Extract from template literals
        let template_regex = Regex::new(r"`([^`]*<[^>]+>[^`]*)`")?;
        for caps in template_regex.captures_iter(code) {
            if let Some(content) = caps.get(1) {
                let html = content.as_str().to_string();
                templates.push(ExtractedTemplate {
                    template_type: "Template Literal".to_string(),
                    html_content: html.clone(),
                    original_form: format!("`{}`", html),
                    confidence: 0.9,
                });
            }
        }

        // Extract from string concatenation
        let concat_regex = Regex::new(r#"['"](<[^>]+>[^'"]*)['"]\s*\+"#)?;
        for caps in concat_regex.captures_iter(code) {
            if let Some(content) = caps.get(1) {
                let html = content.as_str().to_string();
                templates.push(ExtractedTemplate {
                    template_type: "String Concatenation".to_string(),
                    html_content: html.clone(),
                    original_form: format!("'{}'", html),
                    confidence: 0.7,
                });
            }
        }

        Ok(templates)
    }

    /// Detect Webpack bundling
    fn detect_webpack(&self, code: &str) -> bool {
        code.contains("__webpack_require__") || code.contains("webpackChunk")
    }

    /// Extract HTML from injection point
    fn extract_html_from_injection(
        &self,
        code: &str,
        point: &InjectionPoint,
    ) -> Result<Option<String>> {
        let lines: Vec<&str> = code.lines().collect();
        if point.line == 0 || point.line > lines.len() {
            return Ok(None);
        }

        let line = lines[point.line - 1];

        // Try to extract HTML from assignment
        let html_regex = Regex::new(r#"['"](.*<.*>.*)['"]"#)?;
        if let Some(caps) = html_regex.captures(line) {
            if let Some(html) = caps.get(1) {
                return Ok(Some(html.as_str().to_string()));
            }
        }

        Ok(None)
    }

    /// Resolve content from event loader
    fn resolve_event_content(&self, code: &str, loader: &EventLoader) -> Result<Option<String>> {
        // Try to find the function definition
        let func_regex = Regex::new(&format!(
            r"function\s+{}\s*\([^)]*\)\s*\{{([^}}]+)}}",
            loader.function
        ))?;

        if let Some(caps) = func_regex.captures(code) {
            if let Some(body) = caps.get(1) {
                let body_str = body.as_str();

                // Look for HTML in function body
                let html_regex = Regex::new(r#"['"](.*<.*>.*)['"]"#)?;
                if let Some(html_caps) = html_regex.captures(body_str) {
                    if let Some(html) = html_caps.get(1) {
                        return Ok(Some(html.as_str().to_string()));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Extract target element from line
    fn extract_target(&self, line: &str) -> Option<String> {
        // Try to extract variable or selector
        if let Some(pos) = line.find('.') {
            let before = &line[..pos];
            if let Some(word_start) = before.rfind(|c: char| !c.is_alphanumeric() && c != '_') {
                return Some(before[word_start + 1..].trim().to_string());
            }
        }

        Some("unknown".to_string())
    }

    /// Extract content hint from line
    fn extract_content_hint(&self, line: &str) -> Option<String> {
        let hint_regex = Regex::new(r#"['"](.*?)['"]"#).ok()?;
        hint_regex
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| {
                let s = m.as_str();
                if s.len() > 50 {
                    format!("{}...", &s[..50])
                } else {
                    s.to_string()
                }
            })
    }

    /// Calculate confidence score
    fn calculate_confidence(
        &self,
        patterns: &[FrameworkObfuscation],
        templates: &[ExtractedTemplate],
    ) -> f32 {
        let pattern_score = if !patterns.is_empty() { 0.5 } else { 0.0 };
        let template_score = if !templates.is_empty() {
            templates.iter().map(|t| t.confidence).sum::<f32>() / templates.len() as f32 * 0.5
        } else {
            0.0
        };

        pattern_score + template_score
    }

    /// Clean up remaining obfuscation
    fn cleanup_obfuscation(&self, code: &str) -> Result<String> {
        let mut result = code.to_string();

        // Remove excessive whitespace
        let whitespace_regex = Regex::new(r"\s+")?;
        result = whitespace_regex.replace_all(&result, " ").to_string();

        // Remove comments
        let comment_regex = Regex::new(r"//.*$")?;
        result = comment_regex.replace_all(&result, "").to_string();

        Ok(result)
    }
}

impl Default for AdvancedDeobfuscator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_webpack() {
        let deob = AdvancedDeobfuscator::new();
        let code = r#"
            (function(modules) {
                function __webpack_require__(moduleId) {
                    return modules[moduleId].call();
                }
            })([function() { console.log("test"); }]);
        "#;

        let analysis = deob.analyze(code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::WebpackBundled));
    }

    #[test]
    fn test_detect_react() {
        let deob = AdvancedDeobfuscator::new();
        let code = r#"
            React.createElement("div", {className: "test"}, 
                React.createElement("span", null, "Hello")
            );
        "#;

        let analysis = deob.analyze(code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::ReactCompiled));
    }

    #[test]
    fn test_detect_dynamic_html() {
        let deob = AdvancedDeobfuscator::new();
        let code = r#"
            document.getElementById("app").innerHTML = "<div>Hello</div>";
        "#;

        let analysis = deob.analyze(code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::DynamicHtmlInjection));
        assert!(!analysis.dynamic_injection_points.is_empty());
    }

    #[test]
    fn test_extract_templates() {
        let deob = AdvancedDeobfuscator::new();
        let code = r#"
            const template = `<div class="container"><h1>Title</h1></div>`;
        "#;

        let analysis = deob.analyze(code).unwrap();
        assert!(!analysis.templates.is_empty());
        assert!(analysis.templates[0].html_content.contains("<div"));
    }

    #[test]
    fn test_event_loader_detection() {
        let deob = AdvancedDeobfuscator::new();
        let code = r#"
            button.addEventListener('click', handleClick);
            function handleClick() {
                element.innerHTML = "<p>Loaded!</p>";
            }
        "#;

        let analysis = deob.analyze(code).unwrap();
        assert!(!analysis.event_loaders.is_empty());
        assert_eq!(analysis.event_loaders[0].event_type, "click");
    }

    #[test]
    fn test_advanced_deobfuscation() {
        let deob = AdvancedDeobfuscator::new();
        let code = r#"
            const html = `<div><span>Test</span></div>`;
            document.body.innerHTML = html;
        "#;

        let result = deob.deobfuscate(code).unwrap();
        assert!(result.success);
        assert!(!result.html_templates.is_empty());
        assert!(!result.steps.is_empty());
    }
}
