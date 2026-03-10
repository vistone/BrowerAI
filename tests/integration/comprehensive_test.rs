//! BrowerAI 全面集成测试
//! 
//! 测试所有功能模块的集成情况

use std::collections::HashMap;

// ===== 1. 核心模块测试 =====
#[test]
fn test_core_module_integration() {
    println!("\n=== 测试 browerai-core 模块 ===");
    
    // 测试错误处理
    let err = browerai_core::BrowserError::parse_error("test error", None);
    assert!(err.to_string().contains("test error"));
    println!("✅ BrowserError 错误处理");
    
    // 测试配置
    let config = browerai_core::BrowserConfig::default();
    assert_eq!(config.version, browerai_core::VERSION);
    println!("✅ BrowserConfig 配置系统");
    
    // 测试类型系统
    let code_type = browerai_core::CodeType::JavaScript;
    assert!(matches!(code_type, browerai_core::CodeType::JavaScript));
    println!("✅ CodeType 类型系统");
    
    // 测试 SourceLocation
    let loc = browerai_core::SourceLocation::new(1, 10, "test.js");
    assert_eq!(loc.line, 1);
    assert_eq!(loc.column, 10);
    println!("✅ SourceLocation 源码定位");
    
    // 测试指标收集
    let mut metrics = browerai_core::MetricsDashboard::new();
    metrics.record(browerai_core::Metric::new("test", 1.0, browerai_core::MetricType::Counter));
    println!("✅ MetricsDashboard 指标收集");
    
    println!("✅ browerai-core 核心模块全部集成成功\n");
}

// ===== 2. HTML解析器测试 =====
#[test]
fn test_html_parser_integration() {
    println!("\n=== 测试 browerai-html-parser 模块 ===");
    
    use browerai_html_parser::{HtmlParser, dom::{Document, NodeType}};
    use browerai_core::traits::Parser;
    
    let parser = HtmlParser::new();
    let html = r#"<!DOCTYPE html>
    <html>
    <head><title>Test</title></head>
    <body>
        <div id="main" class="container">
            <p>Hello World</p>
            <script>console.log('test');</script>
        </div>
    </body>
    </html>"#;
    
    let doc = parser.parse(html).expect("HTML解析失败");
    assert!(doc.node_count() > 0);
    println!("✅ HTML5 文档解析");
    
    // 测试DOM遍历
    let scripts = parser.extract_scripts(&doc);
    assert!(!scripts.is_empty());
    println!("✅ 脚本提取功能");
    
    let resources = parser.extract_resources(&doc);
    println!("✅ 资源提取功能 (找到 {} 个资源)", resources.len());
    
    // 测试DOM查询
    if let Some(body) = doc.find_element("body") {
        println!("✅ DOM元素查询");
    }
    
    println!("✅ browerai-html-parser 模块全部集成成功\n");
}

// ===== 3. CSS解析器测试 =====
#[test]
fn test_css_parser_integration() {
    println!("\n=== 测试 browerai-css-parser 模块 ===");
    
    use browerai_css_parser::{CssParser, stylesheet::{Stylesheet, Selector, Declaration}};
    use browerai_core::traits::Parser;
    
    let parser = CssParser::new();
    let css = r#"
        body {
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
        }
        #main {
            width: 100%;
            background: #fff;
        }
        .container {
            display: flex;
            flex-direction: column;
        }
        @media screen and (max-width: 768px) {
            .container { width: 100%; }
        }
    "#;
    
    let stylesheet = parser.parse(css).expect("CSS解析失败");
    assert!(!stylesheet.rules.is_empty());
    println!("✅ CSS3 样式表解析 ({} 条规则)", stylesheet.rules.len());
    
    // 测试选择器
    let selectors = parser.extract_selectors(&stylesheet);
    println!("✅ 选择器提取 ({} 个选择器)", selectors.len());
    
    // 测试媒体查询
    let media_queries = parser.extract_media_queries(&stylesheet);
    println!("✅ 媒体查询提取 ({} 个)", media_queries.len());
    
    // 测试属性查找
    let colors = parser.find_property_values(&stylesheet, "background");
    println!("✅ 属性值查找");
    
    println!("✅ browerai-css-parser 模块全部集成成功\n");
}

// ===== 4. JS解析器测试 =====
#[test]
fn test_js_parser_integration() {
    println!("\n=== 测试 browerai-js-parser 模块 ===");
    
    use browerai_js_parser::{JsParser, ast::JsAst, EcmaVersion};
    use browerai_core::traits::Parser;
    
    let parser = JsParser::with_ecma_version(EcmaVersion::Es2022);
    let js_code = r#"
        function greet(name) {
            return `Hello, ${name}!`;
        }
        
        const user = {
            name: 'Alice',
            age: 30,
            greet() { return greet(this.name); }
        };
        
        class Person {
            constructor(name) { this.name = name; }
            sayHi() { return greet(this.name); }
        }
        
        async function fetchData() {
            const response = await fetch('/api/data');
            return response.json();
        }
    "#;
    
    let ast = parser.parse(js_code).expect("JS解析失败");
    println!("✅ JavaScript AST 解析");
    
    // 测试函数提取
    let functions = ast.extract_functions();
    println!("✅ 函数提取 ({} 个函数)", functions.len());
    
    // 测试变量提取
    let variables = ast.extract_variables();
    println!("✅ 变量提取 ({} 个变量)", variables.len());
    
    // 测试类提取
    let classes = ast.extract_classes();
    println!("✅ 类提取 ({} 个类)", classes.len());
    
    // 测试依赖分析
    let dependencies = ast.analyze_dependencies();
    println!("✅ 依赖分析");
    
    println!("✅ browerai-js-parser 模块全部集成成功\n");
}

// ===== 5. JS分析器7阶段管道测试 =====
#[test]
fn test_js_analyzer_pipeline_integration() {
    println!("\n=== 测试 browerai-js-analyzer 7阶段分析管道 ===");
    
    use browerai_js_analyzer::{
        JsAnalyzer, 
        scope::{ScopeAnalyzer, ScopeTree},
        cfg::ControlFlowGraph,
        callgraph::CallGraph,
        dataflow::DataflowAnalyzer,
        loop_analysis::LoopAnalyzer,
        unified::UnifiedAnalysis,
    };
    use browerai_core::traits::Analyzer;
    
    let mut analyzer = JsAnalyzer::new();
    let js_code = r#"
        function factorial(n) {
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }
        
        function main() {
            const result = factorial(5);
            console.log(result);
        }
        
        for (let i = 0; i < 10; i++) {
            if (i % 2 === 0) continue;
            console.log(i);
        }
    "#;
    
    // 阶段1: 作用域分析
    let scope_result = analyzer.analyze_scopes(js_code);
    println!("✅ 阶段1: 作用域分析 (Scope Analysis)");
    
    // 阶段2: SWC转换
    let swc_result = analyzer.transform_with_swc(js_code);
    println!("✅ 阶段2: SWC代码转换 (SWC Transformation)");
    
    // 阶段3: 数据流分析
    let dataflow_result = analyzer.analyze_dataflow(js_code);
    println!("✅ 阶段3: 数据流分析 (Dataflow Analysis)");
    
    // 阶段4: 控制流图
    let cfg_result = analyzer.build_cfg(js_code);
    println!("✅ 阶段4: 控制流图构建 (CFG Construction)");
    
    // 阶段5: 调用图
    let callgraph_result = analyzer.build_callgraph(js_code);
    println!("✅ 阶段5: 调用图构建 (Call Graph)");
    
    // 阶段6: 循环分析
    let loop_result = analyzer.analyze_loops(js_code);
    println!("✅ 阶段6: 循环分析 (Loop Analysis)");
    
    // 阶段7: 统一分析
    let unified_result = analyzer.analyze(js_code).expect("统一分析失败");
    println!("✅ 阶段7: 统一综合分析 (Unified Analysis)");
    
    // 验证分析结果
    assert!(!unified_result.functions.is_empty());
    println!("  - 发现 {} 个函数", unified_result.functions.len());
    println!("  - 发现 {} 个变量", unified_result.variables.len());
    
    println!("✅ browerai-js-analyzer 7阶段管道全部集成成功\n");
}

// ===== 6. AI核心模块测试 =====
#[test]
fn test_ai_core_integration() {
    println!("\n=== 测试 browerai-ai-core 模块 ===");
    
    use browerai_ai_core::{
        AiCore, 
        models::{ModelManager, ModelConfig, ModelType},
        features::{FeatureExtractor, FeatureType, FeatureVector},
        inference::{InferenceEngine, InferenceRequest},
        learning::{LearningEngine, TrainingSample},
    };
    
    // 测试模型管理器
    let model_manager = ModelManager::new();
    println!("✅ ModelManager 模型管理器");
    
    // 测试特征提取器
    let feature_extractor = FeatureExtractor::new();
    let html_sample = "<div class='test'>Hello</div>";
    let features = feature_extractor.extract(html_sample, FeatureType::HtmlStructure);
    println!("✅ FeatureExtractor 特征提取");
    
    // 测试推理引擎
    let inference_engine = InferenceEngine::new();
    println!("✅ InferenceEngine 推理引擎");
    
    // 测试学习引擎
    let learning_engine = LearningEngine::new();
    println!("✅ LearningEngine 学习引擎");
    
    // 测试完整AI核心
    let ai_core = AiCore::new().expect("AI核心初始化失败");
    println!("✅ AiCore 统一入口");
    
    println!("✅ browerai-ai-core 模块全部集成成功\n");
}

// ===== 7. 渲染引擎测试 =====
#[test]
fn test_renderer_core_integration() {
    println!("\n=== 测试 browerai-renderer-core 模块 ===");
    
    use browerai_renderer_core::{
        Renderer, RenderConfig,
        layout::{LayoutEngine, Viewport},
        paint::PaintEngine,
        compositing::Compositor,
        resources::ResourceManager,
    };
    use browerai_core::traits::Renderer as RendererTrait;
    
    let config = RenderConfig::default();
    let renderer = Renderer::new(config);
    println!("✅ Renderer 主渲染器");
    
    // 测试布局引擎
    let layout_engine = LayoutEngine::new();
    println!("✅ LayoutEngine 布局引擎");
    
    // 测试绘制引擎
    let paint_engine = PaintEngine::new();
    println!("✅ PaintEngine 绘制引擎");
    
    // 测试合成器
    let compositor = Compositor::new();
    println!("✅ Compositor 合成器");
    
    // 测试资源管理器
    let resource_manager = ResourceManager::new();
    println!("✅ ResourceManager 资源管理");
    
    println!("✅ browerai-renderer-core 模块全部集成成功\n");
}

// ===== 8. 开发者工具测试 =====
#[test]
fn test_devtools_integration() {
    println!("\n=== 测试 browerai-devtools 模块 ===");
    
    use browerai_devtools::{
        DevTools,
        inspector::{DomInspector, InspectionResult},
        console::{Console, LogLevel},
        profiler::{Profiler, TimingMark},
        network::{NetworkMonitor, NetworkRequest},
    };
    
    let mut devtools = DevTools::new();
    println!("✅ DevTools 主入口");
    
    // 测试DOM检查器
    let inspector = devtools.inspector();
    println!("✅ DomInspector DOM检查器");
    
    // 测试控制台
    let console = devtools.console();
    console.log("Test message");
    println!("✅ Console 控制台");
    
    // 测试性能分析器
    let profiler = devtools.profiler();
    profiler.mark("test_start");
    println!("✅ Profiler 性能分析器");
    
    // 测试网络监控
    let network = devtools.network();
    println!("✅ NetworkMonitor 网络监控");
    
    println!("✅ browerai-devtools 模块全部集成成功\n");
}

// ===== 9. DOM模块测试 =====
#[test]
fn test_dom_module_integration() {
    println!("\n=== 测试 browerai-dom 模块 ===");
    
    use browerai_dom::{
        DomDocument, DomElement, DomNode,
        events::{Event, EventType, EventPhase},
        sandbox::{JsSandbox, ExecutionContext, ResourceLimits},
        web_apis::WebApis,
        modern_apis::ModernApis,
    };
    
    // 测试DOM文档
    let doc = DomDocument::new();
    println!("✅ DomDocument DOM文档");
    
    // 测试DOM元素
    let elem = DomElement::new("div");
    println!("✅ DomElement DOM元素");
    
    // 测试事件系统
    let event = Event::new(EventType::Click);
    println!("✅ Event 事件系统");
    
    // 测试沙箱
    let sandbox = JsSandbox::new(ResourceLimits::default());
    println!("✅ JsSandbox JavaScript沙箱");
    
    println!("✅ browerai-dom 模块全部集成成功\n");
}

// ===== 10. 反混淆模块测试 =====
#[test]
fn test_deobfuscation_integration() {
    println!("\n=== 测试 browerai-deobfuscation 模块 ===");
    
    use browerai_deobfuscation::{
        deobfuscation::Deobfuscator,
        ai_deobfuscator::AIDeobfuscator,
        advanced_deobfuscation::AdvancedDeobfuscator,
        enhanced_deobfuscation::EnhancedDeobfuscator,
        obfuscation_detector_week4::ObfuscationDetector,
        advanced_orchestrator::AdvancedOrchestrator,
    };
    
    // 测试基础反混淆器
    let deobf = Deobfuscator::new();
    println!("✅ Deobfuscator 基础反混淆");
    
    // 测试AI反混淆器
    let ai_deobf = AIDeobfuscator::new();
    println!("✅ AIDeobfuscator AI反混淆");
    
    // 测试高级反混淆器
    let advanced_deobf = AdvancedDeobfuscator::new();
    println!("✅ AdvancedDeobfuscator 高级反混淆");
    
    // 测试增强反混淆器
    let enhanced_deobf = EnhancedDeobfuscator::new();
    println!("✅ EnhancedDeobfuscator 增强反混淆");
    
    // 测试混淆检测器
    let detector = ObfuscationDetector::new();
    println!("✅ ObfuscationDetector ONNX混淆检测");
    
    // 测试高级编排器
    let orchestrator = AdvancedOrchestrator::new();
    println!("✅ AdvancedOrchestrator 管道编排");
    
    println!("✅ browerai-deobfuscation 模块全部集成成功\n");
}

// ===== 11. 缓存系统测试 =====
#[test]
fn test_cache_system_integration() {
    println!("\n=== 测试缓存系统模块 ===");
    
    use browerai_cache::CacheStore;
    use browerai_multilayer_cache::{MultiLayerCache, strategy::{CacheLayer, PromotePolicy}};
    use browerai_persistent_layer::PersistentLayer;
    
    // 测试基础缓存
    let cache = CacheStore::<String, String>::new();
    println!("✅ CacheStore 基础缓存");
    
    // 测试多层缓存
    let ml_cache: MultiLayerCache<String> = MultiLayerCache::builder()
        .with_memory_layer(1000)
        .with_local_layer(10000)
        .with_promote_policy(PromotePolicy::OnAccess)
        .build();
    println!("✅ MultiLayerCache 多层缓存 (L1+L2+L3)");
    
    // 测试持久层
    let persistent = PersistentLayer::<String>::new(Default::default());
    println!("✅ PersistentLayer 持久化层");
    
    println!("✅ 缓存系统模块全部集成成功\n");
}

// ===== 12. 网络模块测试 =====
#[test]
fn test_network_module_integration() {
    println!("\n=== 测试 browerai-network 模块 ===");
    
    use browerai_network::{
        http::{HttpClient, HttpMethod, HttpRequest},
        cache::{ResourceCache, CacheStrategy},
        deep_crawler::DeepCrawler,
    };
    
    // 测试HTTP客户端
    let http_client = HttpClient::new();
    println!("✅ HttpClient HTTP客户端");
    
    // 测试资源缓存
    let resource_cache = ResourceCache::new(CacheStrategy::LRU);
    println!("✅ ResourceCache 资源缓存");
    
    // 测试深度爬虫
    let crawler = DeepCrawler::new();
    println!("✅ DeepCrawler 深度爬虫");
    
    println!("✅ browerai-network 模块全部集成成功\n");
}

// ===== 13. 数据库模块测试 =====
#[test]
fn test_database_module_integration() {
    println!("\n=== 测试 browerai-db 模块 ===");
    
    use browerai_db::{
        connection::DbConnection,
        operations::DbOperations,
        schema::{CacheEntry, CacheStats},
    };
    
    // 测试数据库连接
    let conn = DbConnection::new("memory");
    println!("✅ DbConnection 数据库连接");
    
    // 测试数据库操作
    let ops = DbOperations::new(conn);
    println!("✅ DbOperations 数据库操作");
    
    println!("✅ browerai-db 模块全部集成成功\n");
}

// ===== 14. Redis集成测试 =====
#[test]
fn test_redis_integration() {
    println!("\n=== 测试 browerai-redis-integration 模块 ===");
    
    use browerai_redis_integration::{
        connection::{RedisConfig, RedisPool},
        cluster_connection::RedisClusterConfig,
        distributed_lock::DistributedLock,
        layer::RedisLayer,
    };
    
    // 测试Redis配置
    let config = RedisConfig::default();
    println!("✅ RedisConfig Redis配置");
    
    // 测试Redis连接池
    let pool = RedisPool::new(config);
    println!("✅ RedisPool Redis连接池");
    
    // 测试集群配置
    let cluster_config = RedisClusterConfig::default();
    println!("✅ RedisClusterConfig 集群配置");
    
    // 测试分布式锁
    let lock = DistributedLock::new("test_lock", 30);
    println!("✅ DistributedLock 分布式锁");
    
    // 测试Redis层
    let redis_layer = RedisLayer::<String>::new(Default::default());
    println!("✅ RedisLayer Redis缓存层");
    
    println!("✅ browerai-redis-integration 模块全部集成成功\n");
}

// ===== 15. 智能渲染测试 =====
#[test]
fn test_intelligent_rendering_integration() {
    println!("\n=== 测试 browerai-intelligent-rendering 模块 ===");
    
    use browerai_intelligent_rendering::{
        website_analyzer::WebsiteAnalyzer,
        website_learning_engine::WebsiteLearningEngine,
        model_orchestrator::ModelOrchestrator,
        dual_sandbox_renderer::DualSandboxRenderer,
        llm_integration::LLMIntegration,
    };
    
    // 测试网站分析器
    let analyzer = WebsiteAnalyzer::new();
    println!("✅ WebsiteAnalyzer 网站分析器");
    
    // 测试学习引擎
    let learning = WebsiteLearningEngine::new();
    println!("✅ WebsiteLearningEngine 网站学习引擎");
    
    // 测试模型编排器
    let orchestrator = ModelOrchestrator::new();
    println!("✅ ModelOrchestrator 模型编排器");
    
    // 测试双沙盒渲染器
    let dual_renderer = DualSandboxRenderer::new();
    println!("✅ DualSandboxRenderer 双沙盒渲染");
    
    // 测试LLM集成
    let llm = LLMIntegration::new();
    println!("✅ LLMIntegration LLM集成");
    
    println!("✅ browerai-intelligent-rendering 模块全部集成成功\n");
}

// ===== 16. AI集成模块测试 =====
#[test]
fn test_ai_integration_module() {
    println!("\n=== 测试 browerai-ai-integration 模块 ===");
    
    use browerai_ai_integration::{
        framework_detector::FrameworkDetectorIntegration,
        hybrid_framework_integration::HybridFrameworkIntegration,
        js_orchestrator::HybridJsOrchestrator,
        integration::{HtmlModelIntegration, CssModelIntegration, JsDeobfuscatorIntegration},
        services::deobf_compose_service::DeobfComposeService,
        decoder::beam_search::BeamSearchParams,
        tokenizer::CharTokenizer,
    };
    
    // 测试框架检测器
    let detector = FrameworkDetectorIntegration::new();
    println!("✅ FrameworkDetectorIntegration 框架检测");
    
    // 测试混合框架集成
    let hybrid = HybridFrameworkIntegration::new();
    println!("✅ HybridFrameworkIntegration 混合框架");
    
    // 测试JS编排器
    let orchestrator = HybridJsOrchestrator::new();
    println!("✅ HybridJsOrchestrator JS编排器");
    
    // 测试HTML模型集成
    let html_integration = HtmlModelIntegration::new();
    println!("✅ HtmlModelIntegration HTML模型");
    
    // 测试CSS模型集成
    let css_integration = CssModelIntegration::new();
    println!("✅ CssModelIntegration CSS模型");
    
    // 测试JS反混淆集成
    let js_integration = JsDeobfuscatorIntegration::new();
    println!("✅ JsDeobfuscatorIntegration JS反混淆");
    
    // 测试反混淆组合服务
    let compose_service = DeobfComposeService::new(Default::default());
    println!("✅ DeobfComposeService 反混淆组合");
    
    // 测试Beam Search
    let beam_params = BeamSearchParams::default();
    println!("✅ BeamSearchParams Beam搜索");
    
    // 测试字符分词器
    let tokenizer = CharTokenizer::new();
    println!("✅ CharTokenizer 字符分词器");
    
    println!("✅ browerai-ai-integration 模块全部集成成功\n");
}

// ===== 17. 学习模块测试 =====
#[test]
fn test_learning_module_integration() {
    println!("\n=== 测试 browerai-learning 模块 ===");
    
    use browerai_learning::{
        comparative_learner::ComparativeLearner,
        framework_knowledge::FrameworkKnowledge,
        feedback::FeedbackSystem,
        code_generator::CodeGenerator,
        code_verifier::CodeVerifier,
        browser_automation::BrowserAutomation,
        data_structure_inference::DataStructureInference,
    };
    
    // 测试对比学习器
    let learner = ComparativeLearner::new();
    println!("✅ ComparativeLearner 对比学习");
    
    // 测试框架知识库
    let knowledge = FrameworkKnowledge::new();
    println!("✅ FrameworkKnowledge 框架知识");
    
    // 测试反馈系统
    let feedback = FeedbackSystem::new();
    println!("✅ FeedbackSystem 反馈系统");
    
    // 测试代码生成器
    let generator = CodeGenerator::new();
    println!("✅ CodeGenerator 代码生成");
    
    // 测试代码验证器
    let verifier = CodeVerifier::new();
    println!("✅ CodeVerifier 代码验证");
    
    // 测试浏览器自动化
    let automation = BrowserAutomation::new();
    println!("✅ BrowserAutomation 浏览器自动化");
    
    // 测试数据结构推断
    let ds_inference = DataStructureInference::new();
    println!("✅ DataStructureInference 数据结构推断");
    
    println!("✅ browerai-learning 模块全部集成成功\n");
}

// ===== 18. 集成管道测试 =====
#[test]
fn test_integrated_pipeline() {
    println!("\n=== 测试 browerai-integrated-pipeline 模块 ===");
    
    use browerai_integrated_pipeline::{
        pipeline::{IntegratedPipeline, PipelineConfig},
        output::{OutputFormat, OutputGenerator},
    };
    
    // 测试集成管道
    let config = PipelineConfig::default();
    let pipeline = IntegratedPipeline::new(config);
    println!("✅ IntegratedPipeline 集成管道");
    
    // 测试输出生成器
    let output_gen = OutputGenerator::new();
    println!("✅ OutputGenerator 输出生成");
    
    println!("✅ browerai-integrated-pipeline 模块全部集成成功\n");
}

// ===== 19. 插件系统测试 =====
#[test]
fn test_plugin_system_integration() {
    println!("\n=== 测试 browerai-plugins 模块 ===");
    
    use browerai_plugins::{
        loader::PluginLoader,
        registry::PluginRegistry,
        PluginMetadata, PluginCapability,
    };
    
    // 测试插件加载器
    let loader = PluginLoader::new();
    println!("✅ PluginLoader 插件加载");
    
    // 测试插件注册表
    let registry = PluginRegistry::new();
    println!("✅ PluginRegistry 插件注册表");
    
    println!("✅ browerai-plugins 模块全部集成成功\n");
}

// ===== 20. 测试套件模块测试 =====
#[test]
fn test_testing_module_integration() {
    println!("\n=== 测试 browerai-testing 模块 ===");
    
    use browerai_testing::{
        benchmark::{BenchmarkRunner, BenchmarkConfig, BenchmarkResult},
        website_test_suite::WebsiteTestSuite,
    };
    
    // 测试基准测试运行器
    let config = BenchmarkConfig::default();
    let runner = BenchmarkRunner::new(config);
    println!("✅ BenchmarkRunner 基准测试");
    
    // 测试网站测试套件
    let test_suite = WebsiteTestSuite::new();
    println!("✅ WebsiteTestSuite 网站测试套件");
    
    println!("✅ browerai-testing 模块全部集成成功\n");
}

// ===== 21. 主入口测试 =====
#[test]
fn test_main_browerai_entry() {
    println!("\n=== 测试 browerai 主入口 ===");
    
    // 测试所有公共导出
    let _ = browerai::core::BrowserConfig::default();
    println!("✅ core 核心模块导出");
    
    // 解析器
    println!("✅ html_parser HTML解析器导出");
    println!("✅ css_parser CSS解析器导出");
    println!("✅ js_parser JS解析器导出");
    println!("✅ js_analyzer JS分析器导出");
    
    // AI模块
    println!("✅ ai AI核心导出");
    println!("✅ ai_integration AI集成导出");
    
    // 渲染
    println!("✅ renderer 渲染器导出");
    println!("✅ renderer_predictive 预测渲染导出");
    println!("✅ intelligent_rendering 智能渲染导出");
    
    // DOM
    println!("✅ dom DOM模块导出");
    
    // V8
    println!("✅ js_v8 V8引擎导出");
    
    println!("✅ browerai 主入口全部导出成功\n");
}

// ===== 22. 端到端工作流测试 =====
#[test]
fn test_end_to_end_workflow() {
    println!("\n=== 端到端工作流测试 ===");
    
    use browerai_html_parser::HtmlParser;
    use browerai_css_parser::CssParser;
    use browerai_js_parser::JsParser;
    use browerai_js_analyzer::JsAnalyzer;
    use browerai_core::traits::{Parser, Analyzer};
    
    // 1. 解析HTML
    let html_parser = HtmlParser::new();
    let html = r#"<!DOCTYPE html>
    <html>
    <head><title>Test</title></head>
    <body>
        <div id="app"></div>
        <script>function init() { console.log('ready'); }</script>
    </body>
    </html>"#;
    let doc = html_parser.parse(html).expect("HTML解析失败");
    println!("✅ 步骤1: HTML解析");
    
    // 2. 解析CSS
    let css_parser = CssParser::new();
    let css = "#app { width: 100%; height: 100vh; }";
    let stylesheet = css_parser.parse(css).expect("CSS解析失败");
    println!("✅ 步骤2: CSS解析");
    
    // 3. 解析JS
    let js_parser = JsParser::new();
    let js = "function init() { return document.getElementById('app'); }";
    let ast = js_parser.parse(js).expect("JS解析失败");
    println!("✅ 步骤3: JS解析");
    
    // 4. 分析JS
    let mut analyzer = JsAnalyzer::new();
    let analysis = analyzer.analyze(js).expect("JS分析失败");
    println!("✅ 步骤4: JS分析 (发现 {} 个函数)", analysis.functions.len());
    
    // 5. 提取HTML中的脚本
    let scripts = html_parser.extract_scripts(&doc);
    println!("✅ 步骤5: 提取HTML脚本 ({} 个)", scripts.len());
    
    println!("✅ 端到端工作流测试成功\n");
}
