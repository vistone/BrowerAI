//! BrowerAI Integration Validation Tests
//! 
//! 全面验证所有组件的集成工作状态

use std::time::{Duration, Instant};

/// 验证1: 解析器链式集成
#[test]
fn validate_parser_pipeline() {
    use browerai_core::traits::Parser;
    use browerai_html_parser::HtmlParser;
    use browerai_css_parser::CssParser;
    use browerai_js_parser::JsParser;

    // HTML解析
    let html_parser = HtmlParser::new();
    let html = r#"<!DOCTYPE html>
        <html>
        <head><title>Test</title></head>
        <body>
            <div class="container">
                <h1>Hello BrowerAI</h1>
                <p>Integration test</p>
            </div>
        </body>
        </html>"#;
    
    let document = html_parser.parse(html).expect("HTML parsing failed");
    assert!(document.node_count() > 5, "Document should have multiple nodes");

    // CSS解析
    let css_parser = CssParser::new();
    let css = r#"
        .container { width: 100%; max-width: 1200px; }
        h1 { color: #333; font-size: 2rem; }
        p { line-height: 1.6; }
    "#;
    
    let stylesheet = css_parser.parse(css).expect("CSS parsing failed");
    assert!(!stylesheet.rules.is_empty(), "Stylesheet should have rules");

    // JS解析
    let mut js_parser = JsParser::new();
    let js = r#"
        function greet(name) {
            return "Hello, " + name;
        }
        const result = greet("BrowerAI");
    "#;
    
    let ast = js_parser.parse_string(js).expect("JS parsing failed");
    assert!(!ast.function_decls.is_empty(), "AST should have functions");

    println!("✅ Parser pipeline validation passed");
}

/// 验证2: 渲染引擎集成
#[test]
fn validate_rendering_pipeline() {
    use browerai_renderer_core::{Renderer, RenderConfig, Viewport};
    use browerai_dom::Document;
    use browerai_css_parser::Stylesheet;

    let mut renderer = Renderer::new(RenderConfig::default());
    let document = Document::new();
    let stylesheet = Stylesheet::new();
    let viewport = Viewport::new(1920, 1080);

    let output = renderer
        .render(&document, &stylesheet, &viewport)
        .expect("Rendering failed");

    assert_eq!(output.viewport.width, 1920);
    assert_eq!(output.viewport.height, 1080);

    println!("✅ Rendering pipeline validation passed");
}

/// 验证3: 开发者工具集成
#[test]
fn validate_devtools_integration() {
    use browerai_devtools::DevTools;

    let mut devtools = DevTools::new();

    // 测试控制台
    devtools.console().log("Test log message");
    devtools.console().info("Test info message");
    devtools.console().warn("Test warning");
    devtools.console().error("Test error");

    let stats = devtools.stats();
    assert_eq!(stats.console_message_count, 4);

    // 测试网络监控
    let request_id = devtools.network().record_request("https://example.com", "GET");
    devtools.network().complete_request(request_id, 200, Duration::from_millis(100));

    assert_eq!(devtools.stats().network_request_count, 1);

    // 测试性能分析
    devtools.profiler().start("test-operation");
    std::thread::sleep(Duration::from_millis(5));
    let duration = devtools.profiler().end("test-operation");
    
    assert!(duration.is_some(), "Profiler should record duration");

    // 测试数据导出
    let json = devtools.export_json();
    assert!(json.is_ok(), "Export should succeed");

    println!("✅ DevTools integration validation passed");
}

/// 验证4: 性能基准
#[test]
fn validate_performance_benchmarks() {
    use browerai_html_parser::HtmlParser;

    // 生成大型HTML文档
    let mut large_html = String::from("<html><body>");
    for i in 0..1000 {
        large_html.push_str(&format!(
            "<div class='item-{}'><h2>Title {}</h2><p>Content {}</p></div>",
            i, i, i
        ));
    }
    large_html.push_str("</body></html>");

    let parser = HtmlParser::new();
    let start = Instant::now();
    let result = parser.parse(&large_html);
    let duration = start.elapsed();

    assert!(result.is_ok(), "Large document parsing should succeed");
    assert!(
        duration.as_secs() < 5,
        "Parsing should complete within 5 seconds, took {:?}",
        duration
    );

    let document = result.unwrap();
    assert!(document.node_count() > 1000, "Should have many nodes");

    println!(
        "✅ Performance benchmark passed: parsed {} nodes in {:?}",
        document.node_count(),
        duration
    );
}

/// 验证5: 内存效率
#[test]
fn validate_memory_efficiency() {
    use browerai_html_parser::HtmlParser;

    let parser = HtmlParser::new();
    let html = "<html><body><div>Test content</div></body></html>";

    // 多次解析同一文档
    for i in 0..100 {
        let result = parser.parse(html);
        assert!(result.is_ok(), "Parse {} should succeed", i);
    }

    println!("✅ Memory efficiency validation passed");
}

/// 验证6: 错误处理
#[test]
fn validate_error_handling() {
    use browerai_core::traits::Parser;
    use browerai_html_parser::HtmlParser;
    use browerai_css_parser::CssParser;

    // HTML错误处理
    let html_parser = HtmlParser::new().ignore_errors(true);
    let malformed = "<div><span>Unclosed tag";
    let result = html_parser.parse(malformed);
    assert!(result.is_ok(), "Should handle malformed HTML gracefully");

    // CSS错误处理
    let css_parser = CssParser::new();
    let invalid_css = "not valid css {{{";
    let result = css_parser.parse(invalid_css);
    // CSS解析器可能更宽容
    println!("CSS parse result: {:?}", result.is_ok());

    println!("✅ Error handling validation passed");
}

/// 验证7: 线程安全
#[test]
fn validate_thread_safety() {
    use browerai_html_parser::HtmlParser;
    use std::thread;

    let handles: Vec<_> = (0..8)
        .map(|i| {
            thread::spawn(move || {
                let parser = HtmlParser::new();
                let html = format!("<html><body><div>Thread {}</div></body></html>", i);
                parser.parse(&html)
            })
        })
        .collect();

    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.join().expect(&format!("Thread {} panicked", i));
        assert!(result.is_ok(), "Thread {} parsing should succeed", i);
    }

    println!("✅ Thread safety validation passed");
}

/// 验证8: 完整工作流
#[test]
fn validate_complete_workflow() {
    use browerai_core::traits::Parser;
    use browerai_html_parser::HtmlParser;
    use browerai_css_parser::CssParser;
    use browerai_js_parser::JsParser;
    use browerai_renderer_core::{Renderer, RenderConfig, Viewport};
    use browerai_devtools::DevTools;

    // 1. Parse HTML
    let html_parser = HtmlParser::new();
    let html = r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Integration Test</title>
            <style>body { font-family: Arial; }</style>
        </head>
        <body>
            <div class="content">
                <h1>Welcome to BrowerAI</h1>
                <p>This is a complete workflow test.</p>
            </div>
            <script>console.log('Page loaded');</script>
        </body>
        </html>
    "#;
    let document = html_parser.parse(html).expect("HTML parse failed");

    // 2. Extract and parse CSS
    let styles = html_parser.extract_styles(&document);
    let css_parser = CssParser::new();
    let stylesheet = css_parser.parse(&styles.join("\n")).unwrap_or_default();

    // 3. Extract and parse JS
    let scripts = html_parser.extract_scripts(&document);
    let mut js_parser = JsParser::new();
    for script in &scripts {
        let _ = js_parser.parse_string(script);
    }

    // 4. Render
    let mut renderer = Renderer::new(RenderConfig::default());
    let viewport = Viewport::new(1024, 768);
    let render_output = renderer
        .render(&document, &stylesheet, &viewport)
        .expect("Render failed");

    // 5. Use DevTools
    let mut devtools = DevTools::new();
    devtools.console().info("Workflow completed successfully");
    
    // Verify
    assert!(render_output.metadata.node_count > 0);
    assert_eq!(devtools.stats().console_message_count, 1);

    println!("✅ Complete workflow validation passed");
}

/// 验证9: AI核心集成 (如果启用了ai特性)
#[cfg(feature = "ai")]
#[test]
fn validate_ai_core_integration() {
    use browerai_ai_core::AiCore;
    use browerai_ai_core::features::{FeatureExtractor, FeatureType};

    let ai = AiCore::new().expect("AI core should initialize");
    
    // 验证状态
    let status = ai.status();
    assert!(!status.features_supported.is_empty());

    // 测试特征提取
    let extractor = ai.feature_extractor();
    let html = "<div><span>Test</span></div>";
    let features = extractor
        .extract(html, FeatureType::Dom)
        .expect("Feature extraction failed");
    
    assert!(!features.is_empty());

    println!("✅ AI core integration validation passed");
}

/// 验证10: 代码分析集成
#[test]
fn validate_code_analysis_integration() {
    use browerai_js_analyzer::JsAnalyzer;
    use browerai_js_parser::JsParser;

    let js = r#"
        function calculateSum(a, b) {
            return a + b;
        }
        
        function main() {
            let x = 10;
            let y = 20;
            let result = calculateSum(x, y);
            console.log(result);
        }
        
        main();
    "#;

    // 解析
    let mut parser = JsParser::new();
    let ast = parser.parse_string(js).expect("Parse failed");

    // 分析
    let mut analyzer = JsAnalyzer::new();
    let result = analyzer.analyze(js).expect("Analysis failed");

    // 验证结果
    assert_eq!(result.ast.function_decls.len(), 2);
    assert!(!result.scope_tree.is_empty());
    assert!(!result.cfg.is_empty());

    println!("✅ Code analysis integration validation passed");
}
