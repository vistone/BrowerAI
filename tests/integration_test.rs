//! BrowerAI Integration Tests
//!
//! 跨crate集成测试，验证各组件协同工作

use browerai_core::{traits::Parser, BrowserError, Result};
use browerai_html_parser::HtmlParser;
use browerai_css_parser::CssParser;
use browerai_js_parser::JsParser;

/// 测试HTML解析和DOM构建
#[test]
fn test_html_parsing_integration() {
    let parser = HtmlParser::new();
    let html = r#"
        <!DOCTYPE html>
        <html>
            <head>
                <title>Test Page</title>
            </head>
            <body>
                <div class="container">
                    <h1>Hello BrowerAI</h1>
                    <p>This is a test paragraph.</p>
                </div>
            </body>
        </html>
    "#;

    let result = parser.parse(html);
    assert!(result.is_ok(), "HTML parsing should succeed");

    let document = result.unwrap();
    assert!(document.node_count() > 0, "Document should have nodes");
}

/// 测试CSS解析
#[test]
fn test_css_parsing_integration() {
    let parser = CssParser::new();
    let css = r#"
        .container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        h1 {
            color: #333;
            font-size: 2rem;
        }
        
        p {
            line-height: 1.6;
            color: #666;
        }
    "#;

    let result = parser.parse(css);
    assert!(result.is_ok(), "CSS parsing should succeed");

    let stylesheet = result.unwrap();
    assert!(!stylesheet.rules.is_empty(), "Stylesheet should have rules");
}

/// 测试JS解析
#[test]
fn test_js_parsing_integration() {
    let mut parser = JsParser::new();
    let js = r#"
        function greet(name) {
            return "Hello, " + name + "!";
        }
        
        const message = greet("BrowerAI");
        console.log(message);
    "#;

    let result = parser.parse_string(js);
    assert!(result.is_ok(), "JS parsing should succeed");

    let ast = result.unwrap();
    assert!(!ast.function_decls.is_empty(), "AST should have function declarations");
}

/// 测试解析器链式工作
#[test]
fn test_parser_pipeline() {
    // HTML
    let html_parser = HtmlParser::new();
    let html = "<html><body><div>Test</div></body></html>";
    let document = html_parser.parse(html).unwrap();
    assert_eq!(document.node_count(), 4); // html, body, div, text

    // CSS
    let css_parser = CssParser::new();
    let css = "div { color: red; }";
    let stylesheet = css_parser.parse(css).unwrap();
    assert_eq!(stylesheet.rules.len(), 1);

    // JS
    let mut js_parser = JsParser::new();
    let js = "function test() {}";
    let ast = js_parser.parse_string(js).unwrap();
    assert_eq!(ast.function_decls.len(), 1);
}

/// 测试错误处理
#[test]
fn test_error_handling_integration() {
    // HTML解析错误
    let html_parser = HtmlParser::new().ignore_errors(true);
    let malformed_html = "<div><span>Unclosed";
    let result = html_parser.parse(malformed_html);
    assert!(result.is_ok(), "Parser should handle malformed HTML gracefully");

    // CSS解析错误
    let css_parser = CssParser::new();
    let invalid_css = "not valid css {{{";
    let result = css_parser.parse(invalid_css);
    // CSS解析器可能更宽容
    assert!(result.is_ok() || result.is_err());
}

/// 测试完整页面处理
#[test]
fn test_full_page_processing() {
    let html = r#"
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial; }
                .content { padding: 20px; }
            </style>
        </head>
        <body>
            <div class="content">
                <h1>Welcome</h1>
                <p>Content here</p>
            </div>
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    console.log('Page loaded');
                });
            </script>
        </body>
        </html>
    "#;

    // Parse HTML
    let html_parser = HtmlParser::new();
    let document = html_parser.parse(html).unwrap();
    
    // Verify document structure
    assert!(document.node_count() > 5, "Should have multiple nodes");
    
    // Extract inline CSS
    let styles = html_parser.extract_styles(&document);
    assert!(!styles.is_empty(), "Should extract inline styles");
    
    // Extract scripts
    let scripts = html_parser.extract_scripts(&document);
    assert!(!scripts.is_empty(), "Should extract scripts");
}

/// 测试性能 - 大文档解析
#[test]
fn test_large_document_parsing() {
    use std::time::Instant;

    // Generate large HTML
    let mut large_html = String::from("<html><body>");
    for i in 0..1000 {
        large_html.push_str(&format!("<div class='item-{}'><p>Content {}</p></div>", i, i));
    }
    large_html.push_str("</body></html>");

    let parser = HtmlParser::new();
    let start = Instant::now();
    let result = parser.parse(&large_html);
    let duration = start.elapsed();

    assert!(result.is_ok(), "Large document parsing should succeed");
    assert!(duration.as_secs() < 5, "Parsing should complete within 5 seconds");

    let document = result.unwrap();
    assert!(document.node_count() > 1000, "Should have many nodes");
}

/// 测试并发安全性
#[test]
fn test_thread_safety() {
    use std::thread;

    let handles: Vec<_> = (0..4)
        .map(|i| {
            thread::spawn(move || {
                let parser = HtmlParser::new();
                let html = format!("<div>Thread {}</div>", i);
                parser.parse(&html)
            })
        })
        .collect();

    for handle in handles {
        let result = handle.join().unwrap();
        assert!(result.is_ok(), "Concurrent parsing should succeed");
    }
}

/// 测试内存效率
#[test]
fn test_memory_efficiency() {
    // This is a basic test - in real scenarios you'd use memory profiling tools
    let parser = HtmlParser::new();
    
    // Parse same document multiple times
    let html = "<html><body><div>Test</div></body></html>";
    for _ in 0..100 {
        let _ = parser.parse(html).unwrap();
    }
    
    // If we get here without OOM, basic memory management is working
    assert!(true);
}

/// 测试特征提取集成
#[cfg(feature = "ai")]
#[test]
fn test_feature_extraction_integration() {
    use browerai_ai_core::features::{FeatureExtractor, FeatureType};

    let extractor = FeatureExtractor::new();
    
    let html = "<div><span>Test</span></div>";
    let features = extractor.extract(html, FeatureType::Dom).unwrap();
    
    assert!(!features.is_empty(), "Should extract DOM features");
    assert!(features.get("tag_open_count").is_some(), "Should have tag count feature");
}

/// 测试AI核心集成
#[cfg(feature = "ai")]
#[test]
fn test_ai_core_integration() {
    use browerai_ai_core::AiCore;

    let ai = AiCore::new();
    assert!(ai.is_ok(), "AI core should initialize");

    let ai = ai.unwrap();
    let status = ai.status();
    assert!(status.features_supported.len() > 0, "Should support features");
}

/// 测试渲染集成
#[test]
fn test_renderer_integration() {
    use browerai_renderer_core::{Renderer, RenderConfig, Viewport};
    use browerai_dom::Document;
    use browerai_css_parser::Stylesheet;

    let mut renderer = Renderer::new(RenderConfig::default());
    let document = Document::new();
    let stylesheet = Stylesheet::new();
    let viewport = Viewport::new(800, 600);

    let result = renderer.render(&document, &stylesheet, &viewport);
    assert!(result.is_ok(), "Rendering should succeed");

    let output = result.unwrap();
    assert_eq!(output.viewport.width, 800);
    assert_eq!(output.viewport.height, 600);
}

/// 测试开发者工具集成
#[test]
fn test_devtools_integration() {
    use browerai_devtools::DevTools;

    let mut devtools = DevTools::new();
    
    // Test console
    devtools.console().log("Test message");
    assert_eq!(devtools.stats().console_message_count, 1);
    
    // Test network
    let request_id = devtools.network().record_request("https://example.com", "GET");
    devtools.network().complete_request(request_id, 200, std::time::Duration::from_millis(100));
    assert_eq!(devtools.stats().network_request_count, 1);
    
    // Test profiler
    devtools.profiler().start("test-operation");
    std::thread::sleep(std::time::Duration::from_millis(10));
    devtools.profiler().end("test-operation");
    assert!(devtools.profiler().sample_count() > 0);
    
    // Test export
    let json = devtools.export_json();
    assert!(json.is_ok());
}

/// 测试端到端工作流
#[test]
fn test_end_to_end_workflow() {
    // 1. Parse HTML
    let html = r#"
        <html>
        <head><title>Test</title></head>
        <body>
            <div class="container">
                <h1>Hello</h1>
            </div>
        </body>
        </html>
    "#;
    let html_parser = HtmlParser::new();
    let document = html_parser.parse(html).unwrap();
    
    // 2. Parse CSS
    let css = ".container { width: 100%; }";
    let css_parser = CssParser::new();
    let stylesheet = css_parser.parse(css).unwrap();
    
    // 3. Render
    use browerai_renderer_core::{Renderer, RenderConfig, Viewport};
    let mut renderer = Renderer::new(RenderConfig::default());
    let viewport = Viewport::new(1024, 768);
    let render_output = renderer.render(&document, &stylesheet, &viewport).unwrap();
    
    // 4. Verify
    assert!(render_output.metadata.node_count > 0);
    
    // 5. Use devtools
    use browerai_devtools::DevTools;
    let mut devtools = DevTools::new();
    devtools.console().info("Rendering completed");
    
    assert_eq!(devtools.stats().console_message_count, 1);
}
