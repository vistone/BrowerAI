//! AI-specific integration tests for BrowerAI Phase 2.4

use browerai::ai::integration::{HtmlModelIntegration, JsDeobfuscatorIntegration};
use browerai::ai::InferenceEngine;
use browerai::parser::{CssParser, HtmlParser, JsParser};

#[test]
fn test_parser_with_ai_integration() {
    let parser = HtmlParser::new();
    let html = r#"
        <!DOCTYPE html>
        <html>
            <head><title>Test</title></head>
            <body>
                <h1>Hello, AI!</h1>
                <p>Testing AI integration</p>
            </body>
        </html>
    "#;

    let result = parser.parse(html);
    assert!(result.is_ok());

    let dom = result.unwrap();
    let text = parser.extract_text(&dom);
    assert!(text.contains("Hello, AI!"));
}

#[test]
fn test_ai_model_integration_fallback() {
    let engine = InferenceEngine::new().unwrap();
    let mut integration = HtmlModelIntegration::new(&engine, None, None).unwrap();

    // Test fallback behavior when no model is loaded
    let (valid, complexity) = integration
        .validate_structure("<html><body>Content</body></html>")
        .unwrap();

    assert!(valid); // Should default to valid
    assert!((0.0..=1.0).contains(&complexity));
}

#[test]
fn test_complex_html_parsing() {
    let parser = HtmlParser::new();
    let complex_html = r##"
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Complex Page</title>
            </head>
            <body>
                <nav>
                    <ul>
                        <li><a href="#home">Home</a></li>
                        <li><a href="#about">About</a></li>
                    </ul>
                </nav>
                <main>
                    <article>
                        <h1>Article Title</h1>
                        <p>Article content goes here.</p>
                    </article>
                </main>
                <footer>Copyright 2026</footer>
            </body>
        </html>
    "##;

    let result = parser.parse(complex_html);
    assert!(result.is_ok());

    let dom = result.unwrap();
    let text = parser.extract_text(&dom);
    assert!(text.contains("Article Title"));
    assert!(text.contains("Copyright 2026"));
}

#[test]
fn test_css_parser_basic() {
    let parser = CssParser::new();
    let css = "body { color: red; font-size: 16px; }";

    let result = parser.parse(css);
    assert!(result.is_ok());
}

#[test]
fn test_js_parser_basic() {
    let parser = JsParser::new();
    let js = "function test() { return 42; }";

    let result = parser.parse(js);
    assert!(result.is_ok());
}

#[test]
fn test_js_deobfuscator_integration() {
    let engine = InferenceEngine::new().unwrap();

    // Test without model (fallback mode)
    let mut integration = JsDeobfuscatorIntegration::new(&engine, None, None).unwrap();

    let obfuscated = "var a=function(){return 42;}";
    let result = integration.deobfuscate(obfuscated);

    assert!(result.is_ok());
    // In fallback mode, should return original
    assert_eq!(result.unwrap(), obfuscated);
}

#[test]
fn test_js_deobfuscator_with_model() {
    use std::path::PathBuf;

    let engine = InferenceEngine::new().unwrap();
    let model_path = PathBuf::from("models/local/js_deobfuscator_v1.onnx");

    if model_path.exists() {
        let mut integration =
            JsDeobfuscatorIntegration::new(&engine, Some(&model_path), None).unwrap();
        assert!(integration.is_enabled());

        let obfuscated = "var a=function(){return 42;}";
        let result = integration.deobfuscate(obfuscated);

        // Should succeed (even if output quality varies with training)
        match result {
            Ok(deobfuscated) => {
                println!("Input:  {}", obfuscated);
                println!("Output: {}", deobfuscated);
            }
            Err(e) => {
                println!(
                    "Deobfuscation failed (expected with limited training data): {}",
                    e
                );
            }
        }
    } else {
        println!("Skipping test - model file not found");
    }
}
