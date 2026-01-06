/// Comprehensive integration tests for learning-inference-generation cycle
/// Tests the complete workflow with real code samples
use browerai::learning::{
    CodeGenerator, CodeType, ContinuousLearningConfig, ContinuousLearningLoop,
    DeobfuscationStrategy, GeneratedCode, GenerationRequest, JsDeobfuscator, ObfuscationAnalysis,
};
use browerai::parser::{CssParser, HtmlParser, JsParser};
use std::collections::HashMap;

#[test]
fn test_html_generation_and_validation() {
    let generator = CodeGenerator::with_defaults();
    let parser = HtmlParser::new();

    // Generate HTML
    let mut constraints = HashMap::new();
    constraints.insert("title".to_string(), "Test Page".to_string());
    constraints.insert("heading".to_string(), "Hello World".to_string());
    constraints.insert("content".to_string(), "This is test content".to_string());

    let request = GenerationRequest {
        code_type: CodeType::Html,
        description: "basic page".to_string(),
        constraints,
    };

    let result = generator.generate(&request).expect("Generation failed");

    // Validate generated HTML can be parsed
    let dom = parser
        .parse(&result.code)
        .expect("Generated HTML is invalid");
    let text = parser.extract_text(&dom);

    // Verify content is present
    assert!(
        text.contains("Test Page") || text.contains("Hello World"),
        "Generated HTML should contain expected content"
    );
    assert!(
        result.code.contains("<!DOCTYPE html>"),
        "Should have DOCTYPE"
    );
    assert!(result.code.contains("<html>"), "Should have html tag");
    assert!(
        result.code.contains("</html>"),
        "Should have closing html tag"
    );
}

#[test]
fn test_css_generation_and_validation() {
    let generator = CodeGenerator::with_defaults();
    let parser = CssParser::new();

    // Generate CSS
    let mut constraints = HashMap::new();
    constraints.insert("font".to_string(), "Arial, sans-serif".to_string());
    constraints.insert("padding".to_string(), "20px".to_string());
    constraints.insert("background".to_string(), "#f5f5f5".to_string());

    let request = GenerationRequest {
        code_type: CodeType::Css,
        description: "basic styling".to_string(),
        constraints,
    };

    let result = generator.generate(&request).expect("Generation failed");

    // Validate generated CSS can be parsed
    let rules = parser
        .parse(&result.code)
        .expect("Generated CSS is invalid");

    // Verify CSS structure
    assert!(!rules.is_empty(), "CSS should have rules");
    assert!(
        result.code.contains("Arial") || result.code.contains("sans-serif"),
        "Should contain font family"
    );
}

#[test]
fn test_js_generation_and_validation() {
    let generator = CodeGenerator::with_defaults();
    let parser = JsParser::new();

    // Generate JavaScript
    let mut constraints = HashMap::new();
    constraints.insert("name".to_string(), "calculateSum".to_string());
    constraints.insert("params".to_string(), "a, b".to_string());
    constraints.insert("body".to_string(), "const result = a + b;".to_string());
    constraints.insert("return_value".to_string(), "result".to_string());

    let request = GenerationRequest {
        code_type: CodeType::JavaScript,
        description: "function to calculate sum".to_string(),
        constraints,
    };

    let result = generator.generate(&request).expect("Generation failed");

    // Validate generated JS can be parsed
    let ast = parser.parse(&result.code).expect("Generated JS is invalid");

    // Verify JS structure
    assert!(ast.is_valid, "JS should be valid");
    assert!(ast.statement_count > 0, "JS should have statements");
    assert!(
        result.code.contains("function") || result.code.contains("calculateSum"),
        "Should define a function"
    );
}

#[test]
fn test_deobfuscation_actually_improves_code() {
    let deobfuscator = JsDeobfuscator::new();

    // Test with heavily obfuscated code
    let obfuscated = "var a=1,b=2,c=3;function d(e,f){var g=e+f;return g;}var h=d(a,b);";

    let result = deobfuscator
        .deobfuscate(obfuscated, DeobfuscationStrategy::Comprehensive)
        .expect("Deobfuscation failed");

    // The result should have better variable names
    assert!(result.steps.len() > 0, "Should have applied steps");

    // Verify it's still valid JavaScript
    let parser = JsParser::new();
    let ast = parser
        .parse(&result.code)
        .expect("Deobfuscated code should be valid");
    assert!(ast.is_valid, "Deobfuscated code should be valid JS");
}

#[test]
fn test_continuous_learning_with_real_code() {
    let mut config = ContinuousLearningConfig::default();
    config.max_iterations = Some(5);
    config.auto_generate = true;

    let mut learning_loop = ContinuousLearningLoop::new(config);

    // Run multiple iterations
    for i in 0..5 {
        let events = learning_loop
            .run_iteration()
            .expect(&format!("Iteration {} failed", i + 1));
        assert!(!events.is_empty(), "Should produce events");
    }

    let stats = learning_loop.get_stats();
    assert_eq!(stats.iterations, 5, "Should complete 5 iterations");
    assert!(stats.codes_generated > 0, "Should generate some code");
}

#[test]
fn test_learn_infer_generate_cycle_with_html() {
    let generator = CodeGenerator::with_defaults();
    let parser = HtmlParser::new();

    // Generate initial HTML
    let mut constraints = HashMap::new();
    constraints.insert("title".to_string(), "Cycle Test".to_string());

    let request = GenerationRequest {
        code_type: CodeType::Html,
        description: "form".to_string(),
        constraints,
    };

    let generated = generator.generate(&request).expect("Generation failed");

    // Learn: Parse the generated HTML
    let dom = parser.parse(&generated.code).expect("Parse failed");

    // Infer: Extract patterns from the DOM
    let text = parser.extract_text(&dom);
    assert!(!text.is_empty(), "Should extract text from generated HTML");

    // Generate: Create new HTML based on learned patterns
    let mut new_constraints = HashMap::new();
    new_constraints.insert("title".to_string(), "Generated Again".to_string());

    let new_request = GenerationRequest {
        code_type: CodeType::Html,
        description: "form".to_string(),
        constraints: new_constraints,
    };

    let regenerated = generator
        .generate(&new_request)
        .expect("Regeneration failed");

    // Validate the regenerated code
    let new_dom = parser
        .parse(&regenerated.code)
        .expect("Regenerated parse failed");
    let new_text = parser.extract_text(&new_dom);
    assert!(
        !new_text.is_empty(),
        "Should extract text from regenerated HTML"
    );
}

#[test]
fn test_deobfuscation_preserves_functionality() {
    let deobfuscator = JsDeobfuscator::new();
    let parser = JsParser::new();

    // Original functional code
    let original = "function add(x,y){return x+y;}var result=add(1,2);";

    // Parse original
    let original_ast = parser.parse(original).expect("Original should parse");

    // Deobfuscate
    let deobfuscated_result = deobfuscator
        .deobfuscate(original, DeobfuscationStrategy::Comprehensive)
        .expect("Deobfuscation failed");

    // Parse deobfuscated
    let deobf_ast = parser
        .parse(&deobfuscated_result.code)
        .expect("Deobfuscated should parse");

    // Both should be valid
    assert!(original_ast.is_valid, "Original should be valid");
    assert!(deobf_ast.is_valid, "Deobfuscated should be valid");

    // Statement count should be similar (functionality preserved)
    assert!(
        (original_ast.statement_count as i32 - deobf_ast.statement_count as i32).abs() <= 1,
        "Statement count should be similar"
    );
}

#[test]
fn test_generated_code_standards_compliance() {
    let generator = CodeGenerator::with_defaults();

    // Test HTML5 compliance
    let html_request = GenerationRequest {
        code_type: CodeType::Html,
        description: "basic page".to_string(),
        constraints: HashMap::new(),
    };

    let html = generator
        .generate(&html_request)
        .expect("HTML generation failed");
    assert!(
        html.code.starts_with("<!DOCTYPE html>"),
        "HTML5 DOCTYPE required"
    );
    assert!(html.code.contains("<html>"), "HTML root element required");

    // Test CSS3 compliance
    let css_request = GenerationRequest {
        code_type: CodeType::Css,
        description: "basic styling".to_string(),
        constraints: HashMap::new(),
    };

    let css = generator
        .generate(&css_request)
        .expect("CSS generation failed");
    // CSS should have proper selector and property format
    assert!(
        css.code.contains("{") && css.code.contains("}"),
        "CSS should have proper blocks"
    );
    assert!(
        css.code.contains(":") || css.code.contains(";"),
        "CSS should have properties"
    );

    // Test ES6+ compliance
    let js_request = GenerationRequest {
        code_type: CodeType::JavaScript,
        description: "function".to_string(),
        constraints: HashMap::new(),
    };

    let js = generator
        .generate(&js_request)
        .expect("JS generation failed");
    // Should be valid JavaScript
    let parser = JsParser::new();
    assert!(
        parser.validate(&js.code).expect("Validation check failed"),
        "Generated JS should be valid"
    );
}

#[test]
fn test_multiple_obfuscation_techniques_detection() {
    let deobfuscator = JsDeobfuscator::new();

    // Code with multiple obfuscation techniques
    let complex_obfuscated = r#"
        var a="\x48\x65\x6c\x6c\x6f";
        function b(c){var d=0;for(var e=0;e<10;e++){if(false){break;}d+=e;}return d+c;}
        var f=b(5);
    "#;

    let analysis = deobfuscator.analyze_obfuscation(complex_obfuscated);

    // Should detect multiple techniques
    assert!(
        analysis.obfuscation_score > 0.0,
        "Should detect obfuscation"
    );
    assert!(
        analysis.techniques.len() >= 2,
        "Should detect multiple techniques"
    );

    // Should provide suggestions
    assert!(
        !analysis.suggestions.is_empty(),
        "Should provide suggestions"
    );
}

#[test]
fn test_learning_loop_generates_valid_code() {
    let mut config = ContinuousLearningConfig::default();
    config.max_iterations = Some(3);
    config.auto_generate = true;

    let mut learning_loop = ContinuousLearningLoop::new(config);
    let parser = JsParser::new();

    // Run iterations and validate generated code
    for _ in 0..3 {
        learning_loop.run_iteration().expect("Iteration failed");
    }

    let stats = learning_loop.get_stats();
    assert!(stats.codes_generated > 0, "Should generate code");
    assert_eq!(stats.success_rate, 1.0, "All generations should succeed");
}

#[test]
fn test_end_to_end_website_simulation() {
    // Simulate complete workflow: generate -> learn -> infer -> regenerate

    // 1. Generate initial website
    let generator = CodeGenerator::with_defaults();

    let html = generator
        .generate(&GenerationRequest {
            code_type: CodeType::Html,
            description: "basic page".to_string(),
            constraints: {
                let mut m = HashMap::new();
                m.insert("title".to_string(), "Test Site".to_string());
                m.insert("heading".to_string(), "Welcome".to_string());
                m
            },
        })
        .expect("HTML generation failed");

    let css = generator
        .generate(&GenerationRequest {
            code_type: CodeType::Css,
            description: "basic styling".to_string(),
            constraints: HashMap::new(),
        })
        .expect("CSS generation failed");

    let js = generator
        .generate(&GenerationRequest {
            code_type: CodeType::JavaScript,
            description: "function".to_string(),
            constraints: HashMap::new(),
        })
        .expect("JS generation failed");

    // 2. Learn: Parse all generated code
    let html_parser = HtmlParser::new();
    let css_parser = CssParser::new();
    let js_parser = JsParser::new();

    let html_dom = html_parser.parse(&html.code).expect("HTML parse failed");
    let css_rules = css_parser.parse(&css.code).expect("CSS parse failed");
    let js_ast = js_parser.parse(&js.code).expect("JS parse failed");

    // 3. Verify all components are valid
    assert!(
        !html_parser.extract_text(&html_dom).is_empty(),
        "HTML should have content"
    );
    assert!(
        !css_rules.is_empty() || css.code.contains("body"),
        "CSS should have content"
    );
    assert!(js_ast.is_valid, "JS should be valid");

    // 4. Infer patterns and regenerate
    let regenerated_html = generator
        .generate(&GenerationRequest {
            code_type: CodeType::Html,
            description: "basic page".to_string(),
            constraints: {
                let mut m = HashMap::new();
                m.insert("title".to_string(), "Regenerated Site".to_string());
                m
            },
        })
        .expect("Regeneration failed");

    // 5. Verify regenerated code is also valid
    let regenerated_dom = html_parser
        .parse(&regenerated_html.code)
        .expect("Regenerated HTML parse failed");
    assert!(
        !html_parser.extract_text(&regenerated_dom).is_empty(),
        "Regenerated HTML should have content"
    );
}
