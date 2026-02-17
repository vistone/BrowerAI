use anyhow::Result;

#[test]
fn test_css_parser_basic() {
    use browerai_css_parser::CssParser;
    
    let parser = CssParser::new();
    let css = "body { color: red; }";
    let result = parser.parse(css);
    assert!(result.is_ok());
}

#[cfg(all(feature = "ai", feature = "onnx"))]
#[test]
#[ignore] // Requires ONNX models
fn test_css_parser_with_phase2_models() -> Result<()> {
    use browerai_css_parser::CssParser;
    
    // Create parser from model directory
    let parser = CssParser::from_model_dir("models/onnx_exports")?;
    
    // Test basic parsing
    let css = ".container { display: flex; }";
    let rules = parser.parse(css)?;
    assert!(!rules.is_empty());
    
    Ok(())
}

#[cfg(all(feature = "ai", feature = "onnx"))]
#[test]
#[ignore] // Requires ONNX models
fn test_selector_embedding() -> Result<()> {
    use browerai_css_parser::CssParser;
    
    let parser = CssParser::from_model_dir("models/onnx_exports")?;
    
    // Get embedding for selector
    let selector = ".main-container";
    let embedding = parser.get_selector_embedding(selector)?;
    
    // Embedding should have expected dimensions (50 * 128 = 6400)
    assert_eq!(embedding.len(), 6400);
    
    // Test caching
    let embedding2 = parser.get_selector_embedding(selector)?;
    assert_eq!(embedding, embedding2);
    assert_eq!(parser.cache_size(), 1);
    
    Ok(())
}

#[cfg(all(feature = "ai", feature = "onnx"))]
#[test]
#[ignore] // Requires ONNX models
fn test_property_prediction() -> Result<()> {
    use browerai_css_parser::CssParser;
    
    let parser = CssParser::from_model_dir("models/onnx_exports")?;
    
    // Predict properties for selector
    let selector = ".button";
    let properties = parser.predict_properties(selector)?;
    
    println!("✅ Predicted {} properties for '{}':", properties.len(), selector);
    for prop in properties.iter().take(5) {
        println!("  - {} (confidence: {:.2})", prop.name, prop.confidence);
    }
    
    Ok(())
}

#[cfg(all(feature = "ai", feature = "onnx"))]
#[test]
#[ignore] // Requires ONNX models
fn test_parse_with_ai() -> Result<()> {
    use browerai_css_parser::CssParser;
    
    let parser = CssParser::from_model_dir("models/onnx_exports")?;
    
    let css = r#"
        .header { color: blue; }
        .footer { background: gray; }
    "#;
    
    let enhanced_rules = parser.parse_with_ai(css)?;
    
    println!("✅ Enhanced {} CSS rules:", enhanced_rules.len());
    for rule in &enhanced_rules {
        println!("  - Selector: {}", rule.selector);
        println!("    Embedding dims: {}", rule.embedding.len());
    }
    
    assert!(!enhanced_rules.is_empty());
    
    Ok(())
}

#[cfg(all(feature = "ai", feature = "onnx"))]
#[test]
#[ignore] // Requires ONNX models
fn test_cache_operations() -> Result<()> {
    use browerai_css_parser::CssParser;
    
    let parser = CssParser::from_model_dir("models/onnx_exports")?;
    
    // Add some embeddings to cache
    parser.get_selector_embedding(".item1")?;
    parser.get_selector_embedding(".item2")?;
    parser.get_selector_embedding(".item3")?;
    
    assert_eq!(parser.cache_size(), 3);
    
    // Clear cache
    parser.clear_cache();
    assert_eq!(parser.cache_size(), 0);
    
    Ok(())
}

#[cfg(all(feature = "ai", feature = "onnx"))]
#[test]
#[ignore] // Requires ONNX models
fn test_batch_selector_processing() -> Result<()> {
    use browerai_css_parser::CssParser;
    
    let parser = CssParser::from_model_dir("models/onnx_exports")?;
    
    let selectors = vec![
        ".nav-item",
        "#main-content",
        ".card-title",
        "button.primary",
        "div.container > p",
    ];
    
    for selector in selectors {
        let embedding = parser.get_selector_embedding(selector)?;
        assert_eq!(embedding.len(), 6400);
        
        let properties = parser.predict_properties(selector)?;
        println!("✅ Selector '{}': {} predicted properties", selector, properties.len());
    }
    
    assert_eq!(parser.cache_size(), 5);
    
    Ok(())
}
