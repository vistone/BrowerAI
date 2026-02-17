/// End-to-End Integration Tests for Week 6 Learning Pipeline
/// Tests the complete flow: URL → Feature Extraction → Python API → Feedback

use crate::{
    WebsiteFeatureExtractor, RustPythonBridge, FeedbackCollector,
    FeaturePacket, RenderingComparison, data_models::PageContent,
    learning_sandbox::{WebsiteIntent, intent_analyzer::{ComplexityLevel, LayoutType, PageStructure}},
};
use std::collections::HashMap;

#[test]
fn test_feature_extraction_e2e() {
    // Create a sample page
    let mut page = PageContent::new(
        "https://example.com".to_string(),
        r#"
            <!DOCTYPE html>
            <html>
            <head><title>Example</title></head>
            <body>
                <header><nav><a href="/">Home</a></nav></header>
                <main>
                    <article>
                        <h1>Example Article</h1>
                        <p>Content here</p>
                        <button>Click me</button>
                    </article>
                </main>
                <footer>Copyright 2024</footer>
            </body>
            </html>
        "#.to_string(),
        std::collections::HashMap::new(),
    );

    let intent = WebsiteIntent {
        website_type: "blog".to_string(),
        confidence: 0.95,
        core_features: vec!["articles".to_string(), "navigation".to_string()],
        target_audience: "readers".to_string(),
        design_style: crate::learning_sandbox::intent_analyzer::DesignStyle {
            formality: 0.7,
            colorfulness: 0.8,
            minimalism: 0.3,
            modernity: 0.9,
            primary_colors: None,
            layout_type: None,
        },
        structure: PageStructure {
            has_header: true,
            has_navigation: true,
            has_sidebar: false,
            has_main_content: true,
            has_footer: true,
            layout_type: LayoutType::TwoColumn,
            section_count: 5,
            complexity: ComplexityLevel::Moderate,
        },
        business_model: "content".to_string(),
        type_scores: std::collections::HashMap::new(),
    };

    // Extract features
    let features = WebsiteFeatureExtractor::extract(&page, &intent).unwrap();

    // Verify
    assert_eq!(features.len(), 48);
    assert!(features.iter().all(|f| f.is_finite() && *f >= 0.0));
}

#[test]
fn test_feature_packet_serialization() {
    let packet = FeaturePacket {
        url: "https://example.com".to_string(),
        features: vec![0.1; 48],
        website_intent: "blog".to_string(),
        design_style: "modern".to_string(),
        feedback: None,
        timestamp: 1704067200,
        session_id: "sess123".to_string(),
    };

    // Serialize to JSON
    let json = serde_json::to_string(&packet).unwrap();
    
    // Deserialize back
    let restored: FeaturePacket = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.url, packet.url);
    assert_eq!(restored.features.len(), 48);
    assert_eq!(restored.session_id, packet.session_id);
}

#[test]
fn test_feedback_collection_workflow() {
    let mut collector = FeedbackCollector::new();

    let comparison = RenderingComparison {
        url: "https://example.com".to_string(),
        original_html: r#"<div class="header"><h1>Title</h1></div>"#.to_string(),
        generated_html: r#"<div class="header"><h1>Title</h1></div>"#.to_string(),
        original_css: "h1 { font-size: 24px; }".to_string(),
        generated_css: "h1 { font-size: 24px; }".to_string(),
        original_js: "document.addEventListener('click', () => {});".to_string(),
        generated_js: "document.addEventListener('click', () => {});".to_string(),
        viewport_width: 1920,
        viewport_height: 1080,
        original_visual_hash: Some("abc123".to_string()),
        generated_visual_hash: Some("abc123".to_string()),
        html_similarity: 0.0,
        css_coverage: 0.0,
        js_functionality: 0.0,
        layout_similarity: 0.0,
        element_comparisons: vec![],
        css_rule_comparisons: vec![],
        event_handler_comparisons: vec![],
        overall_quality: 0.0,
        feedback: String::new(),
    };

    // Collect feedback
    let result = collector.compare_rendering(&comparison).unwrap();

    // Verify quality score
    assert!(result.overall_quality >= 0.0 && result.overall_quality <= 1.0);
    assert!(!result.feedback.is_empty());
    
    // Verify history tracking
    assert_eq!(collector.get_history().len(), 1);
    assert!(collector.get_average_quality() > 0.5);
}

#[test]
fn test_rust_python_bridge_creation() {
    let _bridge = RustPythonBridge::new("http://localhost:5000".to_string());

    // Create test packet
    let packet = FeaturePacket {
        url: "https://example.com".to_string(),
        features: vec![0.1; 48],
        website_intent: "blog".to_string(),
        design_style: "modern".to_string(),
        feedback: None,
        timestamp: 1704067200,
        session_id: "test-session".to_string(),
    };

    // Serialize packet
    let json = serde_json::to_string(&packet).unwrap();
    
    // Verify it's valid JSON
    let _: FeaturePacket = serde_json::from_str(&json).unwrap();
}

#[test]
fn test_complete_learning_loop_workflow() {
    // Step 1: Extract features from website
    let page = PageContent::new(
        "https://example.com".to_string(),
        r#"
            <html>
            <head><title>Blog</title></head>
            <body>
                <header>Blog Header</header>
                <main><article><h1>Post</h1><p>Content</p></article></main>
                <footer>Footer</footer>
            </body>
            </html>
        "#.to_string(),
        std::collections::HashMap::new(),
    );

    let intent = WebsiteIntent {
        website_type: "blog".to_string(),
        confidence: 0.9,
        core_features: vec!["articles".to_string()],
        target_audience: "readers".to_string(),
        design_style: crate::learning_sandbox::intent_analyzer::DesignStyle {
            formality: 0.5,
            colorfulness: 0.5,
            minimalism: 0.7,
            modernity: 0.6,
            primary_colors: None,
            layout_type: None,
        },
        structure: PageStructure {
            has_header: true,
            has_navigation: false,
            has_sidebar: false,
            has_main_content: true,
            has_footer: true,
            layout_type: LayoutType::SingleColumn,
            section_count: 3,
            complexity: ComplexityLevel::Simple,
        },
        business_model: "content".to_string(),
        type_scores: std::collections::HashMap::new(),
    };

    // Extract features
    let features = WebsiteFeatureExtractor::extract(&page, &intent).unwrap();
    assert_eq!(features.len(), 48);

    // Step 2: Create feature packet for Python
    let packet = FeaturePacket {
        url: page.url.clone(),
        features: features.clone(),
        website_intent: intent.website_type.clone(),
        design_style: format!("{:?}", intent.design_style.modernity),
        feedback: None,
        timestamp: 1704067200,
        session_id: "session-test".to_string(),
    };

    // Verify packet is valid JSON
    let json = serde_json::to_string(&packet).unwrap();
    let _restored: FeaturePacket = serde_json::from_str(&json).unwrap();

    // Step 3: Simulate feedback collection
    let mut collector = FeedbackCollector::new();
    
    let comparison = RenderingComparison {
        url: packet.url.clone(),
        original_html: page.html.clone(),
        generated_html: page.html.clone(), // Same for this test
        original_css: "body { margin: 0; }".to_string(),
        generated_css: "body { margin: 0; }".to_string(),
        original_js: "".to_string(),
        generated_js: "".to_string(),
        viewport_width: 1920,
        viewport_height: 1080,
        original_visual_hash: None,
        generated_visual_hash: None,
        html_similarity: 0.0,
        css_coverage: 0.0,
        js_functionality: 0.0,
        layout_similarity: 0.0,
        element_comparisons: vec![],
        css_rule_comparisons: vec![],
        event_handler_comparisons: vec![],
        overall_quality: 0.0,
        feedback: String::new(),
    };

    let result = collector.compare_rendering(&comparison).unwrap();
    
    // Step 4: Verify complete workflow
    assert!(result.overall_quality >= 0.0);
    assert!(!result.feedback.is_empty());
    assert_eq!(collector.get_history().len(), 1);
}

#[test]
fn test_training_metrics_workflow() {
    let mut metrics_map = HashMap::new();
    metrics_map.insert("precision".to_string(), 0.91);
    metrics_map.insert("recall".to_string(), 0.93);

    let metrics = crate::TrainingMetrics {
        loss: 0.125,
        accuracy: 0.92,
        learning_rate: 0.001,
        epoch: 42,
        latent_dim: 256,
        additional: metrics_map,
    };

    // Serialize and deserialize
    let json = serde_json::to_string(&metrics).unwrap();
    let restored: crate::TrainingMetrics = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.epoch, 42);
    assert_eq!(restored.latent_dim, 256);
    assert_eq!(restored.additional.len(), 2);
}
