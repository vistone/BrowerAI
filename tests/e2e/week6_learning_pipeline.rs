/// End-to-End Integration Tests for Week 6 Learning Pipeline
/// Tests the complete flow: URL → Feature Extraction → Python API → Feedback

use browerai_learning::{
    WebsiteFeatureExtractor, RustPythonBridge, FeedbackCollector,
    FeaturePacket, RenderingComparison, data_models::PageContent,
    learning_sandbox::WebsiteIntent,
};
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test feature extraction from PageContent
    #[test]
    fn test_feature_extraction_e2e() {
        // Create a sample page
        let mut page = PageContent {
            url: "https://example.com".to_string(),
            base_url: "https://example.com".to_string(),
            html: r#"
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
            dom: None,
            inline_css: vec![],
            inline_js: vec![],
            external_resources: vec![],
        };

        let intent = WebsiteIntent {
            website_type: "blog".to_string(),
            primary_color: "#3498db".to_string(),
            design_style: "modern".to_string(),
            complexity_level: browerai_learning::learning_sandbox::intent_analyzer::ComplexityLevel::Moderate,
            structure: browerai_learning::learning_sandbox::intent_analyzer::PageStructure {
                has_main_content: true,
                layout_type: browerai_learning::learning_sandbox::intent_analyzer::LayoutType::TwoColumn,
            },
            design: browerai_learning::learning_sandbox::intent_analyzer::DesignStyle {
                formality_score: 0.7,
                color_count: 5,
                uses_gradients: true,
            },
        };

        // Extract features
        let features = WebsiteFeatureExtractor::extract(&page, &intent).unwrap();

        // Verify
        assert_eq!(features.len(), 48);
        assert!(features.iter().all(|f| f.is_finite() && *f >= 0.0));
    }

    /// Test feature packet serialization for API communication
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

    /// Test feedback collector workflow
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

    /// Test bridge initialization
    #[test]
    fn test_rust_python_bridge_creation() {
        let bridge = RustPythonBridge::new("http://localhost:5000".to_string());

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

    /// Test complete learning loop workflow (without actual Python connection)
    #[test]
    fn test_complete_learning_loop_workflow() {
        // Step 1: Extract features from website
        let page = PageContent {
            url: "https://example.com".to_string(),
            base_url: "https://example.com".to_string(),
            html: r#"
                <html>
                <head><title>Blog</title></head>
                <body>
                    <header>Blog Header</header>
                    <main><article><h1>Post</h1><p>Content</p></article></main>
                    <footer>Footer</footer>
                </body>
                </html>
            "#.to_string(),
            dom: None,
            inline_css: vec![],
            inline_js: vec![],
            external_resources: vec![],
        };

        let intent = WebsiteIntent {
            website_type: "blog".to_string(),
            primary_color: "#333".to_string(),
            design_style: "minimal".to_string(),
            complexity_level: browerai_learning::learning_sandbox::intent_analyzer::ComplexityLevel::Simple,
            structure: browerai_learning::learning_sandbox::intent_analyzer::PageStructure {
                has_main_content: true,
                layout_type: browerai_learning::learning_sandbox::intent_analyzer::LayoutType::SingleColumn,
            },
            design: browerai_learning::learning_sandbox::intent_analyzer::DesignStyle {
                formality_score: 0.5,
                color_count: 3,
                uses_gradients: false,
            },
        };

        // Extract features
        let features = WebsiteFeatureExtractor::extract(&page, &intent).unwrap();
        assert_eq!(features.len(), 48);

        // Step 2: Create feature packet for Python
        let packet = FeaturePacket {
            url: page.url.clone(),
            features: features.clone(),
            website_intent: intent.website_type.clone(),
            design_style: intent.design_style.clone(),
            feedback: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
            session_id: format!("session-{}", uuid::Uuid::new_v4()),
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

    /// Test training metrics structure
    #[test]
    fn test_training_metrics_workflow() {
        let mut metrics_map = HashMap::new();
        metrics_map.insert("precision".to_string(), 0.91);
        metrics_map.insert("recall".to_string(), 0.93);

        let metrics = browerai_learning::TrainingMetrics {
            loss: 0.125,
            accuracy: 0.92,
            learning_rate: 0.001,
            epoch: 42,
            latent_dim: 256,
            additional: metrics_map,
        };

        // Serialize and deserialize
        let json = serde_json::to_string(&metrics).unwrap();
        let restored: browerai_learning::TrainingMetrics = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.epoch, 42);
        assert_eq!(restored.latent_dim, 256);
        assert_eq!(restored.additional.len(), 2);
    }
}

// Helper module for UUID generation in tests
mod uuid {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    pub struct Uuid;
    
    impl Uuid {
        pub fn new_v4() -> String {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .subsec_nanos();
            
            format!("{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                nanos,
                (nanos >> 16) & 0xffff,
                (nanos >> 32) & 0xffff,
                (nanos >> 48) & 0xffff,
                nanos)
        }
    }
}
