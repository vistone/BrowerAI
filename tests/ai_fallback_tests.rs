/// Integration tests for AI fallback scenarios
/// Tests M1 milestone requirements for AI-Centric Execution Refresh

#[cfg(test)]
mod tests {
    use browerai::{AiConfig, CssParser, FallbackReason, FallbackTracker, HtmlParser, JsParser};

    #[test]
    fn test_html_parser_no_ai_runtime() {
        // Test baseline parser when no AI runtime is available
        let parser = HtmlParser::new();
        let result = parser.parse("<html><body>Hello</body></html>");
        assert!(result.is_ok());
    }

    #[test]
    fn test_css_parser_no_ai_runtime() {
        // Test baseline parser when no AI runtime is available
        let parser = CssParser::new();
        let result = parser.parse("body { color: red; }");
        assert!(result.is_ok());
        assert!(result.unwrap().len() > 0);
    }

    #[test]
    fn test_js_parser_no_ai_runtime() {
        // Test baseline parser when no AI runtime is available
        let parser = JsParser::new();
        let result = parser.parse("console.log('hello');");
        assert!(result.is_ok());
    }

    #[test]
    fn test_fallback_tracker_basic_flow() {
        // Test fallback tracker records attempts, successes, and fallbacks
        let tracker = FallbackTracker::new(10);

        // Record some attempts
        tracker.record_attempt();
        tracker.record_success(10);

        tracker.record_attempt();
        tracker.record_fallback(FallbackReason::ModelNotFound("test.onnx".to_string()));

        tracker.record_attempt();
        tracker.record_success(15);

        let stats = tracker.get_stats();
        assert_eq!(stats.total_attempts, 3);
        assert_eq!(stats.successful, 2);
        assert_eq!(stats.fallback_count, 1);
        assert_eq!(stats.success_rate(), 2.0 / 3.0);
        assert_eq!(stats.fallback_rate(), 1.0 / 3.0);
        assert_eq!(stats.avg_inference_ms(), 12.5);
    }

    #[test]
    fn test_fallback_reason_display_variants() {
        // Test all fallback reason variants
        let reasons = vec![
            FallbackReason::AiDisabled,
            FallbackReason::ModelNotFound("model.onnx".to_string()),
            FallbackReason::ModelLoadFailed("load error".to_string()),
            FallbackReason::InferenceFailed("inference error".to_string()),
            FallbackReason::TimeoutExceeded {
                actual_ms: 150,
                limit_ms: 100,
            },
            FallbackReason::ModelUnhealthy("unhealthy".to_string()),
            FallbackReason::NoModelAvailable,
        ];

        for reason in reasons {
            let display = format!("{}", reason);
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_ai_config_custom() {
        // Test custom AI configuration
        let config = AiConfig {
            enable_ai: false,
            enable_fallback: false,
            enable_logging: false,
            max_inference_time_ms: 50,
        };

        assert!(!config.enable_ai);
        assert!(!config.enable_fallback);
        assert!(!config.enable_logging);
        assert_eq!(config.max_inference_time_ms, 50);
    }

    #[test]
    fn test_fallback_tracker_clear() {
        // Test clearing tracker statistics
        let tracker = FallbackTracker::new(10);
        tracker.record_attempt();
        tracker.record_success(10);
        tracker.record_fallback(FallbackReason::AiDisabled);

        tracker.clear();

        let stats = tracker.get_stats();
        assert_eq!(stats.total_attempts, 0);
        assert_eq!(stats.successful, 0);
        assert_eq!(stats.fallback_count, 0);

        let fallbacks = tracker.get_recent_fallbacks();
        assert_eq!(fallbacks.len(), 0);
    }

    #[test]
    fn test_fallback_tracker_recent_limit() {
        // Test that tracker limits recent fallback history
        let tracker = FallbackTracker::new(5);

        // Record more fallbacks than the limit
        for i in 0..10 {
            tracker.record_fallback(FallbackReason::ModelNotFound(format!("model{}", i)));
        }

        let fallbacks = tracker.get_recent_fallbacks();
        assert_eq!(fallbacks.len(), 5);

        // Should keep the last 5
        let last_reason = &fallbacks[4].1;
        match last_reason {
            FallbackReason::ModelNotFound(name) => assert_eq!(name, "model9"),
            _ => panic!("Expected ModelNotFound"),
        }
    }

    #[test]
    fn test_ai_stats_zero_division() {
        // Test that stats calculations handle zero attempts gracefully
        let stats = browerai::AiStats::default();
        assert_eq!(stats.success_rate(), 0.0);
        assert_eq!(stats.fallback_rate(), 0.0);
        assert_eq!(stats.avg_inference_ms(), 0.0);
    }
}
