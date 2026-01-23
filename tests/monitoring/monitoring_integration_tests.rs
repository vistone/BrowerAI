/// Integration tests for monitoring and metrics
use browerai_api_server::{create_app, metrics, AppState};
use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let state = Arc::new(AppState::new());
        let app = create_app(state);

        // Create test client
        let client = axum_test::TestServer::new(app).unwrap();

        // Test metrics endpoint
        let response = client.get("/api/metrics").await;
        
        assert_eq!(response.status_code(), 200);
        assert!(response.text().contains("browerai_"));
    }

    #[test]
    fn test_metrics_export() {
        // Record some sample metrics
        metrics::record_http_request("/api/health", "GET", 200);
        metrics::record_http_duration("/api/health", "GET", 0.123);
        metrics::record_cache_hit("selector_embedding");
        metrics::record_cache_miss("selector_embedding");
        metrics::update_cache_size(42);
        metrics::record_ai_inference("property_predictor", true, 0.456);

        // Export metrics
        let result = metrics::export_metrics();
        assert!(result.is_ok());

        let metrics_text = result.unwrap();
        
        // Verify key metrics are present
        assert!(metrics_text.contains("browerai_http_requests_total"));
        assert!(metrics_text.contains("browerai_http_request_duration_seconds"));
        assert!(metrics_text.contains("browerai_css_cache_hits_total"));
        assert!(metrics_text.contains("browerai_css_cache_misses_total"));
        assert!(metrics_text.contains("browerai_css_cache_size"));
        assert!(metrics_text.contains("browerai_ai_inference_total"));
        assert!(metrics_text.contains("browerai_ai_inference_duration_seconds"));
    }

    #[test]
    fn test_cache_metrics() {
        // Record cache operations
        for _ in 0..10 {
            metrics::record_cache_hit("selector_embedding");
        }
        
        for _ in 0..3 {
            metrics::record_cache_miss("selector_embedding");
        }

        metrics::update_cache_size(100);

        let metrics_text = metrics::export_metrics().unwrap();
        
        // Verify cache metrics
        assert!(metrics_text.contains("browerai_css_cache_hits_total"));
        assert!(metrics_text.contains("browerai_css_cache_misses_total"));
        assert!(metrics_text.contains("browerai_css_cache_size"));
    }

    #[test]
    fn test_ai_inference_metrics() {
        // Record successful inferences
        metrics::record_ai_inference("selector_embedding", true, 0.123);
        metrics::record_ai_inference("property_predictor", true, 0.456);

        // Record failed inference
        metrics::record_ai_inference("selector_embedding", false, 0.789);

        let metrics_text = metrics::export_metrics().unwrap();
        
        // Verify AI metrics
        assert!(metrics_text.contains("browerai_ai_inference_total"));
        assert!(metrics_text.contains("browerai_ai_inference_duration_seconds"));
        assert!(metrics_text.contains("selector_embedding"));
        assert!(metrics_text.contains("property_predictor"));
    }

    #[test]
    fn test_css_parsing_metrics() {
        // Record CSS parsing
        metrics::record_css_rules_parsed(5, false);
        metrics::record_css_rules_parsed(10, true);

        let metrics_text = metrics::export_metrics().unwrap();
        
        // Verify CSS parsing metrics
        assert!(metrics_text.contains("browerai_css_rules_parsed_total"));
    }

    #[test]
    fn test_http_metrics() {
        // Record various HTTP requests
        metrics::record_http_request("/api/health", "GET", 200);
        metrics::record_http_request("/api/version", "GET", 200);
        metrics::record_http_request("/api/v1/parse/css", "POST", 200);
        metrics::record_http_request("/api/v1/parse/css", "POST", 400);
        metrics::record_http_request("/api/v1/render", "POST", 500);

        // Record durations
        metrics::record_http_duration("/api/health", "GET", 0.001);
        metrics::record_http_duration("/api/v1/parse/css", "POST", 0.123);
        metrics::record_http_duration("/api/v1/render", "POST", 0.456);

        let metrics_text = metrics::export_metrics().unwrap();
        
        // Verify HTTP metrics
        assert!(metrics_text.contains("browerai_http_requests_total"));
        assert!(metrics_text.contains("browerai_http_request_duration_seconds"));
    }
}
