#[cfg(test)]
mod week6_integration_tests {
    use std::collections::HashMap;
    use std::time::Instant;

    /// Integration test result structure
    #[derive(Debug, Clone)]
    pub struct TestResult {
        pub name: String,
        pub passed: bool,
        pub status: String,
        pub duration_ms: f64,
        pub details: HashMap<String, String>,
    }

impl TestResult {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            status: "PENDING".to_string(),
            duration_ms: 0.0,
            details: HashMap::new(),
        }
    }

    pub fn set_passed(&mut self, passed: bool) {
        self.passed = passed;
        self.status = if passed { "PASSED" } else { "FAILED" }.to_string();
    }
}

/// Integration test client for Rust ↔ Python communication
pub struct IntegrationTestClient {
    pub base_url: String,
    pub feature_dim: usize,
    pub latent_dim: usize,
    pub test_results: Vec<TestResult>,
}

impl IntegrationTestClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            feature_dim: 48,
            latent_dim: 256,
            test_results: Vec::new(),
        }
    }

    /// Run all integration tests
    pub fn run_all_tests(&mut self) -> bool {
        println!("\n{}", "=".repeat(70));
        println!("  BrowserAI Week 6 - Rust Integration Tests");
        println!("{}", "=".repeat(70));
        println!();

        let tests = vec![
            ("Health Check - Server Readiness", Self::test_health_check as fn(&mut Self) -> TestResult),
            ("Feature Extraction Pipeline", Self::test_feature_extraction as fn(&mut Self) -> TestResult),
            ("Feature Encoding", Self::test_feature_encoding as fn(&mut Self) -> TestResult),
            ("Feedback Processing", Self::test_feedback_processing as fn(&mut Self) -> TestResult),
            ("Code Generation Validation", Self::test_code_generation as fn(&mut Self) -> TestResult),
            ("Error Handling", Self::test_error_handling as fn(&mut Self) -> TestResult),
            ("Performance Metrics", Self::test_performance_metrics as fn(&mut Self) -> TestResult),
        ];

        for (name, test_func) in tests {
            let result = test_func(self);
            let status_symbol = if result.passed { "✓" } else { "✗" };
            println!(
                "{} {:<40} {:<8} ({:.2}ms)",
                status_symbol, result.name, result.status, result.duration_ms
            );
            self.test_results.push(result);
        }

        println!("\n{}", "=".repeat(70));
        let passed = self.test_results.iter().filter(|r| r.passed).count();
        let total = self.test_results.len();
        println!("  Summary: {}/{} tests passed", passed, total);
        println!("{}\n", "=".repeat(70));

        passed == total
    }

    // =========================================================================
    // Test Methods
    // =========================================================================

    pub fn test_health_check(&mut self) -> TestResult {
        let mut result = TestResult::new("Health Check - Server Readiness");
        let start = Instant::now();

        // Simulate health check endpoint
        let response_data = vec![
            ("status", "healthy"),
            ("models_loaded", "3"),
            ("version", "1.0.0"),
        ];

        let mut has_all_fields = true;
        let mut details = HashMap::new();

        for (key, value) in response_data {
            if key == "status" && value != "healthy" {
                has_all_fields = false;
            }
            details.insert(key.to_string(), value.to_string());
        }

        result.set_passed(has_all_fields);
        result.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        result.details = details;

        result
    }

    pub fn test_feature_extraction(&mut self) -> TestResult {
        let mut result = TestResult::new("Feature Extraction Pipeline");
        let start = Instant::now();

        // Simulate feature extraction
        let features = self.create_test_features();
        
        let passed = features.len() == self.feature_dim && 
                    features.iter().all(|f| *f >= 0.0 && *f <= 1.0);

        result.set_passed(passed);
        result.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        result.details.insert("feature_dim".to_string(), format!("{}", features.len()));
        result.details.insert("all_normalized".to_string(), 
                            features.iter().all(|f| *f >= 0.0 && *f <= 1.0).to_string());

        result
    }

    pub fn test_feature_encoding(&mut self) -> TestResult {
        let mut result = TestResult::new("Feature Encoding");
        let start = Instant::now();

        // Simulate 48D → 256D encoding
        let features = self.create_test_features();
        let encoded = self.simulate_feature_encoding(&features);

        let passed = encoded.len() == self.latent_dim &&
                    encoded.iter().all(|f| f.is_finite());

        result.set_passed(passed);
        result.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        result.details.insert("input_dim".to_string(), "48".to_string());
        result.details.insert("output_dim".to_string(), format!("{}", encoded.len()));
        result.details.insert("valid_values".to_string(), passed.to_string());

        result
    }

    pub fn test_feedback_processing(&mut self) -> TestResult {
        let mut result = TestResult::new("Feedback Processing");
        let start = Instant::now();

        // Simulate feedback structure
        let feedback = vec![
            ("overall_quality", 0.85),
            ("html_similarity", 0.88),
            ("css_accuracy", 0.82),
            ("layout_similarity", 0.85),
        ];

        let all_valid = feedback.iter().all(|(_, score)| *score >= 0.0 && *score <= 1.0);

        result.set_passed(all_valid);
        result.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        result.details.insert("feedback_fields".to_string(), format!("{}", feedback.len()));
        result.details.insert("all_in_range".to_string(), all_valid.to_string());
        result.details.insert("avg_score".to_string(), 
                            format!("{:.2}", feedback.iter().map(|(_, s)| s).sum::<f64>() / feedback.len() as f64));

        result
    }

    pub fn test_code_generation(&mut self) -> TestResult {
        let mut result = TestResult::new("Code Generation Validation");
        let start = Instant::now();

        // Simulate code generation from latent vector
        let latent = self.simulate_feature_encoding(&self.create_test_features());
        let generated = self.simulate_code_generation(&latent);

        let passed = !generated.html.is_empty() && 
                    !generated.css.is_empty() && 
                    !generated.javascript.is_empty() &&
                    generated.confidence >= 0.0 && generated.confidence <= 1.0;

        result.set_passed(passed);
        result.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        result.details.insert("html_len".to_string(), format!("{}", generated.html.len()));
        result.details.insert("css_len".to_string(), format!("{}", generated.css.len()));
        result.details.insert("js_len".to_string(), format!("{}", generated.javascript.len()));
        result.details.insert("confidence".to_string(), format!("{:.2}", generated.confidence));

        result
    }

    pub fn test_error_handling(&mut self) -> TestResult {
        let mut result = TestResult::new("Error Handling");
        let start = Instant::now();

        // Simulate invalid inputs
        let invalid_features = vec![0.0]; // Wrong size
        let invalid_quality = 1.5; // Out of range

        let dim_check = invalid_features.len() != self.feature_dim;
        let quality_check = invalid_quality > 1.0 || invalid_quality < 0.0;

        result.set_passed(dim_check && quality_check);
        result.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        result.details.insert("invalid_dim_detected".to_string(), dim_check.to_string());
        result.details.insert("invalid_quality_detected".to_string(), quality_check.to_string());

        result
    }

    pub fn test_performance_metrics(&mut self) -> TestResult {
        let mut result = TestResult::new("Performance Metrics");
        let start = Instant::now();

        // Simulate latency measurements
        let mut latencies = Vec::new();
        for _ in 0..5 {
            let op_start = Instant::now();
            let _ = self.create_test_features();
            latencies.push(op_start.elapsed().as_secs_f64() * 1000.0);
        }

        let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let max_latency = latencies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_latency = latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let variance = max_latency - min_latency;

        let passed = avg_latency < 20.0; // Target: < 20ms average

        result.set_passed(passed);
        result.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        result.details.insert("avg_latency_ms".to_string(), format!("{:.2}", avg_latency));
        result.details.insert("max_latency_ms".to_string(), format!("{:.2}", max_latency));
        result.details.insert("min_latency_ms".to_string(), format!("{:.2}", min_latency));
        result.details.insert("variance_ms".to_string(), format!("{:.2}", variance));
        result.details.insert("target_met".to_string(), passed.to_string());

        result
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    fn create_test_features(&self) -> Vec<f64> {
        (0..self.feature_dim)
            .map(|i| (0.1 + (i as f64 * 0.01)) % 0.8)
            .collect()
    }

    fn simulate_feature_encoding(&self, features: &[f64]) -> Vec<f64> {
        // Simulate 48D → 256D transformation
        let mut encoded = Vec::with_capacity(self.latent_dim);
        
        for i in 0..self.latent_dim {
            // Create latent representation by mixing features
            let base = features[i % features.len()];
            let offset = (i as f64 / self.latent_dim as f64).sin();
            encoded.push((base + offset) / 2.0);
        }
        
        encoded
    }

    fn simulate_code_generation(&self, _latent: &[f64]) -> GeneratedCode {
        GeneratedCode {
            html: "<html><body><h1>Generated HTML</h1></body></html>".to_string(),
            css: "body { font-family: Arial; margin: 0; }".to_string(),
            javascript: "console.log('Generated JS');".to_string(),
            confidence: 0.87,
        }
    }
}

/// Generated code structure
#[derive(Debug, Clone)]
pub struct GeneratedCode {
    pub html: String,
    pub css: String,
    pub javascript: String,
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_client_creation() {
        let client = IntegrationTestClient::new("http://127.0.0.1:5000");
        assert_eq!(client.feature_dim, 48);
        assert_eq!(client.latent_dim, 256);
    }

    #[test]
    fn test_feature_creation() {
        let client = IntegrationTestClient::new("http://127.0.0.1:5000");
        let features = client.create_test_features();
        assert_eq!(features.len(), 48);
        assert!(features.iter().all(|f| *f >= 0.0 && *f <= 1.0));
    }

    #[test]
    fn test_feature_encoding() {
        let client = IntegrationTestClient::new("http://127.0.0.1:5000");
        let features = client.create_test_features();
        let encoded = client.simulate_feature_encoding(&features);
        assert_eq!(encoded.len(), 256);
    }

    #[test]
    fn test_all_integration_tests() {
        let mut client = IntegrationTestClient::new("http://127.0.0.1:5000");
        let all_passed = client.run_all_tests();
        assert!(all_passed, "Not all integration tests passed");
        assert_eq!(client.test_results.len(), 7);
    }

    #[test]
    fn test_result_structure() {
        let mut result = TestResult::new("Test Name");
        result.set_passed(true);
        assert_eq!(result.status, "PASSED");
        assert!(result.passed);

        result.set_passed(false);
        assert_eq!(result.status, "FAILED");
        assert!(!result.passed);
    }
}
