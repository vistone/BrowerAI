/// Integration tests for ai-core module
///
/// Tests the complete lifecycle of model management, inference, and monitoring.
#[cfg(test)]
mod integration_tests {
    use browerai_ai_core::*;
    use std::path::PathBuf;

    #[test]
    fn test_model_provider_registry() {
        let registry = ModelProviderRegistry::new();

        // Create and register an ONNX provider
        let onnx_provider = std::sync::Arc::new(OnnxModelProvider::new());
        registry.register(onnx_provider.clone()).unwrap();

        // Verify provider is registered
        let providers = registry.list_providers();
        assert!(!providers.is_empty());
        assert_eq!(providers[0].name, "ONNX Runtime");
    }

    #[test]
    fn test_metrics_aggregator() {
        let aggregator = MetricsAggregator::new(100);

        // Record some metrics
        for i in 0..10 {
            let metric = InferenceMetrics {
                model_name: "test_model".to_string(),
                inference_time: std::time::Duration::from_millis(50 + i * 10),
                input_size: 1024,
                output_size: 1024,
                memory_peak_mb: 256,
                cache_hit: i % 2 == 0,
                success: true,
                timestamp: std::time::Instant::now(),
            };
            aggregator.record(metric);
        }

        // Get snapshot
        let snapshot = aggregator.snapshot();
        assert_eq!(snapshot.successful_inferences, 10);
        assert_eq!(snapshot.failed_inferences, 0);
        assert!(snapshot.avg_latency_ms > 0.0);
        assert_eq!(snapshot.success_rate, 1.0);
        assert_eq!(snapshot.cache_hit_rate, 0.5);
    }

    #[test]
    fn test_circuit_breaker_resilience() {
        let config = CircuitBreakerConfig {
            failure_threshold: 0.5,
            request_window: 10,
            timeout_duration: std::time::Duration::from_secs(1),
            enable_recovery: true,
        };

        let cb = CircuitBreaker::new(config);

        // Initially closed
        assert!(cb.allow_request());
        assert_eq!(cb.current_state(), CircuitState::Closed);

        // Simulate failures
        for _ in 0..6 {
            cb.record_failure();
        }

        // Should be open now (60% failure rate > 50% threshold)
        assert_eq!(cb.current_state(), CircuitState::Open);
        assert!(!cb.allow_request());

        // Test recovery
        cb.reset();
        assert_eq!(cb.current_state(), CircuitState::Closed);
        assert!(cb.allow_request());
    }

    #[test]
    fn test_fallback_tracker() {
        let tracker = FallbackTracker::new(100);

        // Record some operations
        tracker.record_attempt();
        tracker.record_success(50);

        tracker.record_attempt();
        tracker.record_fallback(FallbackReason::ModelNotFound("model.onnx".to_string()));

        tracker.record_attempt();
        tracker.record_success(60);

        // Check stats
        let stats = tracker.get_stats();
        assert_eq!(stats.total_attempts, 3);
        assert_eq!(stats.successful, 2);
        assert_eq!(stats.fallback_count, 1);
        assert!(stats.success_rate() > 0.6);
    }

    #[test]
    fn test_model_load_config_builder() {
        let config = ModelLoadConfig::new(PathBuf::from("model.onnx"))
            .with_gpu(true)
            .with_option("precision", "fp16")
            .with_option("device", "cuda:0")
            .with_warmup(true);

        assert!(config.use_gpu);
        assert!(config.warmup);
        assert_eq!(config.options.get("precision"), Some(&"fp16".to_string()));
        assert_eq!(config.options.get("device"), Some(&"cuda:0".to_string()));
    }

    #[test]
    fn test_retry_policy_with_exponential_backoff() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_backoff: std::time::Duration::from_millis(10),
            max_backoff: std::time::Duration::from_millis(100),
            backoff_multiplier: 2.0,
        };

        let policy = RetryPolicy::new(config);
        let mut attempt_count = 0;

        let start = std::time::Instant::now();
        let result = policy.execute(|| {
            attempt_count += 1;
            if attempt_count < 3 {
                Err(anyhow::anyhow!("Temporary error"))
            } else {
                Ok(42)
            }
        });

        assert!(result.is_ok());
        assert_eq!(attempt_count, 3);
        // Should have taken at least 30ms (10 + 20)
        assert!(start.elapsed().as_millis() >= 30);
    }

    #[test]
    fn test_ai_runtime_initialization() {
        let engine = InferenceEngine::new().unwrap();
        let runtime = AiRuntime::new(engine);

        assert!(runtime.is_ai_enabled());
        assert!(runtime.engine().monitor_handle().is_some());
    }

    #[test]
    fn test_provider_capabilities() {
        let provider = OnnxModelProvider::new().with_gpu(true);
        let info = provider.info();

        assert_eq!(info.name, "ONNX Runtime");
        assert!(info.capabilities.gpu);
        assert!(info.capabilities.batch_inference);
        assert!(info.capabilities.quantization);
        assert_eq!(info.capabilities.max_model_size_mb, Some(4096));
    }

    #[test]
    fn test_onnx_provider_format_detection() {
        let provider = OnnxModelProvider::new();

        // Should handle ONNX files
        assert!(provider.can_load(std::path::Path::new("model.onnx")));
        assert!(provider.can_load(std::path::Path::new("path/to/model.ONNX")));

        // Should not handle other formats
        assert!(!provider.can_load(std::path::Path::new("model.pth")));
        assert!(!provider.can_load(std::path::Path::new("model.pb")));
        assert!(!provider.can_load(std::path::Path::new("model")));
    }

    #[test]
    fn test_histogram_percentiles() {
        let mut histogram = HistogramBucket::default();

        // Record 100 measurements: 0-99
        for i in 0..100 {
            histogram.record(i as f64);
        }

        assert_eq!(histogram.count(), 100);
        assert_eq!(histogram.min(), Some(0.0));
        assert_eq!(histogram.max(), Some(99.0));

        // Check percentiles are in expected range
        if let Some(p50) = histogram.percentile(0.5) {
            assert!((40.0..=60.0).contains(&p50));
        }

        if let Some(p95) = histogram.percentile(0.95) {
            assert!((85.0..=99.0).contains(&p95));
        }
    }

    #[test]
    fn test_model_metadata_serialization() {
        use std::collections::HashMap;

        let inputs = vec![TensorInfo {
            name: "image".to_string(),
            dtype: "float32".to_string(),
            shape: vec![1, 224, 224, 3],
        }];

        let outputs = vec![TensorInfo {
            name: "logits".to_string(),
            dtype: "float32".to_string(),
            shape: vec![1, 1000],
        }];

        let mut props = HashMap::new();
        props.insert("framework".to_string(), "PyTorch".to_string());

        let metadata = ModelMetadata {
            name: "ResNet50".to_string(),
            version: "2.0.1".to_string(),
            inputs,
            outputs,
            properties: props,
            framework: "ONNX".to_string(),
        };

        // Should be serializable
        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: ModelMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, "ResNet50");
        assert_eq!(deserialized.version, "2.0.1");
        assert_eq!(deserialized.inputs.len(), 1);
        assert_eq!(deserialized.outputs.len(), 1);
    }

    #[test]
    fn test_inference_metrics_calculations() {
        let metrics = InferenceMetrics {
            model_name: "test".to_string(),
            inference_time: std::time::Duration::from_millis(100),
            input_size: 1024,
            output_size: 2048,
            memory_peak_mb: 512,
            cache_hit: false,
            success: true,
            timestamp: std::time::Instant::now(),
        };

        assert_eq!(metrics.latency_ms(), 100.0);
        assert_eq!(metrics.throughput(), 10.0);

        // Throughput should be output bytes per second
        let mbps = metrics.effective_throughput_mbps();
        assert!(mbps > 0.0);
    }
}
