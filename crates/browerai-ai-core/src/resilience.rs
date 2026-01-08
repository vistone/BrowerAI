/// Resilience and High Availability Patterns
///
/// This module provides patterns for building reliable inference systems:
/// - Circuit breaker for failure detection
/// - Retry with exponential backoff
/// - Fallback strategies
/// - Health checks and recovery
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Normal operation
    Closed,
    /// Failures detected, requests failing fast
    Open,
    /// Testing if service recovered
    HalfOpen,
}

/// Configuration for circuit breaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure rate threshold (0.0 to 1.0)
    pub failure_threshold: f64,
    /// Number of requests before state decision
    pub request_window: usize,
    /// Time to wait before trying to recover
    pub timeout_duration: Duration,
    /// Enable automatic recovery attempts
    pub enable_recovery: bool,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 0.5, // 50% failure rate
            request_window: 10,     // Track last 10 requests
            timeout_duration: Duration::from_secs(30),
            enable_recovery: true,
        }
    }
}

/// Circuit breaker implementation
#[derive(Clone)]
pub struct CircuitBreaker {
    config: Arc<CircuitBreakerConfig>,
    state: Arc<RwLock<CircuitState>>,
    failures: Arc<Mutex<Vec<bool>>>, // true = success, false = failure
    last_state_change: Arc<Mutex<Instant>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config: Arc::new(config),
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failures: Arc::new(Mutex::new(Vec::new())),
            last_state_change: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Check if request should be allowed
    pub fn allow_request(&self) -> bool {
        if let Ok(state) = self.state.read() {
            match *state {
                CircuitState::Closed => true,
                CircuitState::Open => {
                    if self.config.enable_recovery {
                        if let Ok(last_change) = self.last_state_change.lock() {
                            let elapsed = last_change.elapsed();
                            elapsed >= self.config.timeout_duration
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                CircuitState::HalfOpen => true, // Allow test request
            }
        } else {
            false
        }
    }

    /// Record a success
    pub fn record_success(&self) {
        if let Ok(mut failures) = self.failures.lock() {
            failures.push(true);
            if failures.len() > self.config.request_window {
                failures.remove(0);
            }

            // Clone snapshot to avoid nested lock in evaluation
            let snapshot = failures.clone();
            drop(failures);
            self.evaluate_state(&snapshot);
        }
    }

    /// Record a failure
    pub fn record_failure(&self) {
        if let Ok(mut failures) = self.failures.lock() {
            failures.push(false);
            if failures.len() > self.config.request_window {
                failures.remove(0);
            }

            // Clone snapshot to avoid nested lock in evaluation
            let snapshot = failures.clone();
            drop(failures);
            self.evaluate_state(&snapshot);
        }
    }

    /// Evaluate and update circuit state
    fn evaluate_state(&self, failures: &[bool]) {
        if failures.is_empty() {
            return;
        }

        let failure_count = failures.iter().filter(|&&f| !f).count();
        let failure_rate = failure_count as f64 / failures.len() as f64;

        if let Ok(mut state) = self.state.write() {
            let new_state = if failure_rate >= self.config.failure_threshold {
                CircuitState::Open
            } else if failure_rate < (self.config.failure_threshold * 0.5) {
                CircuitState::Closed
            } else if *state == CircuitState::Open && failure_rate < 0.1 {
                CircuitState::HalfOpen
            } else {
                *state
            };

            if new_state != *state {
                *state = new_state;
                if let Ok(mut last_change) = self.last_state_change.lock() {
                    *last_change = Instant::now();
                }
            }
        }
    }

    /// Get current state
    pub fn current_state(&self) -> CircuitState {
        if let Ok(state) = self.state.read() {
            *state
        } else {
            CircuitState::Closed
        }
    }

    /// Reset circuit breaker
    pub fn reset(&self) {
        if let Ok(mut state) = self.state.write() {
            *state = CircuitState::Closed;
        }
        if let Ok(mut failures) = self.failures.lock() {
            failures.clear();
        }
        if let Ok(mut last_change) = self.last_state_change.lock() {
            *last_change = Instant::now();
        }
    }
}

impl std::fmt::Debug for CircuitBreaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitBreaker")
            .field("state", &self.current_state())
            .field("config", &self.config)
            .finish()
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of attempts
    pub max_attempts: u32,
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Backoff multiplier (exponential)
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
            backoff_multiplier: 2.0,
        }
    }
}

/// Retry with exponential backoff
pub struct RetryPolicy {
    config: Arc<RetryConfig>,
}

impl RetryPolicy {
    /// Create a new retry policy
    pub fn new(config: RetryConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }

    /// Execute a function with retry
    pub fn execute<F, T>(&self, mut f: F) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        let mut attempt = 0;
        let mut backoff = self.config.initial_backoff;

        loop {
            match f() {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempt += 1;
                    if attempt >= self.config.max_attempts {
                        return Err(e);
                    }

                    std::thread::sleep(backoff);
                    backoff = Duration::from_secs_f64(
                        (backoff.as_secs_f64() * self.config.backoff_multiplier)
                            .min(self.config.max_backoff.as_secs_f64()),
                    );
                }
            }
        }
    }
}

/// Fallback strategy options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackStrategy {
    /// No fallback
    None,
    /// Use a cached result if available
    CachedResult,
    /// Use a default/dummy result
    DefaultResult,
    /// Use alternative model
    AlternativeModel(String),
    /// Delegate to fallback service
    FallbackService(String),
}

/// Health check interface
pub trait HealthCheck: Send + Sync {
    /// Run health check
    fn check(&self) -> bool;

    /// Get health status details
    fn details(&self) -> HealthDetails;
}

/// Health check details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthDetails {
    /// Is healthy
    pub healthy: bool,
    /// Status message
    pub message: String,
    /// Metrics
    pub metrics: HealthMetrics,
}

/// Health metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// Request count
    pub request_count: u64,
    /// Error count
    pub error_count: u64,
    /// Average latency in ms
    pub avg_latency_ms: f64,
    /// Memory usage in MB
    pub memory_mb: u64,
}

/// Resilient inference wrapper
pub struct ResilientInference {
    circuit_breaker: CircuitBreaker,
    #[allow(dead_code)]
    retry_policy: RetryPolicy,
    fallback_strategy: FallbackStrategy,
}

impl ResilientInference {
    /// Create a new resilient inference wrapper
    pub fn new(
        cb_config: CircuitBreakerConfig,
        retry_config: RetryConfig,
        fallback: FallbackStrategy,
    ) -> Self {
        Self {
            circuit_breaker: CircuitBreaker::new(cb_config),
            retry_policy: RetryPolicy::new(retry_config),
            fallback_strategy: fallback,
        }
    }

    /// Get circuit breaker
    pub fn circuit_breaker(&self) -> &CircuitBreaker {
        &self.circuit_breaker
    }

    /// Get fallback strategy
    pub fn fallback_strategy(&self) -> &FallbackStrategy {
        &self.fallback_strategy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_closed() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert!(cb.allow_request());
        assert_eq!(cb.current_state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 0.5,
            request_window: 4,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Record 2 failures and 2 successes
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        cb.record_success();

        // Failure rate is 50%, should trigger open
        assert_eq!(cb.current_state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.current_state(), CircuitState::Open);

        cb.reset();
        assert_eq!(cb.current_state(), CircuitState::Closed);
    }

    #[test]
    fn test_retry_policy() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(100),
            backoff_multiplier: 2.0,
        };

        let policy = RetryPolicy::new(config);
        let mut attempt_count = 0;

        let result = policy.execute(|| {
            attempt_count += 1;
            if attempt_count < 3 {
                Err(anyhow::anyhow!("Temporary failure"))
            } else {
                Ok("success")
            }
        });

        assert!(result.is_ok());
        assert_eq!(attempt_count, 3);
    }
}
