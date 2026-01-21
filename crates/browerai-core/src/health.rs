/// Unified health status and model state management for BrowerAI
///
/// This module provides centralized health status types that unify previously
/// duplicated definitions across different AI and model management modules.
use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents the health status of a model or service
///
/// This enum unifies health status definitions that were previously duplicated in:
/// - `browerai-ai-core/src/model_manager.rs` (ModelHealth)
/// - `browerai-ai-core/src/config.rs` (FallbackReason)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum HealthStatus {
    /// Entity is healthy and ready to use
    #[default]
    Ready,
    /// Entity is healthy but has warnings
    Warning(String),
    /// Entity is not available due to configuration
    Disabled,
    /// Required file or resource is missing
    MissingFile,
    /// Entity failed to load or initialize
    LoadFailed(String),
    /// Entity failed validation
    ValidationFailed(String),
    /// Entity is experiencing errors during operation
    OperationFailed(String),
    /// Model inference consistently fails
    InferenceFailing,
    /// Entity is unhealthy and needs attention
    Unhealthy(String),
    /// Health status is unknown
    Unknown,
}

impl HealthStatus {
    /// Check if the entity is in a usable state
    pub fn is_usable(&self) -> bool {
        matches!(self, HealthStatus::Ready | HealthStatus::Warning(_))
    }

    /// Check if the entity is in a failed state
    pub fn is_failed(&self) -> bool {
        matches!(
            self,
            HealthStatus::LoadFailed(_)
                | HealthStatus::ValidationFailed(_)
                | HealthStatus::Unhealthy(_)
        )
    }

    /// Get the error message if in a failed state
    pub fn error_message(&self) -> Option<&str> {
        match self {
            HealthStatus::LoadFailed(msg) => Some(msg),
            HealthStatus::ValidationFailed(msg) => Some(msg),
            HealthStatus::Unhealthy(msg) => Some(msg),
            HealthStatus::Warning(msg) => Some(msg),
            _ => None,
        }
    }
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HealthStatus::Ready => write!(f, "Ready"),
            HealthStatus::Warning(msg) => write!(f, "Warning: {}", msg),
            HealthStatus::Disabled => write!(f, "Disabled"),
            HealthStatus::MissingFile => write!(f, "Missing file"),
            HealthStatus::LoadFailed(msg) => write!(f, "Load failed: {}", msg),
            HealthStatus::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            HealthStatus::OperationFailed(msg) => write!(f, "Operation failed: {}", msg),
            HealthStatus::InferenceFailing => write!(f, "Inference failing"),
            HealthStatus::Unhealthy(msg) => write!(f, "Unhealthy: {}", msg),
            HealthStatus::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Reasons why an operation might fall back to baseline behavior
///
/// This enum captures the various reasons an AI-enhanced operation might
/// need to fall back to traditional/baseline methods.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FallbackReason {
    /// AI enhancement is disabled in configuration
    AiDisabled,
    /// Required model file was not found
    ModelNotFound(String),
    /// Model failed to load or initialize
    ModelLoadFailed(String),
    /// Model health check failed
    ModelUnhealthy(String),
    /// Inference operation failed with an error
    InferenceFailed(String),
    /// Inference exceeded the time limit
    TimeoutExceeded { actual_ms: u64, limit_ms: u64 },
    /// No suitable model is available for this operation
    NoModelAvailable,
    /// Model returned an invalid or unusable result
    InvalidResult,
    /// Resource constraints prevented AI operation
    ResourceConstraints(String),
    /// Unknown reason
    Unknown,
}

impl fmt::Display for FallbackReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FallbackReason::AiDisabled => write!(f, "AI disabled"),
            FallbackReason::ModelNotFound(name) => write!(f, "Model not found: {}", name),
            FallbackReason::ModelLoadFailed(err) => write!(f, "Model load failed: {}", err),
            FallbackReason::ModelUnhealthy(reason) => write!(f, "Model unhealthy: {}", reason),
            FallbackReason::InferenceFailed(err) => write!(f, "Inference failed: {}", err),
            FallbackReason::TimeoutExceeded {
                actual_ms,
                limit_ms,
            } => {
                write!(f, "Timeout exceeded: {}ms > {}ms", actual_ms, limit_ms)
            }
            FallbackReason::NoModelAvailable => write!(f, "No model available"),
            FallbackReason::InvalidResult => write!(f, "Invalid result from model"),
            FallbackReason::ResourceConstraints(msg) => {
                write!(f, "Resource constraints: {}", msg)
            }
            FallbackReason::Unknown => write!(f, "Unknown reason"),
        }
    }
}

/// Summary of health status for reporting
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HealthSummary {
    /// Total number of entities checked
    pub total: usize,
    /// Number of healthy entities
    pub healthy: usize,
    /// Number of entities with warnings
    pub warnings: usize,
    /// Number of failed entities
    pub failed: usize,
    /// Overall status
    #[serde(default)]
    pub overall: HealthStatus,
    /// Individual entity statuses
    #[serde(default)]
    pub details: Vec<(String, HealthStatus)>,
}

impl HealthSummary {
    /// Create a new empty health summary
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entity to the summary
    pub fn add_entity(&mut self, name: impl Into<String>, status: HealthStatus) {
        self.total += 1;
        match &status {
            HealthStatus::Ready => self.healthy += 1,
            HealthStatus::Warning(_) => self.warnings += 1,
            _ => self.failed += 1,
        }
        self.details.push((name.into(), status));
        self.update_overall();
    }

    /// Update the overall status based on current counts
    fn update_overall(&mut self) {
        self.overall = if self.failed > 0 {
            HealthStatus::Unhealthy(format!("{} of {} entities failed", self.failed, self.total))
        } else if self.warnings > 0 {
            HealthStatus::Warning(format!(
                "{} of {} entities have warnings",
                self.warnings, self.total
            ))
        } else {
            HealthStatus::Ready
        };
    }

    /// Calculate the health percentage (0.0 to 1.0)
    pub fn health_percentage(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.healthy as f64 / self.total as f64
        }
    }

    /// Check if overall health is acceptable
    pub fn is_healthy(&self) -> bool {
        self.failed == 0 && self.total > 0
    }
}

/// Severity level for issues
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum Severity {
    /// Informational only
    #[default]
    Info,
    /// Minor issue, doesn't affect functionality
    Low,
    /// Moderate issue, may affect performance
    Medium,
    /// Serious issue, affects functionality
    High,
    /// Critical issue, system may be unusable
    Critical,
}

/// A reported issue or problem
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Issue {
    /// Unique identifier for the issue
    #[serde(default)]
    pub id: String,
    /// Severity of the issue
    #[serde(default)]
    pub severity: Severity,
    /// Category of the issue
    #[serde(default)]
    pub category: String,
    /// Human-readable message
    #[serde(default)]
    pub message: String,
    /// Source location of the issue
    #[serde(default)]
    pub location: super::source_loc::SourceLocation,
    /// Suggested fix (if available)
    #[serde(default)]
    pub suggestion: String,
}

impl Issue {
    /// Create a new issue
    pub fn new(
        severity: Severity,
        category: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            id: format!("issue-{}", &uuid::Uuid::new_v4().to_string()[..8]),
            severity,
            category: category.into(),
            message: message.into(),
            location: super::source_loc::SourceLocation::default(),
            suggestion: String::new(),
        }
    }

    /// Create a critical issue
    pub fn critical(category: impl Into<String>, message: impl Into<String>) -> Self {
        let mut issue = Self::new(Severity::Critical, category, message);
        issue.suggestion = "Immediate action required".to_string();
        issue
    }

    /// Create an info issue
    pub fn info(category: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(Severity::Info, category, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_is_usable() {
        assert!(HealthStatus::Ready.is_usable());
        assert!(HealthStatus::Warning("test".to_string()).is_usable());
        assert!(!HealthStatus::MissingFile.is_usable());
        assert!(!HealthStatus::LoadFailed("error".to_string()).is_usable());
    }

    #[test]
    fn test_health_status_is_failed() {
        assert!(!HealthStatus::Ready.is_failed());
        assert!(HealthStatus::LoadFailed("error".to_string()).is_failed());
        assert!(HealthStatus::Unhealthy("error".to_string()).is_failed());
    }

    #[test]
    fn test_health_summary() {
        let mut summary = HealthSummary::new();
        summary.add_entity("model1".to_string(), HealthStatus::Ready);
        summary.add_entity(
            "model2".to_string(),
            HealthStatus::Warning("test".to_string()),
        );
        summary.add_entity("model3".to_string(), HealthStatus::MissingFile);

        assert_eq!(summary.total, 3);
        assert_eq!(summary.healthy, 1);
        assert_eq!(summary.warnings, 1);
        assert_eq!(summary.failed, 1);
        assert!(!summary.is_healthy());
        assert!((summary.health_percentage() - 0.333) < 0.01);
    }

    #[test]
    fn test_fallback_reason_display() {
        assert_eq!(FallbackReason::AiDisabled.to_string(), "AI disabled");
        assert_eq!(
            FallbackReason::ModelNotFound("test".to_string()).to_string(),
            "Model not found: test"
        );
    }
}
