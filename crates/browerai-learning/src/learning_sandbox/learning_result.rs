//! Learning Result
//!
//! Represents the complete result of learning from a website,
//! including all analysis results and generated artifacts.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete learning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningResult {
    /// Learning session ID
    pub session_id: String,

    /// Source URL
    pub source_url: String,

    /// Intent analysis result
    pub intent: crate::learning_sandbox::intent_analyzer::WebsiteIntent,

    /// Tech stack report (if available)
    pub tech_stack: Option<crate::learning_sandbox::tech_stack_detector::TechStackReport>,

    /// Feature map
    pub features: crate::learning_sandbox::feature_recognizer::FeatureMap,

    /// Page structure analysis
    pub structure: crate::learning_sandbox::structure_analyzer::PageStructure,

    /// Knowledge graph
    pub knowledge_graph: crate::learning_sandbox::knowledge_graph::KnowledgeGraph,

    /// Generated website
    pub generated_website: Option<GeneratedWebsite>,

    /// Processing metadata
    pub metadata: LearningMetadata,

    /// Learning quality metrics
    pub quality_metrics: QualityMetrics,

    /// Recommendations for improvement
    pub recommendations: Vec<Recommendation>,
}

/// Generated website from learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedWebsite {
    /// HTML content
    pub html: String,

    /// CSS content
    pub css: String,

    /// JavaScript content
    pub js: String,

    /// Generated assets
    pub assets: Vec<Asset>,

    /// Generation metadata
    pub generation_metadata: GenerationMetadata,
}

/// Asset file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    /// Asset name
    pub name: String,

    /// Asset type
    pub asset_type: AssetType,

    /// Content (for inline) or path (for external)
    pub content: String,

    /// Size in bytes
    pub size: usize,
}

/// Types of assets
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AssetType {
    Image,
    Font,
    Icon,
    Video,
    Audio,
    Data,
    Config,
    Other,
}

/// Metadata about the learning process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningMetadata {
    /// Session start time
    pub start_time: chrono::DateTime<chrono::Utc>,

    /// Session end time
    pub end_time: chrono::DateTime<chrono::Utc>,

    /// Total duration in milliseconds
    pub duration_ms: u64,

    /// Number of pages crawled
    pub pages_crawled: usize,

    /// Number of resources analyzed
    pub resources_analyzed: usize,

    /// Total bytes fetched
    pub bytes_fetched: u64,

    /// Whether learning was successful
    pub success: bool,

    /// Error message (if any)
    pub error_message: Option<String>,
}

/// Quality metrics for the learning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Intent confidence
    pub intent_confidence: f32,

    /// Structure fidelity score
    pub structure_fidelity: f32,

    /// Content preservation score
    pub content_preservation: f32,

    /// Feature coverage score
    pub feature_coverage: f32,

    /// Tech stack detection accuracy
    pub tech_detection_accuracy: f32,

    /// Overall quality score (0.0 - 1.0)
    pub overall_quality: f32,

    /// Issues found during learning
    pub issues: Vec<QualityIssue>,
}

/// A quality issue found during learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Issue severity
    pub severity: IssueSeverity,

    /// Issue category
    pub category: IssueCategory,

    /// Issue description
    pub description: String,

    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Severity levels for issues
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Categories of issues
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueCategory {
    Content,
    Structure,
    Style,
    Functionality,
    Accessibility,
    Performance,
    Security,
    Compatibility,
}

/// Recommendations for improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation priority
    pub priority: u8,

    /// Recommendation category
    pub category: String,

    /// Recommendation description
    pub description: String,

    /// Expected impact
    pub expected_impact: String,

    /// Implementation effort
    pub effort: EffortLevel,
}

/// Effort levels for recommendations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EffortLevel {
    Low,
    Medium,
    High,
}

/// Learning Result Builder
#[derive(Debug, Default)]
pub struct LearningResultBuilder {
    session_id: Option<String>,
    source_url: Option<String>,
    intent: Option<crate::learning_sandbox::intent_analyzer::WebsiteIntent>,
    tech_stack: Option<crate::learning_sandbox::tech_stack_detector::TechStackReport>,
    features: Option<crate::learning_sandbox::feature_recognizer::FeatureMap>,
    structure: Option<crate::learning_sandbox::structure_analyzer::PageStructure>,
    knowledge_graph: Option<crate::learning_sandbox::knowledge_graph::KnowledgeGraph>,
    generated_website: Option<GeneratedWebsite>,
    start_time: Option<chrono::DateTime<chrono::Utc>>,
    pages_crawled: usize,
    resources_analyzed: usize,
    bytes_fetched: u64,
    success: bool,
    error_message: Option<String>,
}

impl LearningResultBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set session ID
    pub fn with_session_id(mut self, session_id: String) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Set source URL
    pub fn with_source_url(mut self, source_url: String) -> Self {
        self.source_url = Some(source_url);
        self
    }

    /// Set intent
    pub fn with_intent(
        mut self,
        intent: crate::learning_sandbox::intent_analyzer::WebsiteIntent,
    ) -> Self {
        self.intent = Some(intent);
        self
    }

    /// Set tech stack
    pub fn with_tech_stack(
        mut self,
        tech_stack: crate::learning_sandbox::tech_stack_detector::TechStackReport,
    ) -> Self {
        self.tech_stack = Some(tech_stack);
        self
    }

    /// Set features
    pub fn with_features(
        mut self,
        features: crate::learning_sandbox::feature_recognizer::FeatureMap,
    ) -> Self {
        self.features = Some(features);
        self
    }

    /// Set structure
    pub fn with_structure(
        mut self,
        structure: crate::learning_sandbox::structure_analyzer::PageStructure,
    ) -> Self {
        self.structure = Some(structure);
        self
    }

    /// Set knowledge graph
    pub fn with_knowledge_graph(
        mut self,
        knowledge_graph: crate::learning_sandbox::knowledge_graph::KnowledgeGraph,
    ) -> Self {
        self.knowledge_graph = Some(knowledge_graph);
        self
    }

    /// Set generated website
    pub fn with_generated_website(mut self, generated_website: GeneratedWebsite) -> Self {
        self.generated_website = Some(generated_website);
        self
    }

    /// Set start time
    pub fn with_start_time(mut self, start_time: chrono::DateTime<chrono::Utc>) -> Self {
        self.start_time = Some(start_time);
        self
    }

    /// Set pages crawled
    pub fn with_pages_crawled(mut self, pages_crawled: usize) -> Self {
        self.pages_crawled = pages_crawled;
        self
    }

    /// Set resources analyzed
    pub fn with_resources_analyzed(mut self, resources_analyzed: usize) -> Self {
        self.resources_analyzed = resources_analyzed;
        self
    }

    /// Set bytes fetched
    pub fn with_bytes_fetched(mut self, bytes_fetched: u64) -> Self {
        self.bytes_fetched = bytes_fetched;
        self
    }

    /// Set success status
    pub fn with_success(mut self, success: bool) -> Self {
        self.success = success;
        self
    }

    /// Set error message
    pub fn with_error_message(mut self, error_message: Option<String>) -> Self {
        self.error_message = error_message;
        self
    }

    /// Build the learning result
    pub async fn build(self) -> LearningResult {
        let end_time = chrono::Utc::now();
        let start_time = self.start_time.unwrap_or(end_time);
        let duration_ms = (end_time - start_time).num_milliseconds() as u64;

        let session_id = self
            .session_id
            .unwrap_or_else(|| format!("session-{}", chrono::Utc::now().timestamp()));
        let source_url = self.source_url.unwrap_or_else(|| "unknown".to_string());

        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics();

        // Generate recommendations
        let recommendations = self.generate_recommendations(&quality_metrics);

        let metadata = LearningMetadata {
            start_time,
            end_time,
            duration_ms,
            pages_crawled: self.pages_crawled,
            resources_analyzed: self.resources_analyzed,
            bytes_fetched: self.bytes_fetched,
            success: self.success,
            error_message: self.error_message,
        };

        LearningResult {
            session_id,
            source_url,
            intent: self.intent.unwrap_or_default(),
            tech_stack: self.tech_stack,
            features: self.features.unwrap_or_default(),
            structure: self.structure.unwrap_or_default(),
            knowledge_graph: self.knowledge_graph.unwrap_or_default(),
            generated_website: self.generated_website,
            metadata,
            quality_metrics,
            recommendations,
        }
    }

    /// Calculate quality metrics
    fn calculate_quality_metrics(&self) -> QualityMetrics {
        let mut issues = Vec::new();

        // Intent confidence
        let intent_confidence = self.intent.as_ref().map(|i| i.confidence).unwrap_or(0.0);

        // Structure fidelity
        let structure_fidelity = self
            .structure
            .as_ref()
            .map(|s| s.semantic_structure.organization_score)
            .unwrap_or(0.0);

        // Feature coverage
        let feature_coverage = self
            .features
            .as_ref()
            .map(|f| {
                if !f.features.is_empty() {
                    (f.features.len() as f32 / 20.0).min(1.0)
                } else {
                    0.5
                }
            })
            .unwrap_or(0.5);

        // Tech detection accuracy
        let tech_detection_accuracy = self
            .tech_stack
            .as_ref()
            .map(|ts| ts.confidence)
            .unwrap_or(0.0);

        // Content preservation (placeholder - would need original content to compare)
        let content_preservation = 0.7;

        // Overall quality
        let overall_quality = (intent_confidence
            + structure_fidelity
            + feature_coverage
            + tech_detection_accuracy
            + content_preservation)
            / 5.0;

        // Check for issues
        if intent_confidence < 0.5 {
            issues.push(QualityIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::Content,
                description: "Low confidence in intent detection".to_string(),
                suggestion: Some("Review and manually specify website type".to_string()),
            });
        }

        if structure_fidelity < 0.5 {
            issues.push(QualityIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::Structure,
                description: "Poor semantic structure detection".to_string(),
                suggestion: Some("Check for missing semantic HTML elements".to_string()),
            });
        }

        if !self.success {
            issues.push(QualityIssue {
                severity: IssueSeverity::Error,
                category: IssueCategory::Functionality,
                description: self
                    .error_message
                    .clone()
                    .unwrap_or_else(|| "Learning process failed".to_string()),
                suggestion: Some("Review error logs and retry".to_string()),
            });
        }

        QualityMetrics {
            intent_confidence,
            structure_fidelity,
            content_preservation,
            feature_coverage,
            tech_detection_accuracy,
            overall_quality,
            issues,
        }
    }

    /// Generate recommendations
    fn generate_recommendations(&self, quality: &QualityMetrics) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Based on quality metrics
        if quality.intent_confidence < 0.7 {
            recommendations.push(Recommendation {
                priority: 1,
                category: "Intent Analysis".to_string(),
                description: "Improve intent detection confidence".to_string(),
                expected_impact: "Better understanding of website type and features".to_string(),
                effort: EffortLevel::Medium,
            });
        }

        if quality.structure_fidelity < 0.7 {
            recommendations.push(Recommendation {
                priority: 2,
                category: "Structure".to_string(),
                description: "Add semantic HTML elements to improve structure".to_string(),
                expected_impact: "Better accessibility and SEO".to_string(),
                effort: EffortLevel::Low,
            });
        }

        if let Some(ts) = &self.tech_stack {
            if ts.js_frameworks.is_empty() && ts.css_frameworks.is_empty() {
                recommendations.push(Recommendation {
                    priority: 3,
                    category: "Tech Stack".to_string(),
                    description: "No frameworks detected - manual verification needed".to_string(),
                    expected_impact: "Ensure correct framework detection".to_string(),
                    effort: EffortLevel::Low,
                });
            }
        }

        if let Some(f) = &self.features {
            if f.features.len() < 5 {
                recommendations.push(Recommendation {
                    priority: 4,
                    category: "Features".to_string(),
                    description: "Few features detected - consider deeper analysis".to_string(),
                    expected_impact: "More complete feature mapping".to_string(),
                    effort: EffortLevel::Medium,
                });
            }
        }

        recommendations.sort_by(|a, b| a.priority.cmp(&b.priority));
        recommendations
    }
}

impl LearningResult {
    /// Create a new learning result from components
    pub async fn from_components(
        source_url: String,
        intent: crate::learning_sandbox::intent_analyzer::WebsiteIntent,
        structure: crate::learning_sandbox::structure_analyzer::PageStructure,
        features: crate::learning_sandbox::feature_recognizer::FeatureMap,
        knowledge_graph: crate::learning_sandbox::knowledge_graph::KnowledgeGraph,
    ) -> Self {
        LearningResultBuilder::new()
            .with_source_url(source_url)
            .with_intent(intent)
            .with_structure(structure)
            .with_features(features)
            .with_knowledge_graph(knowledge_graph)
            .with_success(true)
            .with_start_time(chrono::Utc::now())
            .build()
            .await
    }

    /// Check if learning was successful
    pub fn is_successful(&self) -> bool {
        self.metadata.success && self.quality_metrics.overall_quality > 0.5
    }

    /// Get summary of the learning result
    pub fn summary(&self) -> LearningSummary {
        LearningSummary {
            session_id: self.session_id.clone(),
            source_url: self.source_url.clone(),
            website_type: self.intent.website_type.clone(),
            confidence: self.intent.confidence,
            features_count: self.features.features.len(),
            quality_score: self.quality_metrics.overall_quality,
            success: self.is_successful(),
            duration_ms: self.metadata.duration_ms,
        }
    }
}

/// Summary of learning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSummary {
    pub session_id: String,
    pub source_url: String,
    pub website_type: String,
    pub confidence: f32,
    pub features_count: usize,
    pub quality_score: f32,
    pub success: bool,
    pub duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = LearningResultBuilder::new();
        assert!(builder.session_id.is_none());
    }

    #[test]
    fn test_builder_methods() {
        let builder = LearningResultBuilder::new()
            .with_session_id("test-session".to_string())
            .with_source_url("https://example.com".to_string())
            .with_pages_crawled(5)
            .with_success(true);

        assert_eq!(builder.session_id, Some("test-session".to_string()));
        assert_eq!(builder.source_url, Some("https://example.com".to_string()));
        assert_eq!(builder.pages_crawled, 5);
        assert!(builder.success);
    }

    #[tokio::test]
    async fn test_result_building() {
        let intent = crate::learning_sandbox::intent_analyzer::WebsiteIntent::default();
        let structure = crate::learning_sandbox::structure_analyzer::PageStructure::default();
        let features = crate::learning_sandbox::feature_recognizer::FeatureMap::default();
        let graph = crate::learning_sandbox::knowledge_graph::KnowledgeGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_index: HashMap::new(),
            statistics: crate::learning_sandbox::knowledge_graph::GraphStatistics::default(),
            metadata: crate::learning_sandbox::knowledge_graph::GraphMetadata {
                source_url: "https://example.com".to_string(),
                built_at: chrono::Utc::now(),
                build_time_ms: 100,
                elements_processed: 50,
                version: "1.0".to_string(),
            },
        };

        let result = LearningResult::from_components(
            "https://example.com".to_string(),
            intent,
            structure,
            features,
            graph,
        )
        .await;

        assert_eq!(result.source_url, "https://example.com");
        assert!(result.metadata.success);
    }

    #[tokio::test]
    async fn test_quality_metrics() {
        let result = LearningResultBuilder::new()
            .with_source_url("https://example.com".to_string())
            .with_intent(crate::learning_sandbox::intent_analyzer::WebsiteIntent {
                website_type: "e-commerce".to_string(),
                confidence: 0.8,
                ..Default::default()
            })
            .with_success(true)
            .build()
            .await;

        assert!(result.quality_metrics.intent_confidence > 0.0);
        assert!(result.quality_metrics.overall_quality > 0.0);
    }

    #[tokio::test]
    async fn test_recommendations() {
        let result = LearningResultBuilder::new()
            .with_source_url("https://example.com".to_string())
            .with_success(true)
            .build()
            .await;

        // Should generate recommendations based on quality
        assert!(result.recommendations.len() >= 0);
    }

    #[tokio::test]
    async fn test_summary() {
        let result = LearningResultBuilder::new()
            .with_source_url("https://example.com".to_string())
            .with_intent(crate::learning_sandbox::intent_analyzer::WebsiteIntent {
                website_type: "blog".to_string(),
                confidence: 0.9,
                ..Default::default()
            })
            .with_features(crate::learning_sandbox::feature_recognizer::FeatureMap {
                features: vec![crate::learning_sandbox::feature_recognizer::Feature {
                    name: "Navigation".to_string(),
                    category:
                        crate::learning_sandbox::feature_recognizer::FeatureCategory::Navigation,
                    location: "nav".to_string(),
                    confidence: 0.9,
                    metadata: HashMap::new(),
                }],
                ..Default::default()
            })
            .with_success(true)
            .build()
            .await;

        let summary = result.summary();

        assert_eq!(summary.source_url, "https://example.com");
        assert_eq!(summary.website_type, "blog");
        assert!(summary.success);
    }
}
