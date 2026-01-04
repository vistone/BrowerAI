/// Feedback collection system for continuous learning
/// 
/// Collects user feedback and system metrics to improve AI models

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Type of feedback being collected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FeedbackType {
    /// Parsing was correct/incorrect
    ParsingAccuracy,
    /// Rendering quality feedback
    RenderingQuality,
    /// Performance feedback
    Performance,
    /// User satisfaction rating
    UserSatisfaction,
    /// Model prediction accuracy
    PredictionAccuracy,
    /// Cache hit/miss feedback
    CacheEffectiveness,
    /// Layout correctness
    LayoutCorrectness,
    /// Custom feedback type
    Custom(String),
}

/// Individual feedback entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feedback {
    /// Unique feedback ID
    pub id: String,
    /// Type of feedback
    pub feedback_type: FeedbackType,
    /// Timestamp when feedback was collected
    pub timestamp: u64,
    /// Score (0.0 to 1.0, where 1.0 is best)
    pub score: f32,
    /// Optional text feedback
    pub comment: Option<String>,
    /// Context metadata
    pub context: HashMap<String, String>,
    /// Model ID that generated the result
    pub model_id: Option<String>,
}

impl Feedback {
    /// Create a new feedback entry
    pub fn new(feedback_type: FeedbackType, score: f32) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs();
        
        Self {
            id: format!("fb_{}", timestamp),
            feedback_type,
            timestamp,
            score: score.clamp(0.0, 1.0),
            comment: None,
            context: HashMap::new(),
            model_id: None,
        }
    }

    /// Add a comment to the feedback
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    /// Add context metadata
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Set the model ID
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    /// Check if feedback is positive (score >= 0.7)
    pub fn is_positive(&self) -> bool {
        self.score >= 0.7
    }

    /// Check if feedback is negative (score < 0.4)
    pub fn is_negative(&self) -> bool {
        self.score < 0.4
    }
}

/// Statistics for collected feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackStats {
    pub total_count: usize,
    pub positive_count: usize,
    pub negative_count: usize,
    pub average_score: f32,
    pub feedback_by_type: HashMap<String, usize>,
}

/// Feedback collector that aggregates and stores feedback
pub struct FeedbackCollector {
    /// Collected feedback entries
    feedback: Vec<Feedback>,
    /// Maximum feedback entries to keep in memory
    max_entries: usize,
}

impl FeedbackCollector {
    /// Create a new feedback collector
    pub fn new() -> Self {
        Self {
            feedback: Vec::new(),
            max_entries: 10000,
        }
    }

    /// Create a feedback collector with custom capacity
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            feedback: Vec::with_capacity(max_entries.min(1000)),
            max_entries,
        }
    }

    /// Add feedback to the collector
    pub fn add_feedback(&mut self, feedback: Feedback) {
        self.feedback.push(feedback);
        
        // Remove oldest entries if we exceed max_entries
        if self.feedback.len() > self.max_entries {
            self.feedback.drain(0..self.feedback.len() - self.max_entries);
        }
    }

    /// Get all feedback entries
    pub fn get_all_feedback(&self) -> &[Feedback] {
        &self.feedback
    }

    /// Get feedback by type
    pub fn get_feedback_by_type(&self, feedback_type: &FeedbackType) -> Vec<&Feedback> {
        self.feedback
            .iter()
            .filter(|f| &f.feedback_type == feedback_type)
            .collect()
    }

    /// Get feedback for a specific model
    pub fn get_feedback_by_model(&self, model_id: &str) -> Vec<&Feedback> {
        self.feedback
            .iter()
            .filter(|f| f.model_id.as_ref().map(|id| id == model_id).unwrap_or(false))
            .collect()
    }

    /// Get recent feedback (last n entries)
    pub fn get_recent_feedback(&self, count: usize) -> &[Feedback] {
        let start = self.feedback.len().saturating_sub(count);
        &self.feedback[start..]
    }

    /// Calculate statistics for all feedback
    pub fn get_stats(&self) -> FeedbackStats {
        if self.feedback.is_empty() {
            return FeedbackStats {
                total_count: 0,
                positive_count: 0,
                negative_count: 0,
                average_score: 0.0,
                feedback_by_type: HashMap::new(),
            };
        }

        let total_count = self.feedback.len();
        let positive_count = self.feedback.iter().filter(|f| f.is_positive()).count();
        let negative_count = self.feedback.iter().filter(|f| f.is_negative()).count();
        let average_score = self.feedback.iter().map(|f| f.score).sum::<f32>() / total_count as f32;

        let mut feedback_by_type = HashMap::new();
        for f in &self.feedback {
            let type_key = format!("{:?}", f.feedback_type);
            *feedback_by_type.entry(type_key).or_insert(0) += 1;
        }

        FeedbackStats {
            total_count,
            positive_count,
            negative_count,
            average_score,
            feedback_by_type,
        }
    }

    /// Clear all feedback
    pub fn clear(&mut self) {
        self.feedback.clear();
    }

    /// Export feedback as JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.feedback)
    }

    /// Import feedback from JSON
    pub fn import_json(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let feedback: Vec<Feedback> = serde_json::from_str(json)?;
        for f in feedback {
            self.add_feedback(f);
        }
        Ok(())
    }
}

impl Default for FeedbackCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_creation() {
        let feedback = Feedback::new(FeedbackType::ParsingAccuracy, 0.85);
        assert_eq!(feedback.feedback_type, FeedbackType::ParsingAccuracy);
        assert_eq!(feedback.score, 0.85);
        assert!(feedback.is_positive());
        assert!(!feedback.is_negative());
    }

    #[test]
    fn test_feedback_with_comment() {
        let feedback = Feedback::new(FeedbackType::RenderingQuality, 0.9)
            .with_comment("Excellent rendering");
        assert_eq!(feedback.comment, Some("Excellent rendering".to_string()));
    }

    #[test]
    fn test_feedback_with_context() {
        let feedback = Feedback::new(FeedbackType::Performance, 0.75)
            .with_context("url", "https://example.com")
            .with_model_id("html_parser_v1");
        
        assert_eq!(feedback.context.get("url"), Some(&"https://example.com".to_string()));
        assert_eq!(feedback.model_id, Some("html_parser_v1".to_string()));
    }

    #[test]
    fn test_feedback_score_clamping() {
        let feedback1 = Feedback::new(FeedbackType::UserSatisfaction, 1.5);
        assert_eq!(feedback1.score, 1.0);

        let feedback2 = Feedback::new(FeedbackType::UserSatisfaction, -0.5);
        assert_eq!(feedback2.score, 0.0);
    }

    #[test]
    fn test_feedback_collector_add() {
        let mut collector = FeedbackCollector::new();
        collector.add_feedback(Feedback::new(FeedbackType::ParsingAccuracy, 0.8));
        collector.add_feedback(Feedback::new(FeedbackType::RenderingQuality, 0.9));
        
        assert_eq!(collector.get_all_feedback().len(), 2);
    }

    #[test]
    fn test_feedback_collector_stats() {
        let mut collector = FeedbackCollector::new();
        collector.add_feedback(Feedback::new(FeedbackType::ParsingAccuracy, 0.8));
        collector.add_feedback(Feedback::new(FeedbackType::ParsingAccuracy, 0.3));
        collector.add_feedback(Feedback::new(FeedbackType::RenderingQuality, 0.9));
        
        let stats = collector.get_stats();
        assert_eq!(stats.total_count, 3);
        assert_eq!(stats.positive_count, 2); // 0.8 and 0.9 are positive
        assert_eq!(stats.negative_count, 1); // 0.3 is negative
        assert!((stats.average_score - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_feedback_collector_by_type() {
        let mut collector = FeedbackCollector::new();
        collector.add_feedback(Feedback::new(FeedbackType::ParsingAccuracy, 0.8));
        collector.add_feedback(Feedback::new(FeedbackType::RenderingQuality, 0.9));
        collector.add_feedback(Feedback::new(FeedbackType::ParsingAccuracy, 0.7));
        
        let parsing_feedback = collector.get_feedback_by_type(&FeedbackType::ParsingAccuracy);
        assert_eq!(parsing_feedback.len(), 2);
    }

    #[test]
    fn test_feedback_collector_by_model() {
        let mut collector = FeedbackCollector::new();
        collector.add_feedback(
            Feedback::new(FeedbackType::ParsingAccuracy, 0.8)
                .with_model_id("model_v1")
        );
        collector.add_feedback(
            Feedback::new(FeedbackType::ParsingAccuracy, 0.9)
                .with_model_id("model_v2")
        );
        
        let model_v1_feedback = collector.get_feedback_by_model("model_v1");
        assert_eq!(model_v1_feedback.len(), 1);
    }

    #[test]
    fn test_feedback_collector_recent() {
        let mut collector = FeedbackCollector::new();
        for i in 0..10 {
            collector.add_feedback(Feedback::new(FeedbackType::Performance, i as f32 / 10.0));
        }
        
        let recent = collector.get_recent_feedback(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_feedback_collector_max_entries() {
        let mut collector = FeedbackCollector::with_capacity(5);
        for i in 0..10 {
            collector.add_feedback(Feedback::new(FeedbackType::Performance, i as f32 / 10.0));
        }
        
        // Should only keep the last 5 entries
        assert_eq!(collector.get_all_feedback().len(), 5);
    }

    #[test]
    fn test_feedback_json_export_import() {
        let mut collector = FeedbackCollector::new();
        collector.add_feedback(Feedback::new(FeedbackType::ParsingAccuracy, 0.8));
        collector.add_feedback(Feedback::new(FeedbackType::RenderingQuality, 0.9));
        
        let json = collector.export_json().unwrap();
        
        let mut new_collector = FeedbackCollector::new();
        new_collector.import_json(&json).unwrap();
        
        assert_eq!(new_collector.get_all_feedback().len(), 2);
    }
}
