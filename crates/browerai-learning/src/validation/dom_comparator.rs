//! DOM Comparator
//!
//! Compares DOM structures between original and generated websites
//! to measure structural similarity and identify differences.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// DOM comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomComparisonResult {
    /// Overall similarity score (0.0 - 1.0)
    pub similarity_score: f32,

    /// Structural similarity score
    pub structural_similarity: f32,

    /// Content similarity score
    pub content_similarity: f32,

    /// Element matching score
    pub element_similarity: f32,

    /// Detailed comparison results
    pub comparison_details: ComparisonDetails,

    /// Differences found
    pub differences: Vec<DomDifference>,

    /// Missing elements in generated
    pub missing_elements: Vec<MissingElement>,

    /// Extra elements in generated
    pub extra_elements: Vec<ExtraElement>,

    /// Metadata
    pub metadata: ComparisonMetadata,
}

/// Detailed comparison information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonDetails {
    /// Number of matching elements
    pub matching_elements: usize,

    /// Number of different elements
    pub different_elements: usize,

    /// Number of missing elements
    pub missing_elements: usize,

    /// Number of extra elements
    pub extra_elements: usize,

    /// Total elements in original
    pub original_element_count: usize,

    /// Total elements in generated
    pub generated_element_count: usize,

    /// Tag distribution comparison
    pub tag_distribution: TagDistributionComparison,

    /// Attribute comparison
    pub attribute_comparison: AttributeComparison,
}

/// Comparison of tag distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagDistributionComparison {
    /// Original tag counts
    pub original: HashMap<String, usize>,

    /// Generated tag counts
    pub generated: HashMap<String, usize>,

    /// Tags with significant differences
    pub significant_differences: Vec<TagDifference>,
}

/// A difference in tag count
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagDifference {
    pub tag: String,
    pub original_count: usize,
    pub generated_count: usize,
    pub difference: i64,
    pub percent_change: f32,
}

/// Attribute comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeComparison {
    /// Total attributes compared
    pub total_attributes: usize,

    /// Matching attributes
    pub matching_attributes: usize,

    /// Different attributes
    pub different_attributes: usize,

    /// Most common attribute differences
    pub common_differences: Vec<AttributeDifference>,
}

/// An attribute difference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeDifference {
    pub selector: String,
    pub attribute: String,
    pub original_value: String,
    pub generated_value: String,
}

/// A DOM difference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomDifference {
    /// Difference type
    pub difference_type: DifferenceType,

    /// Element selector
    pub selector: String,

    /// Original content/structure
    pub original: String,

    /// Generated content/structure
    pub generated: String,

    /// Severity of difference
    pub severity: DifferenceSeverity,

    /// Impact assessment
    pub impact: String,
}

/// Types of differences
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DifferenceType {
    Content,
    Structure,
    Attribute,
    Style,
    Order,
    Missing,
    Extra,
}

/// Severity levels for differences
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DifferenceSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// A missing element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingElement {
    /// Element selector
    pub selector: String,

    /// Element tag
    pub tag: String,

    /// Element content (if text)
    pub content: Option<String>,

    /// Reason it's important
    pub importance: ImportanceLevel,

    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Importance levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ImportanceLevel {
    Essential,
    Important,
    Optional,
}

/// An extra element in generated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraElement {
    /// Element selector
    pub selector: String,

    /// Element tag
    pub tag: String,

    /// Element content (if text)
    pub content: Option<String>,

    /// Whether it's problematic
    pub is_problematic: bool,

    /// Suggestion
    pub suggestion: Option<String>,
}

/// Comparison metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetadata {
    /// Original URL
    pub original_url: String,

    /// Comparison timestamp
    pub compared_at: chrono::DateTime<chrono::Utc>,

    /// Time taken for comparison
    pub comparison_time_ms: u64,

    /// Comparison version
    pub version: String,
}

/// DOM Comparator Configuration
#[derive(Debug, Clone)]
pub struct DomComparatorConfig {
    /// Minimum similarity threshold
    pub min_similarity_threshold: f32,

    /// Include detailed differences
    pub include_details: bool,

    /// Maximum differences to report
    pub max_differences: usize,

    /// Compare attributes
    pub compare_attributes: bool,

    /// Compare styles
    pub compare_styles: bool,

    /// Ignore specific elements
    pub ignore_selectors: Vec<String>,
}

impl Default for DomComparatorConfig {
    fn default() -> Self {
        Self {
            min_similarity_threshold: 0.7,
            include_details: true,
            max_differences: 100,
            compare_attributes: true,
            compare_styles: false,
            ignore_selectors: vec![
                "script".to_string(),
                "style".to_string(),
                "noscript".to_string(),
                "iframe".to_string(),
            ],
        }
    }
}

/// DOM Comparator
///
/// Compares DOM structures between original and generated websites.
#[derive(Debug, Clone)]
pub struct DomComparator {
    config: DomComparatorConfig,
}

impl DomComparator {
    /// Create a new DOM comparator
    pub fn new() -> Self {
        Self::with_config(DomComparatorConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: DomComparatorConfig) -> Self {
        Self { config }
    }

    /// Compare two DOM structures
    ///
    /// # Arguments
    /// * `original_html` - Original HTML content
    /// * `generated_html` - Generated HTML content
    /// * `original_url` - Source URL of original
    ///
    /// # Returns
    /// * `DomComparisonResult` with comparison results
    pub async fn compare(
        &self,
        original_html: &str,
        generated_html: &str,
        original_url: &str,
    ) -> DomComparisonResult {
        let start_time = chrono::Utc::now();

        // Parse HTML (simplified - in full implementation would use proper parser)
        let original_elements = self.parse_elements(original_html);
        let generated_elements = self.parse_elements(generated_html);

        // Count elements
        let original_count = original_elements.len();
        let generated_count = generated_elements.len();

        // Calculate tag distributions
        let original_tags = self.count_tags(original_html);
        let generated_tags = self.count_tags(generated_html);

        // Calculate similarities
        let element_similarity = self.calculate_element_similarity(&original_elements, &generated_elements);
        let content_similarity = self.calculate_content_similarity(original_html, generated_html);
        let structural_similarity = self.calculate_structural_similarity(original_html, generated_html);

        // Overall similarity
        let similarity_score = (element_similarity + content_similarity + structural_similarity) / 3.0;

        // Find differences
        let differences = if self.config.include_details {
            self.find_differences(original_html, generated_html)
        } else {
            Vec::new()
        };

        // Find missing elements
        let missing_elements = self.find_missing_elements(&original_elements, &generated_elements);

        // Find extra elements
        let extra_elements = self.find_extra_elements(&original_elements, &generated_elements);

        // Calculate tag distribution comparison
        let tag_distribution = self.compare_tag_distributions(&original_tags, &generated_tags);

        // Attribute comparison (simplified)
        let attribute_comparison = self.compare_attributes(original_html, generated_html);

        let end_time = chrono::Utc::now();
        let comparison_time_ms = (end_time - start_time).num_milliseconds() as u64;

        let details = ComparisonDetails {
            matching_elements: original_elements
                .iter()
                .filter(|e| generated_elements.contains(e))
                .count(),
            different_elements: differences.len(),
            missing_elements: missing_elements.len(),
            extra_elements: extra_elements.len(),
            original_element_count: original_count,
            generated_element_count: generated_count,
            tag_distribution,
            attribute_comparison,
        };

        DomComparisonResult {
            similarity_score,
            structural_similarity,
            content_similarity,
            element_similarity,
            comparison_details: details,
            differences,
            missing_elements,
            extra_elements,
            metadata: ComparisonMetadata {
                original_url: original_url.to_string(),
                compared_at: chrono::Utc::now(),
                comparison_time_ms,
                version: "1.0".to_string(),
            },
        }
    }

    /// Parse elements from HTML (simplified)
    fn parse_elements(&self, html: &str) -> Vec<ParsedElement> {
        let mut elements = Vec::new();

        // Simple regex-based parsing (in production, use proper HTML parser)
        let tag_pattern = r#"<(\w+)([^>]*)>([^<]*)</\1>"#;

        // This is a simplified implementation
        let tags = [
            "div", "span", "p", "a", "ul", "ol", "li", "table", "tr", "td",
            "h1", "h2", "h3", "h4", "h5", "h6",
            "header", "nav", "main", "footer", "section", "article", "aside",
            "form", "input", "button", "select", "textarea",
        ];

        for tag in tags {
            let pattern = format!(r#"<{}"#", tag);
            let count = html.matches(&pattern).count();
            if count > 0 {
                for i in 0..count {
                    elements.push(ParsedElement {
                        tag: tag.to_string(),
                        id: format!("{}-{}", tag, i),
                        attributes: HashMap::new(),
                        content: None,
                    });
                }
            }
        }

        elements
    }

    /// Count tags in HTML
    fn count_tags(&self, html: &str) -> HashMap<String, usize> {
        let mut counts = HashMap::new();

        let tags = [
            "div", "span", "p", "a", "ul", "ol", "li", "table", "tr", "td", "th",
            "form", "input", "button", "select", "textarea",
            "h1", "h2", "h3", "h4", "h5", "h6",
            "header", "nav", "main", "footer", "section", "article", "aside",
            "img", "video", "audio",
        ];

        let html_lower = html.to_lowercase();

        for tag in tags {
            let pattern = format!("<{}", tag);
            let count = html_lower.matches(&pattern).count();
            if count > 0 {
                counts.insert(tag.to_string(), count);
            }
        }

        counts
    }

    /// Calculate element similarity
    fn calculate_element_similarity(
        &self,
        original: &[ParsedElement],
        generated: &[ParsedElement],
    ) -> f32 {
        if original.is_empty() && generated.is_empty() {
            return 1.0;
        }

        if original.is_empty() || generated.is_empty() {
            return 0.0;
        }

        // Count matching tags
        let original_tags: HashMap<&String, usize> =
            original.iter().map(|e| &e.tag).fold(HashMap::new(), |mut m, t| {
                *m.entry(t).or_insert(0) += 1;
                m
            });

        let generated_tags: HashMap<&String, usize> =
            generated.iter().map(|e| &e.tag).fold(HashMap::new(), |mut m, t| {
                *m.entry(t).or_insert(0) += 1;
                m
            });

        // Calculate Jaccard similarity
        let all_tags: HashSet<_> = original_tags.keys().chain(generated_tags.keys()).collect();

        let mut intersection = 0;
        let mut union = 0;

        for tag in &all_tags {
            let o = original_tags.get(tag).copied().unwrap_or(0);
            let g = generated_tags.get(tag).copied().unwrap_or(0);
            intersection += o.min(g);
            union += o.max(g);
        }

        if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Calculate content similarity
    fn calculate_content_similarity(&self, original: &str, generated: &str) -> f32 {
        // Extract text content
        let original_text = self.extract_text(original);
        let generated_text = self.extract_text(generated);

        if original_text.is_empty() && generated_text.is_empty() {
            return 1.0;
        }

        if original_text.is_empty() || generated_text.is_empty() {
            return 0.0;
        }

        // Simple word overlap
        let original_words: HashSet<&str> =
            original_text.split_whitespace().collect();
        let generated_words: HashSet<&str> =
            generated_text.split_whitespace().collect();

        let intersection: HashSet<_> = original_words.intersection(&generated_words).collect();
        let union: HashSet<_> = original_words.union(&generated_words).collect();

        if union.is_empty() {
            1.0
        } else {
            intersection.len() as f32 / union.len() as f32
        }
    }

    /// Extract text from HTML
    fn extract_text(&self, html: &str) -> String {
        html.replace(|c: char| !c.is_alphanumeric() && c != ' ', " ")
    }

    /// Calculate structural similarity
    fn calculate_structural_similarity(&self, original: &str, generated: &str) -> f32 {
        // Check for semantic elements
        let semantic_elements = [
            "header", "nav", "main", "footer", "section", "article", "aside",
        ];

        let mut original_semantic = 0;
        let mut generated_semantic = 0;

        for element in &semantic_elements {
            if original.contains(&format!("<{}", element)) {
                original_semantic += 1;
            }
            if generated.contains(&format!("<{}", element)) {
                generated_semantic += 1;
            }
        }

        if original_semantic == 0 && generated_semantic == 0 {
            return 0.7; // Default if no semantic elements
        }

        let intersection = original_semantic.min(generated_semantic);
        let union = original_semantic.max(generated_semantic);

        if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Find differences between HTML structures
    fn find_differences(&self, original: &str, generated: &str) -> Vec<DomDifference> {
        let mut differences = Vec::new();

        // Check for missing key elements
        let key_elements = ["header", "nav", "main", "footer"];
        for element in &key_elements {
            let has_original = original.contains(&format!("<{}", element));
            let has_generated = generated.contains(&format!("<{}", element));

            if has_original && !has_generated {
                differences.push(DomDifference {
                    difference_type: DifferenceType::Missing,
                    selector: element.to_string(),
                    original: format!("<{}>", element),
                    generated: "".to_string(),
                    severity: if element == &"main" || element == &"nav" {
                        DifferenceSeverity::High
                    } else {
                        DifferenceSeverity::Medium
                    },
                    impact: format!("Missing {} element", element),
                });
            }
        }

        differences.truncate(self.config.max_differences);
        differences
    }

    /// Find missing elements
    fn find_missing_elements(
        &self,
        original: &[ParsedElement],
        generated: &[ParsedElement],
    ) -> Vec<MissingElement> {
        let mut missing = Vec::new();

        // Simple check for semantic elements
        let semantic_tags = ["header", "nav", "main", "footer", "section", "article"];

        for tag in semantic_tags {
            let original_has = original.iter().any(|e| e.tag == tag);
            let generated_has = generated.iter().any(|e| e.tag == tag);

            if original_has && !generated_has {
                missing.push(MissingElement {
                    selector: tag.to_string(),
                    tag: tag.to_string(),
                    content: None,
                    importance: if tag == "main" || tag == "nav" {
                        ImportanceLevel::Essential
                    } else {
                        ImportanceLevel::Important
                    },
                    suggestion: Some(format!("Add <{}> element", tag)),
                });
            }
        }

        missing
    }

    /// Find extra elements
    fn find_extra_elements(
        &self,
        original: &[ParsedElement],
        generated: &[ParsedElement],
    ) -> Vec<ExtraElement> {
        let mut extra = Vec::new();

        // Check for unexpected wrapper elements
        let generated_only: Vec<_> = generated
            .iter()
            .filter(|g| !original.contains(g))
            .collect();

        for element in generated_only.iter().take(10) {
            extra.push(ExtraElement {
                selector: format!("#{}", element.id),
                tag: element.tag.clone(),
                content: element.content.clone(),
                is_problematic: false,
                suggestion: None,
            });
        }

        extra
    }

    /// Compare tag distributions
    fn compare_tag_distributions(
        &self,
        original: &HashMap<String, usize>,
        generated: &HashMap<String, usize>,
    ) -> TagDistributionComparison {
        let mut significant_differences = Vec::new();

        let all_tags: HashSet<_> = original.keys().chain(generated.keys()).collect();

        for tag in all_tags {
            let o = original.get(tag).copied().unwrap_or(0);
            let g = generated.get(tag).copied().unwrap_or(0);
            let diff = g as i64 - o as i64;

            // Significant difference: more than 20% change
            if o > 0 && (diff.abs() as f32 / o as f32) > 0.2 {
                significant_differences.push(TagDifference {
                    tag: tag.clone(),
                    original_count: o,
                    generated_count: g,
                    difference: diff,
                    percent_change: (diff as f32 / o as f32) * 100.0,
                });
            }
        }

        TagDistributionComparison {
            original: original.clone(),
            generated: generated.clone(),
            significant_differences,
        }
    }

    /// Compare attributes (simplified)
    fn compare_attributes(&self, original: &str, generated: &str) -> AttributeComparison {
        AttributeComparison {
            total_attributes: 0,
            matching_attributes: 0,
            different_attributes: 0,
            common_differences: Vec::new(),
        }
    }
}

/// A parsed HTML element
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ParsedElement {
    tag: String,
    id: String,
    attributes: HashMap<String, String>,
    content: Option<String>,
}

impl Default for DomComparator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparator_creation() {
        let comparator = DomComparator::new();
        assert!(comparator.config.min_similarity_threshold > 0.0);
    }

    #[test]
    fn test_tag_counting() {
        let comparator = DomComparator::new();
        let html = r#"
            <div>
                <header><h1>Title</h1></header>
                <nav><a href="#">Link</a></nav>
                <main>
                    <section><article>Content</article></section>
                </main>
                <footer>Footer</footer>
            </div>
        "#;

        let counts = comparator.count_tags(html);

        assert!(counts.get("div").copied().unwrap_or(0) > 0);
        assert!(counts.get("header").copied().unwrap_or(0) > 0);
    }

    #[test]
    fn test_element_similarity() {
        let comparator = DomComparator::new();

        let original = vec![
            ParsedElement {
                tag: "div".to_string(),
                id: "d1".to_string(),
                attributes: HashMap::new(),
                content: None,
            },
            ParsedElement {
                tag: "header".to_string(),
                id: "h1".to_string(),
                attributes: HashMap::new(),
                content: None,
            },
        ];

        let generated = vec![
            ParsedElement {
                tag: "div".to_string(),
                id: "d1".to_string(),
                attributes: HashMap::new(),
                content: None,
            },
        ];

        let similarity = comparator.calculate_element_similarity(&original, &generated);
        assert!(similarity > 0.0 && similarity <= 1.0);
    }

    #[test]
    fn test_content_similarity() {
        let comparator = DomComparator::new();

        let original = "Hello world this is a test";
        let generated = "Hello world this is different";

        let similarity = comparator.calculate_content_similarity(original, generated);
        assert!(similarity > 0.0 && similarity <= 1.0);
    }

    #[test]
    fn test_structural_similarity() {
        let comparator = DomComparator::new();

        let original = "<header></header><nav></nav><main></main><footer></footer>";
        let generated = "<header></header><main></main><footer></footer>";

        let similarity = comparator.calculate_structural_similarity(original, generated);
        assert!(similarity > 0.0 && similarity <= 1.0);
    }

    #[tokio::test]
    async fn test_full_comparison() {
        let comparator = DomComparator::new();

        let original = r#"
            <!DOCTYPE html>
            <html>
            <head><title>Test</title></head>
            <body>
                <header><h1>Title</h1></header>
                <nav><a href="#">Link</a></nav>
                <main>
                    <section><article>Content</article></section>
                </main>
                <footer>Footer</footer>
            </body>
            </html>
        "#;

        let generated = r#"
            <!DOCTYPE html>
            <html>
            <head><title>Test</title></head>
            <body>
                <header><h1>Title</h1></header>
                <main>
                    <section><article>Content</article></section>
                </main>
                <footer>Footer</footer>
            </body>
            </html>
        "#;

        let result = comparator.compare(original, generated, "https://example.com").await;

        assert!(result.similarity_score >= 0.0);
        assert!(result.similarity_score <= 1.0);
        assert!(result.metadata.comparison_time_ms >= 0);
    }

    #[tokio::test]
    async fn test_identical_html() {
        let comparator = DomComparator::new();

        let html = r#"
            <header></header>
            <main></main>
            <footer></footer>
        "#;

        let result = comparator.compare(html, html, "https://example.com").await;

        assert!(result.similarity_score > 0.9);
        assert!(result.missing_elements.is_empty());
    }

    #[tokio::test]
    async fn test_completely_different() {
        let comparator = DomComparator::new();

        let original = r#"<div><header><nav><main><footer></footer></main></nav></header></div>"#;
        let generated = r#"<span><p><a><img></a></p></span>"#;

        let result = comparator.compare(original, generated, "https://example.com").await;

        assert!(result.similarity_score < 0.5);
    }
}
