/// Enhanced Feedback Collection for Learning Loop
/// Compares original rendering with generated rendering and scores quality

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comparison result between original and generated rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderingComparison {
    /// URL of the page
    pub url: String,
    
    /// Original HTML
    pub original_html: String,
    
    /// Generated HTML
    pub generated_html: String,
    
    /// Original CSS
    pub original_css: String,
    
    /// Generated CSS
    pub generated_css: String,
    
    /// Original JavaScript
    pub original_js: String,
    
    /// Generated JavaScript
    pub generated_js: String,
    
    /// Viewport width for comparison
    pub viewport_width: u32,
    
    /// Viewport height for comparison
    pub viewport_height: u32,
    
    /// Rendered visual hash of original
    pub original_visual_hash: Option<String>,
    
    /// Rendered visual hash of generated
    pub generated_visual_hash: Option<String>,
    
    /// HTML structure similarity (0.0-1.0)
    pub html_similarity: f32,
    
    /// CSS coverage percentage
    pub css_coverage: f32,
    
    /// JavaScript functionality similarity
    pub js_functionality: f32,
    
    /// Visual layout similarity (0.0-1.0)
    pub layout_similarity: f32,
    
    /// Detailed element-by-element comparison
    pub element_comparisons: Vec<ElementComparison>,
    
    /// CSS rule comparisons
    pub css_rule_comparisons: Vec<CSSRuleComparison>,
    
    /// JavaScript event handler comparisons
    pub event_handler_comparisons: Vec<EventHandlerComparison>,
    
    /// Overall quality score (0.0-1.0)
    pub overall_quality: f32,
    
    /// Detailed feedback for improvements
    pub feedback: String,
}

/// Element-level comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementComparison {
    /// CSS selector path
    pub selector: String,
    
    /// Element type (div, button, etc.)
    pub element_type: String,
    
    /// Original element HTML
    pub original_html: String,
    
    /// Generated element HTML
    pub generated_html: String,
    
    /// Is this element present in generated
    pub is_present: bool,
    
    /// Attribute similarity (0.0-1.0)
    pub attribute_similarity: f32,
    
    /// Content similarity (0.0-1.0)
    pub content_similarity: f32,
    
    /// Class list matches
    pub class_match: bool,
    
    /// ID attribute match
    pub id_match: bool,
}

/// CSS rule comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CSSRuleComparison {
    /// CSS selector
    pub selector: String,
    
    /// Original CSS properties
    pub original_properties: HashMap<String, String>,
    
    /// Generated CSS properties
    pub generated_properties: HashMap<String, String>,
    
    /// Matched property count
    pub matched_properties: usize,
    
    /// Missing properties from generation
    pub missing_properties: Vec<String>,
    
    /// Additional properties in generation
    pub extra_properties: Vec<String>,
    
    /// Rule similarity score (0.0-1.0)
    pub similarity: f32,
}

/// JavaScript event handler comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventHandlerComparison {
    /// Event type (click, mouseover, etc.)
    pub event_type: String,
    
    /// Element selector
    pub element_selector: String,
    
    /// Is event handler present in generated
    pub is_present: bool,
    
    /// Handler function name/signature
    pub handler_signature: String,
    
    /// Event delegation usage
    pub uses_delegation: bool,
}

/// Comprehensive feedback collection engine
pub struct FeedbackCollector {
    /// Enable visual comparison (requires headless browser)
    pub enable_visual_comparison: bool,
    
    /// Enable JavaScript testing
    pub enable_js_testing: bool,
    
    /// Similarity threshold for warning (0.0-1.0)
    pub warning_threshold: f32,
    
    /// History of comparisons
    comparison_history: Vec<RenderingComparison>,
}

impl FeedbackCollector {
    /// Create new feedback collector
    pub fn new() -> Self {
        Self {
            enable_visual_comparison: true,
            enable_js_testing: true,
            warning_threshold: 0.75,
            comparison_history: Vec::new(),
        }
    }
    
    /// Collect comprehensive feedback on generated rendering
    pub fn compare_rendering(
        &mut self,
        original: &RenderingComparison,
    ) -> Result<RenderingComparison> {
        // Analyze HTML structure
        let html_similarity = Self::compare_html_structure(
            &original.original_html,
            &original.generated_html,
        )?;
        
        // Analyze CSS rules
        let css_coverage = Self::compare_css_rules(
            &original.original_css,
            &original.generated_css,
        )?;
        
        // Analyze JavaScript
        let js_functionality = Self::compare_javascript(
            &original.original_js,
            &original.generated_js,
        )?;
        
        // Compare visual layout if enabled
        let layout_similarity = if self.enable_visual_comparison {
            Self::compare_visual_layout(
                &original.original_visual_hash,
                &original.generated_visual_hash,
            )?
        } else {
            0.0
        };
        
        // Element-level analysis
        let element_comparisons = Self::analyze_elements(
            &original.original_html,
            &original.generated_html,
        )?;
        
        // CSS rule analysis
        let css_rule_comparisons = Self::analyze_css_rules(
            &original.original_css,
            &original.generated_css,
        )?;
        
        // Event handler analysis
        let event_handler_comparisons = Self::analyze_event_handlers(
            &original.original_js,
            &original.generated_js,
        )?;
        
        // Calculate overall quality
        let overall_quality = Self::calculate_overall_quality(
            html_similarity,
            css_coverage,
            js_functionality,
            layout_similarity,
            element_comparisons.len(),
        );
        
        // Generate detailed feedback
        let feedback = Self::generate_feedback(
            html_similarity,
            css_coverage,
            js_functionality,
            layout_similarity,
            &element_comparisons,
        );
        
        let mut result = original.clone();
        result.html_similarity = html_similarity;
        result.css_coverage = css_coverage;
        result.js_functionality = js_functionality;
        result.layout_similarity = layout_similarity;
        result.element_comparisons = element_comparisons;
        result.css_rule_comparisons = css_rule_comparisons;
        result.event_handler_comparisons = event_handler_comparisons;
        result.overall_quality = overall_quality;
        result.feedback = feedback;
        
        self.comparison_history.push(result.clone());
        
        Ok(result)
    }
    
    /// Compare HTML structure similarity
    fn compare_html_structure(original: &str, generated: &str) -> Result<f32> {
        // Simple tag count comparison
        let original_tags = Self::count_html_tags(original);
        let generated_tags = Self::count_html_tags(generated);
        
        // Calculate similarity based on tag type distribution
        let mut match_count = 0;
        let mut total_tags = 0;
        
        for (tag, count) in &original_tags {
            let generated_count = generated_tags.get(tag).unwrap_or(&0);
            let diff = (*count as f32 - *generated_count as f32).abs();
            let expected = (*count as f32 + *generated_count as f32) / 2.0;
            
            if expected > 0.0 {
                match_count += (1.0 - (diff / expected).min(1.0)) as usize;
            }
            total_tags += 1;
        }
        
        Ok(if total_tags > 0 {
            match_count as f32 / total_tags as f32
        } else {
            0.0
        })
    }
    
    /// Compare CSS rules coverage
    fn compare_css_rules(original: &str, generated: &str) -> Result<f32> {
        // Count CSS rules
        let original_rules = original.matches('{').count();
        let generated_rules = generated.matches('{').count();
        
        if original_rules == 0 {
            return Ok(1.0);
        }
        
        let coverage = generated_rules as f32 / original_rules as f32;
        Ok(coverage.min(1.0))
    }
    
    /// Compare JavaScript functionality
    fn compare_javascript(original: &str, generated: &str) -> Result<f32> {
        // Count function definitions
        let original_functions = original.matches("function").count()
            + original.matches("=>").count();
        let generated_functions = generated.matches("function").count()
            + generated.matches("=>").count();
        
        if original_functions == 0 {
            return Ok(1.0);
        }
        
        let coverage = generated_functions as f32 / original_functions as f32;
        Ok(coverage.min(1.0))
    }
    
    /// Compare visual layout similarity
    fn compare_visual_layout(
        original_hash: &Option<String>,
        generated_hash: &Option<String>,
    ) -> Result<f32> {
        match (original_hash, generated_hash) {
            (Some(orig), Some(gen)) if orig == gen => Ok(1.0),
            (Some(_), Some(_)) => Ok(0.6), // Hashes differ, likely layout differs
            _ => Ok(0.0), // Cannot compare
        }
    }
    
    /// Analyze element-by-element comparison
    fn analyze_elements(original: &str, generated: &str) -> Result<Vec<ElementComparison>> {
        // Simple analysis: count key elements
        let selectors = vec!["header", "nav", "main", "footer", "section", "article"];
        let mut comparisons = Vec::new();
        
        for selector in selectors {
            let original_present = original.contains(&format!("<{}>", selector))
                || original.contains(&format!("class=\"{}\"", selector));
            let generated_present = generated.contains(&format!("<{}>", selector))
                || generated.contains(&format!("class=\"{}\"", selector));
            
            if original_present {
                comparisons.push(ElementComparison {
                    selector: selector.to_string(),
                    element_type: selector.to_string(),
                    original_html: format!("<{}>", selector),
                    generated_html: if generated_present {
                        format!("<{}>", selector)
                    } else {
                        "NOT FOUND".to_string()
                    },
                    is_present: generated_present,
                    attribute_similarity: if generated_present { 1.0 } else { 0.0 },
                    content_similarity: 0.5,
                    class_match: true,
                    id_match: false,
                });
            }
        }
        
        Ok(comparisons)
    }
    
    /// Analyze CSS rules
    fn analyze_css_rules(original: &str, generated: &str) -> Result<Vec<CSSRuleComparison>> {
        // Simplified CSS analysis
        let mut comparisons = Vec::new();
        
        // Extract common selectors
        for selector in &["body", "h1", "button", ".container"] {
            if original.contains(selector) {
                comparisons.push(CSSRuleComparison {
                    selector: selector.to_string(),
                    original_properties: HashMap::new(),
                    generated_properties: HashMap::new(),
                    matched_properties: 0,
                    missing_properties: Vec::new(),
                    extra_properties: Vec::new(),
                    similarity: if generated.contains(selector) {
                        0.8
                    } else {
                        0.3
                    },
                });
            }
        }
        
        Ok(comparisons)
    }
    
    /// Analyze event handlers
    fn analyze_event_handlers(original: &str, generated: &str) -> Result<Vec<EventHandlerComparison>> {
        let event_types = vec!["click", "mouseover", "submit", "change", "load"];
        let mut comparisons = Vec::new();
        
        for event_type in event_types {
            let original_has = original.contains(&format!(".on{}", event_type))
                || original.contains(&format!("on{}=", event_type));
            let generated_has = generated.contains(&format!(".on{}", event_type))
                || generated.contains(&format!("on{}=", event_type));
            
            if original_has {
                comparisons.push(EventHandlerComparison {
                    event_type: event_type.to_string(),
                    element_selector: "*".to_string(),
                    is_present: generated_has,
                    handler_signature: format!("{}Handler", event_type),
                    uses_delegation: false,
                });
            }
        }
        
        Ok(comparisons)
    }
    
    /// Calculate overall quality score
    fn calculate_overall_quality(
        html_sim: f32,
        css_cov: f32,
        js_func: f32,
        layout_sim: f32,
        element_count: usize,
    ) -> f32 {
        // Weighted average
        (html_sim * 0.3 + css_cov * 0.3 + js_func * 0.2 + layout_sim * 0.2)
            * (1.0 - (element_count as f32 * 0.05).min(0.3))
    }
    
    /// Generate detailed feedback
    fn generate_feedback(
        html_sim: f32,
        css_cov: f32,
        js_func: f32,
        layout_sim: f32,
        elements: &[ElementComparison],
    ) -> String {
        let mut feedback = String::new();
        
        feedback.push_str(&format!("HTML Similarity: {:.1}%\n", html_sim * 100.0));
        feedback.push_str(&format!("CSS Coverage: {:.1}%\n", css_cov * 100.0));
        feedback.push_str(&format!("JS Functionality: {:.1}%\n", js_func * 100.0));
        feedback.push_str(&format!("Layout Similarity: {:.1}%\n", layout_sim * 100.0));
        
        if html_sim < 0.8 {
            feedback.push_str("\n⚠️  HTML structure differs significantly from original.\n");
        }
        
        if css_cov < 0.8 {
            feedback.push_str("\n⚠️  CSS coverage incomplete - some styles may be missing.\n");
        }
        
        if js_func < 0.8 {
            feedback.push_str("\n⚠️  JavaScript functionality differs from original.\n");
        }
        
        let missing_elements: Vec<_> = elements
            .iter()
            .filter(|e| !e.is_present)
            .collect();
        
        if !missing_elements.is_empty() {
            feedback.push_str("\n❌ Missing elements:\n");
            for elem in missing_elements {
                feedback.push_str(&format!("  - {}\n", elem.selector));
            }
        }
        
        feedback
    }
    
    /// Count HTML tags
    fn count_html_tags(html: &str) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        
        let tags = vec![
            "div", "span", "p", "h1", "h2", "h3", "button", "input", "form",
            "nav", "header", "footer", "main", "article", "section", "aside",
        ];
        
        for tag in tags {
            let count = html.matches(&format!("<{}", tag)).count();
            if count > 0 {
                counts.insert(tag.to_string(), count);
            }
        }
        
        counts
    }
    
    /// Get comparison history
    pub fn get_history(&self) -> &[RenderingComparison] {
        &self.comparison_history
    }
    
    /// Get average quality over time
    pub fn get_average_quality(&self) -> f32 {
        if self.comparison_history.is_empty() {
            return 0.0;
        }
        
        let sum: f32 = self
            .comparison_history
            .iter()
            .map(|c| c.overall_quality)
            .sum();
        
        sum / self.comparison_history.len() as f32
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
    fn test_html_similarity() {
        let original = "<div><p>Hello</p></div>";
        let generated = "<div><p>Hello</p></div>";
        
        let similarity = FeedbackCollector::compare_html_structure(original, generated)
            .unwrap();
        
        assert!(similarity > 0.9);
    }
    
    #[test]
    fn test_css_coverage() {
        let original = "body { color: red; } h1 { font-size: 24px; }";
        let generated = "body { color: blue; }";
        
        let coverage = FeedbackCollector::compare_css_rules(original, generated).unwrap();
        
        assert!(coverage > 0.4 && coverage < 0.7);
    }
    
    #[test]
    fn test_overall_quality_calculation() {
        let quality = FeedbackCollector::calculate_overall_quality(
            0.9, // html
            0.8, // css
            0.85, // js
            0.8, // layout
            3,   // elements
        );
        
        assert!(quality > 0.50 && quality < 0.90);
    }
    
    #[test]
    fn test_feedback_collector_creation() {
        let collector = FeedbackCollector::new();
        
        assert!(collector.enable_visual_comparison);
        assert!(collector.enable_js_testing);
        assert_eq!(collector.warning_threshold, 0.75);
    }
}
