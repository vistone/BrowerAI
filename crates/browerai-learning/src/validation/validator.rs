//! Website Validator
//!
//! Validates generated websites against original content.

use crate::data_models::PageContent;
use crate::pipeline::{GeneratedWebsite, ValidationCheck, ValidationReport};

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Minimum HTML similarity score (0.0 - 1.0)
    pub min_html_similarity: f32,

    /// Minimum CSS coverage percentage
    pub min_css_coverage: f32,

    /// Maximum JS error count allowed
    pub max_js_errors: usize,

    /// Whether to check for required elements
    pub check_required_elements: bool,

    /// Required element selectors
    pub required_elements: Vec<String>,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            min_html_similarity: 0.7,
            min_css_coverage: 0.6,
            max_js_errors: 3,
            check_required_elements: true,
            required_elements: vec![
                "header".to_string(),
                "main".to_string(),
                "footer".to_string(),
            ],
        }
    }
}

/// Website Validator
///
/// Validates generated websites against original content.
#[derive(Debug, Clone)]
pub struct WebsiteValidator {
    config: ValidatorConfig,
}

impl WebsiteValidator {
    /// Create a new website validator
    pub fn new() -> Self {
        Self {
            config: ValidatorConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ValidatorConfig) -> Self {
        Self { config }
    }

    /// Validate generated website against original content
    ///
    /// # Arguments
    /// * `original` - Original page content
    /// * `generated` - Generated website content
    ///
    /// # Returns
    /// * `ValidationReport` with validation results
    pub async fn validate(
        &self,
        original: &PageContent,
        generated: &GeneratedWebsite,
    ) -> Option<ValidationReport> {
        let mut checks = Vec::new();
        let mut total_score = 0.0;
        let mut check_count = 0;

        // Check 1: HTML structure validation
        let html_check = self.validate_html_structure(original, &generated.html);
        checks.push(html_check.clone());
        total_score += html_check.score;
        check_count += 1;

        // Check 2: CSS validation
        let css_check = self.validate_css(&generated.css);
        checks.push(css_check.clone());
        total_score += css_check.score;
        check_count += 1;

        // Check 3: JS syntax validation
        let js_check = self.validate_js_syntax(&generated.js);
        checks.push(js_check.clone());
        total_score += js_check.score;
        check_count += 1;

        // Check 4: Required elements check
        if self.config.check_required_elements {
            let elements_check = self.validate_required_elements(&generated.html);
            checks.push(elements_check.clone());
            total_score += elements_check.score;
            check_count += 1;
        }

        // Check 5: Content preservation check
        let content_check = self.validate_content_preservation(original, &generated.html);
        checks.push(content_check.clone());
        total_score += content_check.score;
        check_count += 1;

        let overall_score = if check_count > 0 {
            total_score / check_count as f32
        } else {
            1.0
        };

        let passed =
            overall_score >= self.config.min_html_similarity && checks.iter().all(|c| c.passed);

        Some(ValidationReport {
            passed,
            score: overall_score,
            checks,
        })
    }

    /// Validate HTML structure
    fn validate_html_structure(
        &self,
        _original: &PageContent,
        generated_html: &str,
    ) -> ValidationCheck {
        let score = if generated_html.contains("<!DOCTYPE html>")
            && generated_html.contains("<html")
            && generated_html.contains("<head>")
            && generated_html.contains("<body>")
        {
            1.0
        } else {
            0.5
        };

        let passed = score >= 0.8;

        ValidationCheck {
            name: "HTML Structure".to_string(),
            passed,
            message: if passed {
                "Generated HTML has valid structure".to_string()
            } else {
                "Generated HTML is missing required structure".to_string()
            },
            score,
        }
    }

    /// Validate CSS
    fn validate_css(&self, css: &str) -> ValidationCheck {
        let has_content = !css.trim().is_empty();
        let has_selectors = css.contains("{") && css.contains("}");
        let has_rules = css.contains(":") && css.contains(";");

        let score = if has_content && has_selectors && has_rules {
            1.0
        } else if has_content {
            0.7
        } else {
            0.3
        };

        let passed = score >= 0.6;

        ValidationCheck {
            name: "CSS Validity".to_string(),
            passed,
            message: if passed {
                "Generated CSS is valid".to_string()
            } else {
                "Generated CSS is empty or invalid".to_string()
            },
            score,
        }
    }

    /// Validate JavaScript syntax
    fn validate_js_syntax(&self, js: &str) -> ValidationCheck {
        let has_content = !js.trim().is_empty();

        let score = if has_content {
            // Basic syntax check - count braces
            let open_braces = js.matches('{').count();
            let close_braces = js.matches('}').count();
            let open_parens = js.matches('(').count();
            let close_parens = js.matches(')').count();

            if open_braces == close_braces && open_parens == close_parens {
                1.0
            } else {
                0.5
            }
        } else {
            0.5 // Empty JS is acceptable for static sites
        };

        let passed = score >= 0.6;

        ValidationCheck {
            name: "JavaScript Syntax".to_string(),
            passed,
            message: if passed {
                "JavaScript syntax is valid".to_string()
            } else {
                "JavaScript has syntax errors".to_string()
            },
            score,
        }
    }

    /// Validate required elements exist
    fn validate_required_elements(&self, html: &str) -> ValidationCheck {
        let mut found_count = 0;
        for element in &self.config.required_elements {
            if html.contains(&format!("<{}", element))
                || html.contains(&format!("<{}", element.to_lowercase()))
            {
                found_count += 1;
            }
        }

        let score = found_count as f32 / self.config.required_elements.len() as f32;
        let passed = score >= 0.8;

        ValidationCheck {
            name: "Required Elements".to_string(),
            passed,
            message: format!(
                "Found {}/{} required elements",
                found_count,
                self.config.required_elements.len()
            ),
            score,
        }
    }

    /// Validate content preservation
    fn validate_content_preservation(
        &self,
        original: &PageContent,
        generated_html: &str,
    ) -> ValidationCheck {
        // Use the HTML field for text extraction since dom is a complex type
        let original_text = self.extract_text_from_html(&original.html);
        let generated_text = self.extract_text_from_html(generated_html);

        let original_words: Vec<&str> = original_text.split_whitespace().collect();
        let generated_words: Vec<&str> = generated_text.split_whitespace().collect();

        if original_words.is_empty() || generated_words.is_empty() {
            return ValidationCheck {
                name: "Content Preservation".to_string(),
                passed: true,
                message: "No text content to compare".to_string(),
                score: 1.0,
            };
        }

        let original_set: std::collections::HashSet<&str> =
            original_words.iter().copied().collect();
        let generated_set: std::collections::HashSet<&str> =
            generated_words.iter().copied().collect();

        let intersection: std::collections::HashSet<&str> =
            original_set.intersection(&generated_set).copied().collect();

        let similarity = intersection.len() as f32 / original_set.len() as f32;
        let passed = similarity >= 0.5;

        ValidationCheck {
            name: "Content Preservation".to_string(),
            passed,
            message: format!("Content similarity: {:.1}%", similarity * 100.0),
            score: similarity,
        }
    }

    /// Extract text from HTML string
    fn extract_text_from_html(&self, html: &str) -> String {
        html.replace(|c: char| !c.is_alphanumeric() && c != ' ', " ")
    }
}

impl Default for WebsiteValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_models::PageMetadata;
    use crate::pipeline::GeneratedWebsite;
    use std::collections::HashMap;

    #[test]
    fn test_validator_creation() {
        let validator = WebsiteValidator::new();
        assert!(validator.config.min_html_similarity > 0.0);
    }

    #[test]
    fn test_validator_with_config() {
        let config = ValidatorConfig {
            min_html_similarity: 0.8,
            ..Default::default()
        };
        let validator = WebsiteValidator::with_config(config);
        assert_eq!(validator.config.min_html_similarity, 0.8);
    }

    #[test]
    fn test_html_structure_validation() {
        let validator = WebsiteValidator::new();

        let valid_html = r#"<!DOCTYPE html>
<html>
<head>
    <title>Test</title>
</head>
<body>
    <header>Header</header>
    <main>Content</main>
    <footer>Footer</footer>
</body>
</html>"#;

        let check = validator.validate_html_structure(
            &PageContent::new(
                "https://example.com".to_string(),
                valid_html.to_string(),
                HashMap::new(),
            ),
            valid_html,
        );
        assert!(check.passed);
        assert_eq!(check.name, "HTML Structure");
    }

    #[test]
    fn test_css_validation() {
        let validator = WebsiteValidator::new();

        let valid_css = r#"
            body {
                margin: 0;
                padding: 20px;
                background-color: #ffffff;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
        "#;

        let check = validator.validate_css(valid_css);
        assert!(check.passed);
        assert_eq!(check.name, "CSS Validity");
    }

    #[test]
    fn test_js_syntax_validation() {
        let validator = WebsiteValidator::new();

        let valid_js = r#"
            document.addEventListener('DOMContentLoaded', function() {
                console.log('Page loaded');
                const container = document.querySelector('.container');
                if (container) {
                    container.classList.add('loaded');
                }
            });
        "#;

        let check = validator.validate_js_syntax(valid_js);
        assert!(check.passed);
        assert_eq!(check.name, "JavaScript Syntax");
    }

    #[test]
    fn test_required_elements_validation() {
        let validator = WebsiteValidator::new();

        let html = r#"
            <header>Header</header>
            <main>Content</main>
            <nav>Navigation</nav>
            <footer>Footer</footer>
        "#;

        let check = validator.validate_required_elements(html);
        assert!(check.passed);
        assert_eq!(check.name, "Required Elements");
    }

    #[tokio::test]
    async fn test_full_validation() {
        let validator = WebsiteValidator::new();

        let page_content = PageContent {
            url: "https://example.com".to_string(),
            html: "<!DOCTYPE html><html><body>Test</body></html>".to_string(),
            dom: HashMap::new(),
            inline_css: vec![],
            inline_js: vec![],
            external_resources: vec![],
            metadata: PageMetadata::default(),
            status_code: 200,
            final_url: "https://example.com".to_string(),
            fetched_at: chrono::DateTime::from_timestamp(0, 0).unwrap_or_default(),
            response_headers: HashMap::new(),
            base_url: "https://example.com".to_string(),
        };

        let generated = GeneratedWebsite {
            html: "<!DOCTYPE html><html><head><title>Test</title></head><body><h1>Welcome</h1></body></html>".to_string(),
            css: "body { margin: 0; }".to_string(),
            js: "console.log('test');".to_string(),
            assets: vec![],
        };

        let report = validator.validate(&page_content, &generated).await.unwrap();
        assert!(report.checks.len() >= 4);
        assert!(report.score > 0.0);
    }
}
