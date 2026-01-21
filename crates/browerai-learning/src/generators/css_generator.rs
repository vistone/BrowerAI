//! CSS Generator
//!
//! Generates CSS styles based on website intent and design preferences.

use crate::learning_sandbox::intent_analyzer::WebsiteIntent;
use anyhow::Result;

/// CSS Generator Configuration
#[derive(Debug, Clone)]
pub struct CssGeneratorConfig {
    /// Use CSS custom properties (variables)
    pub use_variables: bool,

    /// Generate responsive styles
    pub responsive: bool,

    /// Include accessibility styles
    pub accessibility: bool,

    /// Base font size in pixels
    pub base_font_size: u16,
}

impl Default for CssGeneratorConfig {
    fn default() -> Self {
        Self {
            use_variables: true,
            responsive: true,
            accessibility: true,
            base_font_size: 16,
        }
    }
}

/// CSS Generator
///
/// Generates CSS styles based on website intent.
#[derive(Debug, Clone)]
pub struct CssGenerator {
    config: CssGeneratorConfig,
}

impl CssGenerator {
    /// Create a new CSS generator
    pub fn new() -> Self {
        Self::with_config(CssGeneratorConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: CssGeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate CSS based on intent
    pub async fn generate(&self, intent: &WebsiteIntent) -> Result<String> {
        let mut css = String::new();

        // CSS Variables
        if self.config.use_variables {
            css.push_str(":root {\n");
            css.push_str("    --primary-color: #007bff;\n");
            css.push_str("    --secondary-color: #6c757d;\n");
            css.push_str("    --accent-color: #28a745;\n");
            css.push_str("    --background-color: #ffffff;\n");
            css.push_str("    --text-color: #333333;\n");
            css.push_str("    --font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;\n");
            css.push_str(&format!(
                "    --base-font-size: {}px;\n",
                self.config.base_font_size
            ));
            css.push_str("}\n\n");
        }

        // CSS Reset
        css.push_str("/* CSS Reset */\n");
        css.push_str("* {\n");
        css.push_str("    margin: 0;\n");
        css.push_str("    padding: 0;\n");
        css.push_str("    box-sizing: border-box;\n");
        css.push_str("}\n\n");

        // Base Styles
        css.push_str("/* Base Styles */\n");
        css.push_str("html {\n");
        css.push_str("    scroll-behavior: smooth;\n");
        css.push_str("}\n\n");

        css.push_str("body {\n");
        css.push_str("    font-family: var(--font-family, sans-serif);\n");
        css.push_str(&format!(
            "    font-size: var(--base-font-size, {}px);\n",
            self.config.base_font_size
        ));
        css.push_str("    line-height: 1.6;\n");
        css.push_str("    color: var(--text-color, #333);\n");
        css.push_str("    background-color: var(--background-color, #fff);\n");
        css.push_str("}\n\n");

        // Typography
        css.push_str("/* Typography */\n");
        css.push_str("h1, h2, h3, h4, h5, h6 {\n");
        css.push_str("    margin-bottom: 0.5em;\n");
        css.push_str("    font-weight: 600;\n");
        css.push_str("    line-height: 1.2;\n");
        css.push_str("}\n\n");

        css.push_str("h1 { font-size: 2.5rem; }\n");
        css.push_str("h2 { font-size: 2rem; }\n");
        css.push_str("h3 { font-size: 1.75rem; }\n");
        css.push_str("h4 { font-size: 1.5rem; }\n\n");

        // Layout
        css.push_str("/* Layout */\n");
        css.push_str(".container {\n");
        css.push_str("    max-width: 1200px;\n");
        css.push_str("    margin: 0 auto;\n");
        css.push_str("    padding: 0 20px;\n");
        css.push_str("}\n\n");

        css.push_str(".main-content {\n");
        css.push_str("    min-height: 60vh;\n");
        css.push_str("}\n\n");

        // Header
        css.push_str("/* Header */\n");
        css.push_str(".header {\n");
        css.push_str("    background-color: #fff;\n");
        css.push_str("    box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n");
        css.push_str("    padding: 1rem 0;\n");
        css.push_str("    position: sticky;\n");
        css.push_str("    top: 0;\n");
        css.push_str("    z-index: 1000;\n");
        css.push_str("}\n\n");

        // Navigation
        css.push_str("/* Navigation */\n");
        css.push_str(".navigation {\n");
        css.push_str("    background-color: #f8f9fa;\n");
        css.push_str("    padding: 0.5rem 0;\n");
        css.push_str("}\n\n");

        css.push_str(".nav-list {\n");
        css.push_str("    display: flex;\n");
        css.push_str("    list-style: none;\n");
        css.push_str("    gap: 1rem;\n");
        css.push_str("}\n\n");

        css.push_str(".nav-list a {\n");
        css.push_str("    text-decoration: none;\n");
        css.push_str("    color: var(--text-color, #333);\n");
        css.push_str("    padding: 0.5rem 1rem;\n");
        css.push_str("    border-radius: 4px;\n");
        css.push_str("    transition: background-color 0.2s;\n");
        css.push_str("}\n\n");

        css.push_str(".nav-list a:hover {\n");
        css.push_str("    background-color: rgba(0,0,0,0.05);\n");
        css.push_str("}\n\n");

        // Main content
        css.push_str("/* Main Content */\n");
        css.push_str(".hero {\n");
        css.push_str("    background: linear-gradient(135deg, var(--primary-color, #007bff), var(--accent-color, #28a745));\n");
        css.push_str("    color: #fff;\n");
        css.push_str("    padding: 4rem 0;\n");
        css.push_str("    text-align: center;\n");
        css.push_str("}\n\n");

        css.push_str(".hero h2 {\n");
        css.push_str("    font-size: 2.5rem;\n");
        css.push_str("    margin-bottom: 1rem;\n");
        css.push_str("}\n\n");

        // Features section
        css.push_str("/* Features */\n");
        css.push_str(".features {\n");
        css.push_str("    padding: 3rem 0;\n");
        css.push_str("}\n\n");

        css.push_str(".features h3 {\n");
        css.push_str("    text-align: center;\n");
        css.push_str("    margin-bottom: 2rem;\n");
        css.push_str("}\n\n");

        css.push_str(".feature-grid {\n");
        css.push_str("    display: grid;\n");
        css.push_str("    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));\n");
        css.push_str("    gap: 2rem;\n");
        css.push_str("}\n\n");

        css.push_str(".feature-card {\n");
        css.push_str("    background: #fff;\n");
        css.push_str("    border-radius: 8px;\n");
        css.push_str("    padding: 1.5rem;\n");
        css.push_str("    box-shadow: 0 2px 8px rgba(0,0,0,0.1);\n");
        css.push_str("    transition: transform 0.2s, box-shadow 0.2s;\n");
        css.push_str("}\n\n");

        css.push_str(".feature-card:hover {\n");
        css.push_str("    transform: translateY(-2px);\n");
        css.push_str("    box-shadow: 0 4px 12px rgba(0,0,0,0.15);\n");
        css.push_str("}\n\n");

        // Sidebar
        if intent.structure.has_sidebar {
            css.push_str("/* Sidebar */\n");
            css.push_str(".sidebar {\n");
            css.push_str("    padding: 1.5rem;\n");
            css.push_str("    background: #f8f9fa;\n");
            css.push_str("    border-radius: 8px;\n");
            css.push_str("}\n\n");
        }

        // Footer
        css.push_str("/* Footer */\n");
        css.push_str(".footer {\n");
        css.push_str("    background-color: #343a40;\n");
        css.push_str("    color: #fff;\n");
        css.push_str("    padding: 2rem 0;\n");
        css.push_str("    text-align: center;\n");
        css.push_str("}\n\n");

        // Buttons
        css.push_str("/* Buttons */\n");
        css.push_str(".btn {\n");
        css.push_str("    display: inline-block;\n");
        css.push_str("    padding: 0.75rem 1.5rem;\n");
        css.push_str("    border: none;\n");
        css.push_str("    border-radius: 4px;\n");
        css.push_str("    font-size: 1rem;\n");
        css.push_str("    cursor: pointer;\n");
        css.push_str("    transition: background-color 0.2s;\n");
        css.push_str("}\n\n");

        css.push_str(".btn-primary {\n");
        css.push_str("    background-color: var(--primary-color, #007bff);\n");
        css.push_str("    color: #fff;\n");
        css.push_str("}\n\n");

        css.push_str(".btn-primary:hover {\n");
        css.push_str("    background-color: #0056b3;\n");
        css.push_str("}\n\n");

        // Accessibility
        if self.config.accessibility {
            css.push_str("/* Accessibility */\n");
            css.push_str("a:focus,\n");
            css.push_str("button:focus {\n");
            css.push_str("    outline: 2px solid var(--primary-color, #007bff);\n");
            css.push_str("    outline-offset: 2px;\n");
            css.push_str("}\n\n");

            css.push_str("@media (prefers-reduced-motion: reduce) {\n");
            css.push_str("    * {\n");
            css.push_str("        animation: none !important;\n");
            css.push_str("        transition: none !important;\n");
            css.push_str("    }\n");
            css.push_str("}\n\n");
        }

        // Responsive styles
        if self.config.responsive {
            css.push_str("/* Responsive */\n");
            css.push_str("@media (max-width: 768px) {\n");
            css.push_str("    .container {\n");
            css.push_str("        padding: 0 15px;\n");
            css.push_str("    }\n\n");

            css.push_str("    .nav-list {\n");
            css.push_str("        flex-wrap: wrap;\n");
            css.push_str("        justify-content: center;\n");
            css.push_str("    }\n\n");

            css.push_str("    .feature-grid {\n");
            css.push_str("        grid-template-columns: 1fr;\n");
            css.push_str("    }\n\n");

            css.push_str("    .hero h2 {\n");
            css.push_str("        font-size: 2rem;\n");
            css.push_str("    }\n");
            css.push_str("}\n\n");

            css.push_str("@media (max-width: 480px) {\n");
            css.push_str("    h1 { font-size: 2rem; }\n");
            css.push_str("    h2 { font-size: 1.5rem; }\n");
            css.push_str("    .btn {\n");
            css.push_str("        width: 100%;\n");
            css.push_str("        text-align: center;\n");
            css.push_str("    }\n");
            css.push_str("}\n");
        }

        // Design style adaptations
        if intent.design_style.minimalism > 0.7 {
            css.push_str("\n/* Minimalist Additions */\n");
            css.push_str(".header, .footer {\n");
            css.push_str("    padding: 0.5rem 0;\n");
            css.push_str("}\n");
        }

        if intent.design_style.modernity > 0.7 {
            css.push_str("\n/* Modern Additions */\n");
            css.push_str(".feature-card {\n");
            css.push_str("    border-radius: 12px;\n");
            css.push_str("    border: 1px solid rgba(0,0,0,0.05);\n");
            css.push_str("}\n");
        }

        Ok(css)
    }
}

impl Default for CssGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning_sandbox::intent_analyzer::{
        ComplexityLevel, DesignStyle, LayoutType, PageStructure, WebsiteIntent,
    };
    use std::collections::HashMap;

    #[test]
    fn test_css_generator_creation() {
        let generator = CssGenerator::new();
        assert!(generator.config.use_variables);
        assert!(generator.config.responsive);
    }

    #[test]
    fn test_generate_css() {
        let generator = CssGenerator::new();

        let intent = WebsiteIntent {
            website_type: "e-commerce".to_string(),
            confidence: 0.9,
            core_features: vec!["shopping_cart".to_string()],
            target_audience: "consumers".to_string(),
            design_style: DesignStyle {
                formality: 0.5,
                colorfulness: 0.5,
                minimalism: 0.6,
                modernity: 0.7,
                primary_colors: None,
                layout_type: None,
            },
            structure: PageStructure {
                has_header: true,
                has_navigation: true,
                has_sidebar: false,
                has_main_content: true,
                has_footer: true,
                layout_type: LayoutType::SingleColumn,
                section_count: 5,
                complexity: ComplexityLevel::Moderate,
            },
            business_model: "B2C".to_string(),
            type_scores: HashMap::new(),
        };

        let css = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(generator.generate(&intent))
            .unwrap();

        assert!(css.contains(":root"));
        assert!(css.contains(".container"));
        assert!(css.contains(".header"));
        assert!(css.contains(".footer"));
        assert!(css.contains("@media"));
    }
}
