//! Website Feature Extractor
//!
//! Extracts a 48-dimensional feature vector from website content and intent analysis.
//! This feature vector is used as input for the Python learning engine.
//!
//! Feature dimensions:
//! [0-9]:   HTML structure (10)
//! [10-17]: CSS features (8)
//! [18-27]: JavaScript features (10)
//! [28-35]: Page structure (8)
//! [36-42]: Design style (7)
//! [43-47]: Complexity metrics (5)
//!
//! Total: 48 dimensions

use anyhow::Result;
use regex::Regex;
use crate::data_models::PageContent;
use crate::learning_sandbox::{WebsiteIntent, intent_analyzer::ComplexityLevel};
use std::collections::HashSet;

/// Website Feature Extractor
///
/// Converts PageContent and WebsiteIntent into a standardized 48-dimensional feature vector
/// for use in the Python learning engine.
pub struct WebsiteFeatureExtractor;

impl WebsiteFeatureExtractor {
    /// Extract 48-dimensional feature vector from page content and intent
    ///
    /// # Arguments
    /// * `page_content` - The parsed web page content
    /// * `intent` - The analyzed website intent
    ///
    /// # Returns
    /// A vector of exactly 48 f32 values
    pub fn extract(page_content: &PageContent, intent: &WebsiteIntent) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(48);

        // Aggregate CSS and JavaScript content
        let css = page_content.all_css();
        let javascript = page_content.all_js();

        // 0-9: HTML Structure Features (10)
        features.extend(Self::extract_html_metrics(&page_content.html)?);

        // 10-17: CSS Features (8)
        features.extend(Self::extract_css_metrics(&css)?);

        // 18-27: JavaScript Features (10)
        features.extend(Self::extract_js_metrics(&javascript)?);

        // 28-35: Page Structure Features (8)
        features.extend(Self::extract_structure_metrics(page_content, intent)?);

        // 36-42: Design Style Features (7)
        features.extend(Self::extract_design_style_metrics(intent)?);

        // 43-47: Complexity Metrics (5)
        features.extend(Self::extract_complexity_metrics(page_content)?);

        // Ensure we have exactly 48 features
        anyhow::ensure!(
            features.len() == 48,
            "Feature vector size mismatch: expected 48, got {}",
            features.len()
        );

        // Validate all features are finite
        for (i, &feature) in features.iter().enumerate() {
            anyhow::ensure!(
                feature.is_finite(),
                "Non-finite feature at position {}: {}",
                i,
                feature
            );
        }

        Ok(features)
    }

    /// Extract HTML structure metrics (features 0-9)
    fn extract_html_metrics(html: &str) -> Result<Vec<f32>> {
        let mut metrics = Vec::with_capacity(10);

        // 0: HTML line count
        metrics.push(html.lines().count() as f32);

        // 1: HTML size in KB
        metrics.push((html.len() as f32) / 1024.0);

        // 2: Total tag count
        let tag_count = html.matches('<').count();
        metrics.push(tag_count as f32);

        // 3: Div count
        let div_count = Self::count_html_tag(html, "div");
        metrics.push(div_count as f32);

        // 4: Class attribute count
        let class_count = html.matches("class=").count();
        metrics.push(class_count as f32);

        // 5: ID attribute count
        let id_count = html.matches("id=").count();
        metrics.push(id_count as f32);

        // 6: Semantic tag count (header, main, section, article, nav, aside, footer)
        let semantic_tags = [
            "header", "main", "section", "article", "nav", "aside", "footer",
        ];
        let semantic_count: usize = semantic_tags
            .iter()
            .map(|tag| Self::count_html_tag(html, tag))
            .sum();
        metrics.push(semantic_count as f32);

        // 7: Form count
        let form_count = Self::count_html_tag(html, "form");
        metrics.push(form_count as f32);

        // 8: Input element count
        let input_count = Self::count_html_tag(html, "input");
        metrics.push(input_count as f32);

        // 9: Button element count
        let button_count = Self::count_html_tag(html, "button");
        metrics.push(button_count as f32);

        Ok(metrics)
    }

    /// Extract CSS metrics (features 10-17)
    fn extract_css_metrics(css: &str) -> Result<Vec<f32>> {
        let mut metrics = Vec::with_capacity(8);

        // 10: CSS size in KB
        metrics.push((css.len() as f32) / 1024.0);

        // 11: CSS rule count (count occurrences of '{')
        let rule_count = css.matches('{').count();
        metrics.push(rule_count as f32);

        // 12: Unique color count (approximate via hex color patterns)
        let color_count = Self::count_hex_colors(css);
        metrics.push(color_count as f32);

        // 13: Font family count (count 'font-family:' declarations)
        let font_count = css.matches("font-family").count();
        metrics.push(font_count as f32);

        // 14: Media query count
        let media_query_count = css.matches("@media").count();
        metrics.push(media_query_count as f32);

        // 15: Animation count (count @keyframes and animation declarations)
        let animation_count = css.matches("@keyframes").count() + css.matches("animation").count();
        metrics.push(animation_count as f32);

        // 16: Gradient count
        let gradient_count = css.matches("gradient").count();
        metrics.push(gradient_count as f32);

        // 17: Border-radius count
        let border_radius_count = css.matches("border-radius").count();
        metrics.push(border_radius_count as f32);

        Ok(metrics)
    }

    /// Extract JavaScript metrics (features 18-27)
    fn extract_js_metrics(javascript: &str) -> Result<Vec<f32>> {
        let mut metrics = Vec::with_capacity(10);

        // 18: JS size in KB
        metrics.push((javascript.len() as f32) / 1024.0);

        // 19: JS line count
        metrics.push(javascript.lines().count() as f32);

        // 20: Function declaration count
        let func_count = javascript.matches("function").count()
            + (javascript.matches("=>").count() / 2);
        metrics.push(func_count as f32);

        // 21: Class declaration count
        let class_count = javascript.matches("class").count();
        metrics.push(class_count as f32);

        // 22: Variable declarations (var, let, const count)
        let var_count = javascript.matches("var ").count()
            + javascript.matches("let ").count()
            + javascript.matches("const ").count();
        metrics.push(var_count as f32);

        // 23: Event listener count (addEventListener calls)
        let event_count = javascript.matches("addEventListener").count();
        metrics.push(event_count as f32);

        // 24: API call count (fetch, XMLHttpRequest, axios patterns)
        let api_count = javascript.matches("fetch").count()
            + javascript.matches("XMLHttpRequest").count()
            + javascript.matches(".post(").count()
            + javascript.matches(".get(").count();
        metrics.push(api_count as f32);

        // 25: Library imports (import, require, script includes)
        let import_count = javascript.matches("import").count()
            + javascript.matches("require(").count();
        metrics.push(import_count as f32);

        // 26: Async function count
        let async_count = javascript.matches("async").count();
        metrics.push(async_count as f32);

        // 27: Comment ratio (comments / total lines)
        let comment_lines = javascript.lines().filter(|l| l.trim().starts_with("//")).count();
        let comment_ratio = if javascript.lines().count() > 0 {
            (comment_lines as f32) / (javascript.lines().count() as f32)
        } else {
            0.0
        };
        metrics.push(comment_ratio);

        Ok(metrics)
    }

    /// Extract page structure metrics (features 28-35)
    fn extract_structure_metrics(
        page_content: &PageContent,
        intent: &WebsiteIntent,
    ) -> Result<Vec<f32>> {
        let mut metrics = Vec::with_capacity(8);

        // 28: Has header (boolean as f32)
        metrics.push(if intent.structure.has_header { 1.0 } else { 0.0 });

        // 29: Has footer (boolean as f32)
        metrics.push(if intent.structure.has_footer { 1.0 } else { 0.0 });

        // 30: Has navigation (boolean as f32)
        metrics.push(if intent.structure.has_navigation { 1.0 } else { 0.0 });

        // 31: Has sidebar (boolean as f32)
        metrics.push(if intent.structure.has_sidebar { 1.0 } else { 0.0 });

        // 32: Maximum nesting depth
        let max_depth = Self::calculate_max_nesting_depth(&page_content.html);
        metrics.push(max_depth as f32);

        // 33: Section count
        let section_count = Self::count_html_tag(&page_content.html, "section");
        metrics.push(section_count as f32);

        // 34: Article count
        let article_count = Self::count_html_tag(&page_content.html, "article");
        metrics.push(article_count as f32);

        // 35: Aside count
        let aside_count = Self::count_html_tag(&page_content.html, "aside");
        metrics.push(aside_count as f32);

        Ok(metrics)
    }

    /// Extract design style metrics (features 36-42)
    fn extract_design_style_metrics(intent: &WebsiteIntent) -> Result<Vec<f32>> {
        let mut metrics = Vec::with_capacity(7);

        // 36: Formality (0.0 = casual, 1.0 = formal)
        metrics.push(intent.design_style.formality);

        // 37: Colorfulness (0.0 = minimal, 1.0 = colorful)
        metrics.push(intent.design_style.colorfulness);

        // 38: Minimalism (0.0 = complex, 1.0 = minimal)
        metrics.push(intent.design_style.minimalism);

        // 39: Modernity (0.0 = traditional, 1.0 = modern)
        metrics.push(intent.design_style.modernity);

        // 40: Complexity score (normalized from ComplexityLevel)
        let complexity_score = match &intent.structure.complexity {
            ComplexityLevel::Simple => 0.2,
            ComplexityLevel::Moderate => 0.5,
            ComplexityLevel::Complex => 0.7,
            ComplexityLevel::VeryComplex => 0.9,
        };
        metrics.push(complexity_score);

        // 41: Primary color count (if available)
        let color_count = if let Some(ref colors) = intent.design_style.primary_colors {
            colors.len() as f32
        } else {
            0.0
        };
        metrics.push(color_count);

        // 42: Layout type score (single-column=0.3, grid=0.7, flex=0.9)
        let layout_score = if let Some(ref layout) = intent.design_style.layout_type {
            match layout.to_lowercase().as_str() {
                s if s.contains("single") => 0.3,
                s if s.contains("grid") => 0.7,
                s if s.contains("flex") => 0.9,
                _ => 0.5,
            }
        } else {
            0.5
        };
        metrics.push(layout_score);

        Ok(metrics)
    }

    /// Extract complexity metrics (features 43-47)
    fn extract_complexity_metrics(page_content: &PageContent) -> Result<Vec<f32>> {
        let mut metrics = Vec::with_capacity(5);

        let css = page_content.all_css();
        let javascript = page_content.all_js();

        // 43: Total page size in KB (HTML + CSS + JS)
        let total_size =
            ((page_content.html.len() + css.len() + javascript.len()) as f32) / 1024.0;
        metrics.push(total_size);

        // 44: Image count
        let image_count = Self::count_html_tag(&page_content.html, "img");
        metrics.push(image_count as f32);

        // 45: Video count
        let video_count = Self::count_html_tag(&page_content.html, "video");
        metrics.push(video_count as f32);

        // 46: External script count
        let script_count = page_content
            .external_resources
            .iter()
            .filter(|r| r.resource_type == crate::data_models::ResourceType::JavaScript)
            .count();
        metrics.push(script_count as f32);

        // 47: CDN resource ratio (external resources / total resources)
        let total_resource_count = page_content.external_resources.len() + page_content.inline_css.len() + page_content.inline_js.len();
        let cdn_ratio = if total_resource_count > 0 {
            (page_content.external_resources.len() as f32) / (total_resource_count as f32)
        } else {
            0.0
        };
        metrics.push(cdn_ratio);

        Ok(metrics)
    }

    /// Count occurrences of an HTML tag
    fn count_html_tag(html: &str, tag: &str) -> usize {
        let pattern = format!(r"<{}\s|<{}>", tag, tag);
        if let Ok(re) = Regex::new(&pattern) {
            re.find_iter(html).count()
        } else {
            0
        }
    }

    /// Count hex color values in CSS
    fn count_hex_colors(css: &str) -> usize {
        if let Ok(re) = Regex::new(r"#[0-9a-fA-F]{3,6}") {
            let mut colors = HashSet::new();
            for color in re.find_iter(css) {
                colors.insert(color.as_str().to_lowercase());
            }
            colors.len()
        } else {
            0
        }
    }

    /// Calculate maximum nesting depth in HTML
    fn calculate_max_nesting_depth(html: &str) -> usize {
        let mut max_depth = 0;
        let mut current_depth = 0;

        for line in html.lines() {
            let opens = line.matches('<').count() - line.matches("</").count();
            let closes = line.matches("</").count();

            current_depth += opens;
            if current_depth > max_depth {
                max_depth = current_depth;
            }
            current_depth = current_depth.saturating_sub(closes);
        }

        max_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning_sandbox::intent_analyzer::{
        DesignStyle, PageStructure, LayoutType, ComplexityLevel,
    };

    fn create_test_page_content() -> PageContent {
        let mut content = PageContent::new(
            "https://example.com".to_string(),
            "\
                <!DOCTYPE html>\
                <html>\
                <head><title>Test</title></head>\
                <body>\
                    <header><nav><a href=\"#\">Home</a></nav></header>\
                    <main>\
                        <section><h1>Content</h1></section>\
                        <article><p>Article</p></article>\
                    </main>\
                    <footer></footer>\
                </body>\
                </html>\
            "
            .to_string(),
            Default::default(),
        );

        // Add inline CSS
        content.add_inline_css(
            "body { color: #ff0000; } @media (max-width: 600px) {}".to_string(),
            "style".to_string(),
        );

        // Add inline JavaScript
        content.add_inline_js(
            "function test() { const x = 1; } addEventListener('click', () => {});".to_string(),
            "script".to_string(),
        );

        content
    }

    fn create_test_intent() -> WebsiteIntent {
        WebsiteIntent {
            website_type: "blog".to_string(),
            confidence: 0.85,
            core_features: vec!["articles".to_string(), "search".to_string()],
            target_audience: "readers".to_string(),
            design_style: DesignStyle {
                formality: 0.7,
                colorfulness: 0.6,
                minimalism: 0.4,
                modernity: 0.8,
                primary_colors: Some(vec!["#ff0000".to_string()]),
                layout_type: Some("grid".to_string()),
            },
            structure: PageStructure {
                has_header: true,
                has_navigation: true,
                has_sidebar: false,
                has_main_content: true,
                has_footer: true,
                layout_type: LayoutType::Grid,
                section_count: 3,
                complexity: ComplexityLevel::Moderate,
            },
            business_model: "advertising".to_string(),
            type_scores: Default::default(),
        }
    }

    #[test]
    fn test_feature_extraction_returns_48_dimensions() {
        let content = create_test_page_content();
        let intent = create_test_intent();

        let features = WebsiteFeatureExtractor::extract(&content, &intent).unwrap();

        assert_eq!(features.len(), 48);
    }

    #[test]
    fn test_all_features_are_finite() {
        let content = create_test_page_content();
        let intent = create_test_intent();

        let features = WebsiteFeatureExtractor::extract(&content, &intent).unwrap();

        for (i, &feature) in features.iter().enumerate() {
            assert!(
                feature.is_finite(),
                "Feature {} is not finite: {}",
                i,
                feature
            );
        }
    }

    #[test]
    fn test_features_are_non_negative() {
        let content = create_test_page_content();
        let intent = create_test_intent();

        let features = WebsiteFeatureExtractor::extract(&content, &intent).unwrap();

        for (i, &feature) in features.iter().enumerate() {
            assert!(
                feature >= 0.0,
                "Feature {} is negative: {}",
                i,
                feature
            );
        }
    }

    #[test]
    fn test_html_metrics_extraction() {
        let html = "<div class='test' id='id1'><form><input/><button></button></form></div>";
        let metrics = WebsiteFeatureExtractor::extract_html_metrics(html).unwrap();

        assert_eq!(metrics.len(), 10);
        assert!(metrics[2] > 0.0); // tag count
        assert!(metrics[3] >= 1.0); // div count
    }

    #[test]
    fn test_css_metrics_extraction() {
        let css = "body { color: #ff0000; } @media (max-width: 600px) { color: #00ff00; }";
        let metrics = WebsiteFeatureExtractor::extract_css_metrics(css).unwrap();

        assert_eq!(metrics.len(), 8);
        assert!(metrics[1] > 0.0); // rule count
        assert!(metrics[2] >= 2.0); // color count
    }

    #[test]
    fn test_js_metrics_extraction() {
        let js = "function test() { const x = 1; } addEventListener('click', () => {});";
        let metrics = WebsiteFeatureExtractor::extract_js_metrics(js).unwrap();

        assert_eq!(metrics.len(), 10);
        assert!(metrics[1] > 0.0); // line count
        assert!(metrics[2] > 0.0); // function count
    }

    #[test]
    fn test_consistency() {
        let content = create_test_page_content();
        let intent = create_test_intent();

        let features1 = WebsiteFeatureExtractor::extract(&content, &intent).unwrap();
        let features2 = WebsiteFeatureExtractor::extract(&content, &intent).unwrap();

        assert_eq!(features1, features2);
    }
}
