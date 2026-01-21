//! Structure Analyzer
//!
//! Analyzes the page structure including layout type, DOM depth,
//! element distribution, and structural patterns.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Page structure analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageStructure {
    /// Detected layout type
    pub layout_type: LayoutType,

    /// Layout confidence score
    pub layout_confidence: f32,

    /// DOM tree information
    pub dom_info: DomInfo,

    /// Element distribution analysis
    pub element_distribution: ElementDistribution,

    /// Structural patterns detected
    pub patterns: Vec<StructuralPattern>,

    /// Semantic structure
    pub semantic_structure: SemanticStructure,

    /// Complexity metrics
    pub complexity_metrics: ComplexityMetrics,

    /// Analysis metadata
    pub metadata: StructureMetadata,
}

/// Layout types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LayoutType {
    SingleColumn,
    TwoColumn,
    ThreeColumn,
    Grid,
    Masonry,
    Magazine,
    Dashboard,
    SplitScreen,
    FullWidth,
    Unknown,
}

/// Information about the DOM tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomInfo {
    /// Maximum DOM depth
    pub max_depth: usize,

    /// Total number of elements
    pub total_elements: usize,

    /// Number of text nodes
    pub text_nodes: usize,

    /// Average depth of elements
    pub average_depth: f32,

    /// DOM width (max siblings at any level)
    pub max_siblings: usize,

    /// Depth distribution
    pub depth_distribution: HashMap<usize, usize>,
}

/// Distribution of element types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementDistribution {
    /// Semantic elements count
    pub semantic_elements: HashMap<String, usize>,

    /// Interactive elements count
    pub interactive_elements: HashMap<String, usize>,

    /// Media elements count
    pub media_elements: HashMap<String, usize>,

    /// Form elements count
    pub form_elements: HashMap<String, usize>,

    /// Most common tag names
    pub top_tags: Vec<(String, usize)>,

    /// Total element count
    pub total_count: usize,
}

/// Structural patterns detected on the page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralPattern {
    /// Pattern name
    pub name: String,

    /// Pattern type
    pub pattern_type: PatternType,

    /// Confidence of detection
    pub confidence: f32,

    /// Location in page
    pub location: String,

    /// Pattern description
    pub description: String,
}

/// Types of structural patterns
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PatternType {
    Container,
    Grid,
    Flexbox,
    List,
    Table,
    Card,
    Modal,
    Navigation,
    Hero,
    Sidebar,
    Footer,
    Header,
    Section,
    Article,
    Aside,
    Custom,
}

/// Semantic structure of the page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticStructure {
    /// Header elements found
    pub has_header: bool,

    /// Navigation elements found
    pub has_navigation: bool,

    /// Main content area found
    pub has_main: bool,

    /// Footer elements found
    pub has_footer: bool,

    /// Sidebar elements found
    pub has_sidebar: bool,

    /// Article elements found
    pub has_article: bool,

    /// Section elements found
    pub has_section: bool,

    /// Aside elements found
    pub has_aside: bool,

    /// Heading hierarchy
    pub heading_levels: HashMap<usize, usize>,

    /// Content organization score
    pub organization_score: f32,
}

/// Complexity metrics for the page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Structural complexity (0.0 - 1.0)
    pub structural_complexity: f32,

    /// Nesting complexity (0.0 - 1.0)
    pub nesting_complexity: f32,

    /// Element variety score (0.0 - 1.0)
    pub element_variety: f32,

    /// Overall complexity rating
    pub complexity_rating: ComplexityLevel,

    /// Estimated rendering complexity
    pub estimated_render_time: f32,
}

/// Complexity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComplexityLevel {
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}

/// Metadata about the analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureMetadata {
    /// URL analyzed
    pub url: String,

    /// Analysis timestamp
    pub analyzed_at: chrono::DateTime<chrono::Utc>,

    /// Time taken for analysis
    pub analysis_time_ms: u64,

    /// HTML size in bytes
    pub html_size: usize,

    /// Number of characters processed
    pub characters_processed: usize,
}

/// Structure Analyzer Configuration
#[derive(Debug, Clone)]
pub struct StructureAnalyzerConfig {
    /// Minimum pattern confidence
    pub min_confidence: f32,

    /// Enable detailed DOM analysis
    pub detailed_dom_analysis: bool,

    /// Maximum DOM depth to analyze
    pub max_depth_analysis: usize,

    /// Enable semantic analysis
    pub semantic_analysis: bool,
}

impl Default for StructureAnalyzerConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            detailed_dom_analysis: true,
            max_depth_analysis: 50,
            semantic_analysis: true,
        }
    }
}

/// Structure Analyzer
///
/// Analyzes the structure and layout of web pages.
#[derive(Debug, Clone)]
pub struct StructureAnalyzer {
    config: StructureAnalyzerConfig,
}

impl StructureAnalyzer {
    /// Create a new structure analyzer
    pub fn new() -> Self {
        Self::with_config(StructureAnalyzerConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: StructureAnalyzerConfig) -> Self {
        Self { config }
    }

    /// Analyze page structure
    ///
    /// # Arguments
    /// * `page` - Page content to analyze
    ///
    /// # Returns
    /// * `PageStructure` with analysis results
    pub async fn analyze(&self, page: &crate::data_models::PageContent) -> PageStructure {
        let start_time = chrono::Utc::now();

        let html = &page.html;

        // Analyze DOM structure
        let dom_info = self.analyze_dom_structure(html);

        // Analyze element distribution
        let element_distribution = self.analyze_element_distribution(html);

        // Detect layout type
        let (layout_type, layout_confidence) = self.detect_layout_type(html, &element_distribution);

        // Detect structural patterns
        let patterns = self.detect_patterns(html);

        // Analyze semantic structure
        let semantic_structure = self.analyze_semantic_structure(html);

        // Calculate complexity metrics
        let complexity_metrics = self.calculate_complexity(&dom_info, &element_distribution);

        let end_time = chrono::Utc::now();
        let analysis_time_ms = (end_time - start_time).num_milliseconds() as u64;

        PageStructure {
            layout_type,
            layout_confidence,
            dom_info,
            element_distribution,
            patterns,
            semantic_structure,
            complexity_metrics,
            metadata: StructureMetadata {
                url: page.url.clone(),
                analyzed_at: chrono::Utc::now(),
                analysis_time_ms,
                html_size: html.len(),
                characters_processed: html.len(),
            },
        }
    }

    /// Analyze DOM structure
    fn analyze_dom_structure(&self, html: &str) -> DomInfo {
        let mut max_depth = 0;
        let mut total_elements = 0;
        let mut text_nodes = 0;
        let mut depth_distribution: HashMap<usize, usize> = HashMap::new();
        let mut current_depth = 0;
        let mut max_siblings = 0;

        // Count tag occurrences
        let tag_counts = self.count_tags(html);

        // Simple depth estimation based on nesting patterns
        for line in html.lines() {
            let indent = line.len() - line.trim_start().len();
            let depth = (indent / 2).min(self.config.max_depth_analysis);

            if line.contains('<') && !line.contains("</") && !line.contains("/>") {
                if !line.trim_start().starts_with("</") {
                    total_elements += 1;

                    // Update depth distribution
                    *depth_distribution.entry(depth).or_insert(0) += 1;

                    if depth > max_depth {
                        max_depth = depth;
                    }
                }
            }

            if line.contains("<script") {
                text_nodes += 1;
            }
        }

        // Calculate average depth
        let total_depth: usize = depth_distribution.iter().map(|(d, c)| d * c).sum();
        let average_depth = if total_elements > 0 {
            total_depth as f32 / total_elements as f32
        } else {
            0.0
        };

        // Estimate max siblings from common patterns
        max_siblings = tag_counts.values().max().copied().unwrap_or(0);

        DomInfo {
            max_depth,
            total_elements,
            text_nodes,
            average_depth,
            max_siblings,
            depth_distribution,
        }
    }

    /// Count tag occurrences in HTML
    fn count_tags(&self, html: &str) -> HashMap<String, usize> {
        let mut counts = HashMap::new();

        let tags = [
            "div", "span", "p", "a", "ul", "ol", "li", "table", "tr", "td", "th",
            "form", "input", "button", "select", "option", "h1", "h2", "h3", "h4", "h5", "h6",
            "header", "nav", "main", "footer", "section", "article", "aside",
            "img", "video", "audio", "canvas", "svg",
            "script", "style", "link", "meta",
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

    /// Analyze element distribution
    fn analyze_element_distribution(&self, html: &str) -> ElementDistribution {
        let tag_counts = self.count_tags(html);

        let mut semantic_elements = HashMap::new();
        let mut interactive_elements = HashMap::new();
        let mut media_elements = HashMap::new();
        let mut form_elements = HashMap::new();

        // Categorize elements
        for (tag, count) in &tag_counts {
            match tag.as_str() {
                "header" | "nav" | "main" | "footer" | "section" | "article" | "aside" => {
                    semantic_elements.insert(tag.clone(), *count);
                }
                "button" | "a" | "input" | "select" | "textarea" => {
                    interactive_elements.insert(tag.clone(), *count);
                }
                "img" | "video" | "audio" | "canvas" | "svg" => {
                    media_elements.insert(tag.clone(), *count);
                }
                "form" | "input" | "select" | "textarea" | "label" => {
                    form_elements.insert(tag.clone(), *count);
                }
                _ => {}
            }
        }

        // Get top tags
        let mut top_tags: Vec<_> = tag_counts.into_iter().collect();
        top_tags.sort_by(|a, b| b.1.cmp(&a.1));
        top_tags.truncate(10);

        let total_count = tag_counts.values().sum();

        ElementDistribution {
            semantic_elements,
            interactive_elements,
            media_elements,
            form_elements,
            top_tags,
            total_count,
        }
    }

    /// Detect layout type based on HTML structure
    fn detect_layout_type(
        &self,
        html: &str,
        distribution: &ElementDistribution,
    ) -> (LayoutType, f32) {
        let html_lower = html.to_lowercase();

        // Check for specific layout patterns
        let sidebar_score = self.calculate_sidebar_score(html, distribution);
        let grid_score = self.calculate_grid_score(html);
        let flex_score = self.calculate_flex_score(html);

        // Dashboard detection
        if html_lower.contains("dashboard") && html_lower.contains("sidebar") {
            return (LayoutType::Dashboard, 0.9);
        }

        // Magazine layout (lots of sections with varying sizes)
        if distribution.semantic_elements.get("section").copied().unwrap_or(0) > 5
            && distribution.semantic_elements.get("article").copied().unwrap_or(0) > 3
        {
            return (LayoutType::Magazine, 0.75);
        }

        // Grid detection
        if grid_score > 0.7 {
            return (LayoutType::Grid, grid_score);
        }

        // Two column detection
        if sidebar_score > 0.6 {
            return (LayoutType::TwoColumn, sidebar_score);
        }

        // Flexbox detection
        if flex_score > 0.5 {
            return (LayoutType::SplitScreen, flex_score);
        }

        // Check for three column layout
        let columns_count = html_lower.matches("col-").count();
        if columns_count >= 3 {
            return (LayoutType::ThreeColumn, 0.7);
        }

        // Default to single column or unknown
        if distribution.total_count < 50 {
            return (LayoutType::SingleColumn, 0.6);
        }

        (LayoutType::Unknown, 0.5)
    }

    /// Calculate sidebar presence score
    fn calculate_sidebar_score(&self, html: &str, distribution: &ElementDistribution) -> f32 {
        let mut score = 0.0;

        if html.to_lowercase().contains("sidebar") {
            score += 0.4;
        }

        if distribution.semantic_elements.get("aside").copied().unwrap_or(0) > 0 {
            score += 0.3;
        }

        // Check for common sidebar patterns
        let sidebar_patterns = ["left-sidebar", "right-sidebar", ".sidebar", "#sidebar"];
        for pattern in &sidebar_patterns {
            if html.to_lowercase().contains(pattern) {
                score += 0.3;
                break;
            }
        }

        score.min(1.0)
    }

    /// Calculate grid layout score
    fn calculate_grid_score(&self, html: &str) -> f32 {
        let mut score = 0.0;

        if html.contains("display: grid") || html.contains("display:grid") {
            score += 0.5;
        }

        if html.contains("grid-template") {
            score += 0.3;
        }

        // Check for CSS grid classes
        let grid_patterns = ["grid-", "col-grid", "grid-container"];
        for pattern in &grid_patterns {
            if html.contains(pattern) {
                score += 0.2;
                break;
            }
        }

        score.min(1.0)
    }

    /// Calculate flexbox score
    fn calculate_flex_score(&self, html: &str) -> f32 {
        let mut score = 0.0;

        if html.contains("display: flex") || html.contains("display:flex") {
            score += 0.4;
        }

        if html.contains("flex-") {
            score += 0.3;
        }

        // Check for flex utility classes
        let flex_patterns = ["d-flex", "flexbox", "flex-container"];
        for pattern in &flex_patterns {
            if html.contains(pattern) {
                score += 0.2;
                break;
            }
        }

        score.min(1.0)
    }

    /// Detect structural patterns
    fn detect_patterns(&self, html: &str) -> Vec<StructuralPattern> {
        let mut patterns = Vec::new();
        let html_lower = html.to_lowercase();

        // Header pattern
        if html_lower.contains("<header") {
            patterns.push(StructuralPattern {
                name: "Header".to_string(),
                pattern_type: PatternType::Header,
                confidence: 0.9,
                location: "header".to_string(),
                description: "Page header section detected".to_string(),
            });
        }

        // Navigation pattern
        if html_lower.contains("<nav") {
            patterns.push(StructuralPattern {
                name: "Navigation".to_string(),
                pattern_type: PatternType::Navigation,
                confidence: 0.9,
                location: "nav".to_string(),
                description: "Navigation section detected".to_string(),
            });
        }

        // Hero section pattern
        if html_lower.contains("hero") || html_lower.contains("jumbotron") {
            patterns.push(StructuralPattern {
                name: "Hero Section".to_string(),
                pattern_type: PatternType::Hero,
                confidence: 0.8,
                location: ".hero".to_string(),
                description: "Hero/banner section detected".to_string(),
            });
        }

        // Card pattern
        if html_lower.contains("card") && html_lower.contains("-card") {
            patterns.push(StructuralPattern {
                name: "Card Layout".to_string(),
                pattern_type: PatternType::Card,
                confidence: 0.7,
                location: ".card".to_string(),
                description: "Card-based layout detected".to_string(),
            });
        }

        // Grid pattern
        if html.contains("display: grid") || html.contains("grid-template") {
            patterns.push(StructuralPattern {
                name: "CSS Grid".to_string(),
                pattern_type: PatternType::Grid,
                confidence: 0.8,
                location: ".grid".to_string(),
                description: "CSS Grid layout detected".to_string(),
            });
        }

        // Footer pattern
        if html_lower.contains("<footer") {
            patterns.push(StructuralPattern {
                name: "Footer".to_string(),
                pattern_type: PatternType::Footer,
                confidence: 0.9,
                location: "footer".to_string(),
                description: "Page footer detected".to_string(),
            });
        }

        // Modal pattern
        if html_lower.contains("modal") {
            patterns.push(StructuralPattern {
                name: "Modal Dialog".to_string(),
                pattern_type: PatternType::Modal,
                confidence: 0.75,
                location: ".modal".to_string(),
                description: "Modal dialog detected".to_string(),
            });
        }

        patterns
    }

    /// Analyze semantic structure
    fn analyze_semantic_structure(&self, html: &str) -> SemanticStructure {
        let html_lower = html.to_lowercase();

        let has_header = html_lower.contains("<header");
        let has_navigation = html_lower.contains("<nav");
        let has_main = html_lower.contains("<main");
        let has_footer = html_lower.contains("<footer");
        let has_sidebar = html_lower.contains("<aside");
        let has_article = html_lower.contains("<article");
        let has_section = html_lower.contains("<section");
        let has_aside = html_lower.contains("<aside");

        // Analyze heading hierarchy
        let mut heading_levels: HashMap<usize, usize> = HashMap::new();
        for i in 1..=6 {
            let pattern = format!("<h{}>", i);
            let count = html_lower.matches(&pattern).count();
            if count > 0 {
                heading_levels.insert(i, count);
            }
        }

        // Calculate organization score
        let mut org_score = 0.0;
        if has_header {
            org_score += 0.15;
        }
        if has_navigation {
            org_score += 0.15;
        }
        if has_main {
            org_score += 0.2;
        }
        if has_footer {
            org_score += 0.15;
        }
        if !heading_levels.is_empty() {
            org_score += 0.2;
        }
        if has_article || has_section {
            org_score += 0.15;
        }

        SemanticStructure {
            has_header,
            has_navigation,
            has_main,
            has_footer,
            has_sidebar,
            has_article,
            has_section,
            has_aside,
            heading_levels,
            organization_score: org_score,
        }
    }

    /// Calculate complexity metrics
    fn calculate_complexity(
        &self,
        dom_info: &DomInfo,
        distribution: &ElementDistribution,
    ) -> ComplexityMetrics {
        // Structural complexity based on depth and width
        let depth_factor = (dom_info.max_depth as f32 / 20.0).min(1.0);
        let width_factor = (dom_info.max_siblings as f32 / 30.0).min(1.0);
        let structural_complexity = (depth_factor * 0.6 + width_factor * 0.4) * 0.7;

        // Nesting complexity
        let avg_depth_factor = (dom_info.average_depth / 10.0).min(1.0);
        let nesting_complexity = avg_depth_factor * 0.5;

        // Element variety
        let variety = (distribution.top_tags.len() as f32 / 15.0).min(1.0);
        let element_variety = variety * 0.3;

        // Overall complexity
        let total_complexity = (structural_complexity + nesting_complexity + element_variety).min(1.0);

        let complexity_rating = if total_complexity < 0.3 {
            ComplexityLevel::Simple
        } else if total_complexity < 0.5 {
            ComplexityLevel::Moderate
        } else if total_complexity < 0.7 {
            ComplexityLevel::Complex
        } else {
            ComplexityLevel::VeryComplex
        };

        // Estimated render time (very rough estimate)
        let estimated_render_time = dom_info.total_elements as f32 * 0.1;

        ComplexityMetrics {
            structural_complexity,
            nesting_complexity,
            element_variety,
            complexity_rating,
            estimated_render_time,
        }
    }
}

impl Default for StructureAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = StructureAnalyzer::new();
        assert!(analyzer.config.min_confidence > 0.0);
    }

    #[test]
    fn test_tag_counting() {
        let analyzer = StructureAnalyzer::new();
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

        let counts = analyzer.count_tags(html);

        assert!(counts.get("div").copied().unwrap_or(0) > 0);
        assert!(counts.get("header").copied().unwrap_or(0) > 0);
    }

    #[test]
    fn test_element_distribution() {
        let analyzer = StructureAnalyzer::new();
        let html = r#"
            <header></header>
            <nav></nav>
            <main>
                <section></section>
                <article></article>
            </main>
            <footer></footer>
        "#;

        let distribution = analyzer.analyze_element_distribution(html);

        assert!(distribution.semantic_elements.len() > 0);
        assert!(distribution.total_count > 0);
    }

    #[test]
    fn test_layout_detection() {
        let analyzer = StructureAnalyzer::new();

        // Test dashboard layout
        let dashboard_html = r#"
            <div class="dashboard">
                <aside class="sidebar">Sidebar</aside>
                <main>Content</main>
            </div>
        "#;
        let distribution = analyzer.analyze_element_distribution(dashboard_html);
        let (layout, confidence) = analyzer.detect_layout_type(dashboard_html, &distribution);

        assert_eq!(layout, LayoutType::Dashboard);
        assert!(confidence > 0.8);
    }

    #[test]
    fn test_pattern_detection() {
        let analyzer = StructureAnalyzer::new();
        let html = r#"
            <header></header>
            <nav></nav>
            <div class="hero"></div>
            <footer></footer>
        "#;

        let patterns = analyzer.detect_patterns(html);

        assert!(patterns.iter().any(|p| p.name == "Header"));
        assert!(patterns.iter().any(|p| p.name == "Hero Section"));
    }

    #[test]
    fn test_semantic_analysis() {
        let analyzer = StructureAnalyzer::new();
        let html = r#"
            <header>Header</header>
            <nav>Nav</nav>
            <main>
                <section><article>Article</article></section>
            </main>
            <footer>Footer</footer>
        "#;

        let semantic = analyzer.analyze_semantic_structure(html);

        assert!(semantic.has_header);
        assert!(semantic.has_navigation);
        assert!(semantic.has_main);
        assert!(semantic.has_footer);
        assert!(semantic.has_section);
    }

    #[test]
    fn test_complexity_calculation() {
        let analyzer = StructureAnalyzer::new();
        let dom_info = DomInfo {
            max_depth: 10,
            total_elements: 100,
            text_nodes: 20,
            average_depth: 3.5,
            max_siblings: 15,
            depth_distribution: HashMap::new(),
        };
        let distribution = analyzer.analyze_element_distribution("<div></div>");

        let complexity = analyzer.calculate_complexity(&dom_info, &distribution);

        assert!(complexity.structural_complexity > 0.0);
        assert!(complexity.complexity_rating != ComplexityLevel::Unknown);
    }

    #[tokio::test]
    async fn test_full_analysis() {
        let analyzer = StructureAnalyzer::new();

        let page = crate::data_models::PageContent {
            url: "https://example.com".to_string(),
            html: r#"
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
            "#.to_string(),
            ..Default::default()
        };

        let structure = analyzer.analyze(&page).await;

        assert!(structure.layout_confidence > 0.0);
        assert!(structure.dom_info.total_elements > 0);
        assert!(!structure.patterns.is_empty());
    }
}
