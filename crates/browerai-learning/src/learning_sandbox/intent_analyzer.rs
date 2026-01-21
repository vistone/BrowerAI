//! Intent Analyzer
//!
//! Analyzes web page content to understand the website's intent, type, and features.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::data_models::PageContent;

/// Website intent analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebsiteIntent {
    /// Primary website type
    pub website_type: String,

    /// Website type confidence score (0.0 - 1.0)
    pub confidence: f32,

    /// Core features identified
    pub core_features: Vec<String>,

    /// Target audience description
    pub target_audience: String,

    /// Design style analysis
    pub design_style: DesignStyle,

    /// Page structure analysis
    pub structure: PageStructure,

    /// Business model inference
    pub business_model: String,

    /// Detailed type scores
    pub type_scores: HashMap<String, f32>,
}

impl Default for WebsiteIntent {
    fn default() -> Self {
        Self {
            website_type: "unknown".to_string(),
            confidence: 0.0,
            core_features: Vec::new(),
            target_audience: "unknown".to_string(),
            design_style: DesignStyle::default(),
            structure: PageStructure::default(),
            business_model: "unknown".to_string(),
            type_scores: HashMap::new(),
        }
    }
}

/// Design style analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignStyle {
    /// Formality level (0.0 = casual, 1.0 = formal)
    pub formality: f32,

    /// Colorfulness level (0.0 = minimal, 1.0 = colorful)
    pub colorfulness: f32,

    /// Minimalism level (0.0 = complex, 1.0 = minimal)
    pub minimalism: f32,

    /// Modernity level (0.0 = traditional, 1.0 = modern)
    pub modernity: f32,

    /// Primary color palette (if detected)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_colors: Option<Vec<String>>,

    /// Layout type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layout_type: Option<String>,
}

impl Default for DesignStyle {
    fn default() -> Self {
        Self {
            formality: 0.5,
            colorfulness: 0.5,
            minimalism: 0.5,
            modernity: 0.5,
            primary_colors: None,
            layout_type: None,
        }
    }
}

/// Page structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageStructure {
    /// Has header section
    pub has_header: bool,

    /// Has navigation
    pub has_navigation: bool,

    /// Has sidebar
    pub has_sidebar: bool,

    /// Has main content area
    pub has_main_content: bool,

    /// Has footer
    pub has_footer: bool,

    /// Layout type
    pub layout_type: LayoutType,

    /// Estimated content sections
    pub section_count: usize,

    /// Estimated page complexity
    pub complexity: ComplexityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum LayoutType {
    #[serde(rename = "single-column")]
    SingleColumn,
    #[serde(rename = "two-column")]
    TwoColumn,
    #[serde(rename = "three-column")]
    ThreeColumn,
    #[serde(rename = "grid")]
    Grid,
    #[serde(rename = "masonry")]
    Masonry,
    #[serde(rename = "magazine")]
    Magazine,
    #[serde(rename = "unknown")]
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum ComplexityLevel {
    #[serde(rename = "simple")]
    #[default]
    Simple,
    #[serde(rename = "moderate")]
    Moderate,
    #[serde(rename = "complex")]
    Complex,
    #[serde(rename = "very-complex")]
    VeryComplex,
}

impl Default for PageStructure {
    fn default() -> Self {
        Self {
            has_header: false,
            has_navigation: false,
            has_sidebar: false,
            has_main_content: true,
            has_footer: false,
            layout_type: LayoutType::Unknown,
            section_count: 0,
            complexity: ComplexityLevel::Simple,
        }
    }
}

/// Intent Analyzer Configuration
#[derive(Debug, Clone)]
pub struct IntentAnalyzerConfig {
    /// Minimum confidence threshold
    pub min_confidence: f32,

    /// Enable feature extraction
    pub extract_features: bool,

    /// Enable design analysis
    pub analyze_design: bool,
}

impl Default for IntentAnalyzerConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            extract_features: true,
            analyze_design: true,
        }
    }
}

/// Intent Analyzer
///
/// Analyzes web page content to determine website intent, type, and features.
#[derive(Debug, Clone)]
pub struct IntentAnalyzer {
    config: IntentAnalyzerConfig,

    // Keywords for different website types
    ecommerce_keywords: HashSet<&'static str>,
    blog_keywords: HashSet<&'static str>,
    social_keywords: HashSet<&'static str>,
    docs_keywords: HashSet<&'static str>,
    corporate_keywords: HashSet<&'static str>,
    portfolio_keywords: HashSet<&'static str>,
    forum_keywords: HashSet<&'static str>,
    news_keywords: HashSet<&'static str>,

    // Design-related keywords
    modern_keywords: HashSet<&'static str>,
    minimal_keywords: HashSet<&'static str>,
    formal_keywords: HashSet<&'static str>,
}

impl IntentAnalyzer {
    /// Create a new intent analyzer with default configuration
    pub fn new() -> Self {
        Self::with_config(IntentAnalyzerConfig::default())
    }

    /// Create an intent analyzer with custom configuration
    pub fn with_config(config: IntentAnalyzerConfig) -> Self {
        Self {
            config,
            ecommerce_keywords: HashSet::from([
                "shop",
                "cart",
                "product",
                "price",
                "buy",
                "checkout",
                "order",
                "payment",
                "shipping",
                "discount",
                "sale",
                "add to cart",
                "shopping",
                "estore",
                "e-commerce",
                "product catalog",
                "online store",
            ]),
            blog_keywords: HashSet::from([
                "article",
                "post",
                "author",
                "date",
                "comment",
                "blog",
                "read",
                "subscribe",
                "category",
                "tag",
                "blog post",
                "published",
                "reading time",
            ]),
            social_keywords: HashSet::from([
                "profile",
                "message",
                "friend",
                "follow",
                "post",
                "share",
                "like",
                "notification",
                "feed",
                "timeline",
                "social media",
                "community",
                "connect",
            ]),
            docs_keywords: HashSet::from([
                "documentation",
                "docs",
                "guide",
                "tutorial",
                "api",
                "reference",
                "install",
                "getting started",
                "manual",
                "help center",
                "knowledge base",
            ]),
            corporate_keywords: HashSet::from([
                "company",
                "about us",
                "services",
                "contact",
                "team",
                "careers",
                "investors",
                "newsroom",
                "corporate",
                "enterprise",
                "business",
            ]),
            portfolio_keywords: HashSet::from([
                "portfolio",
                "work",
                "project",
                "gallery",
                "showcase",
                "creative",
                "design",
                "photography",
                "case study",
                "client",
                "testimonials",
            ]),
            forum_keywords: HashSet::from([
                "forum",
                "discussion",
                "thread",
                "reply",
                "topic",
                "community",
                "member",
                "register",
                "bbcode",
                "moderator",
                "posting",
            ]),
            news_keywords: HashSet::from([
                "news",
                "headline",
                "breaking",
                "report",
                "journalism",
                "press release",
                "media",
                "current events",
                "update",
                "announcement",
            ]),
            modern_keywords: HashSet::from([
                "modern",
                "contemporary",
                "sleek",
                "clean",
                "responsive",
                "minimal",
                "flat design",
            ]),
            minimal_keywords: HashSet::from([
                "minimal",
                "simple",
                "clean",
                "lightweight",
                "essential",
                "stripped down",
                "bare bones",
            ]),
            formal_keywords: HashSet::from([
                "professional",
                "business",
                "corporate",
                "formal",
                "official",
                "enterprise",
            ]),
        }
    }

    /// Analyze page content to determine intent
    pub fn analyze(&self, page: &PageContent) -> WebsiteIntent {
        let text = self.extract_text(page);
        let text_lower = text.to_lowercase();
        let words: HashSet<&str> = text_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .collect();

        // Calculate type scores
        let mut type_scores: HashMap<String, f32> = HashMap::new();

        type_scores.insert(
            "e-commerce".to_string(),
            self.calculate_type_score(&words, &self.ecommerce_keywords, 40.0),
        );
        type_scores.insert(
            "blog".to_string(),
            self.calculate_type_score(&words, &self.blog_keywords, 30.0),
        );
        type_scores.insert(
            "social".to_string(),
            self.calculate_type_score(&words, &self.social_keywords, 25.0),
        );
        type_scores.insert(
            "documentation".to_string(),
            self.calculate_type_score(&words, &self.docs_keywords, 25.0),
        );
        type_scores.insert(
            "corporate".to_string(),
            self.calculate_type_score(&words, &self.corporate_keywords, 20.0),
        );
        type_scores.insert(
            "portfolio".to_string(),
            self.calculate_type_score(&words, &self.portfolio_keywords, 20.0),
        );
        type_scores.insert(
            "forum".to_string(),
            self.calculate_type_score(&words, &self.forum_keywords, 20.0),
        );
        type_scores.insert(
            "news".to_string(),
            self.calculate_type_score(&words, &self.news_keywords, 20.0),
        );

        // Find best match
        let (website_type, max_score) = type_scores
            .iter()
            .max_by_key(|(_, score)| **score as i32)
            .map(|(t, s)| (t.clone(), *s))
            .unwrap_or_else(|| ("unknown".to_string(), 0.0));

        // Calculate confidence (normalized score)
        let confidence = (max_score / 100.0).min(1.0);

        // Extract core features
        let core_features = self.extract_features(&text_lower, &words);

        // Analyze design style
        let design_style = if self.config.analyze_design {
            self.analyze_design_style(&text_lower, page)
        } else {
            DesignStyle::default()
        };

        // Analyze page structure
        let structure = self.analyze_structure(page);

        // Infer target audience
        let target_audience = self.infer_audience(&website_type, &core_features);

        // Infer business model
        let business_model = self.infer_business_model(&website_type);

        WebsiteIntent {
            website_type,
            confidence,
            core_features,
            target_audience,
            design_style,
            structure,
            business_model,
            type_scores,
        }
    }

    /// Calculate type score based on keyword matching
    fn calculate_type_score(
        &self,
        words: &HashSet<&str>,
        keywords: &HashSet<&'static str>,
        max_score: f32,
    ) -> f32 {
        let matches = keywords.iter().filter(|kw| words.contains(*kw)).count() as f32;

        // Normalize by number of keywords
        let normalized = matches / keywords.len() as f32;

        // Apply weighting
        (normalized * max_score).min(max_score)
    }

    /// Extract text content from page
    fn extract_text(&self, page: &PageContent) -> String {
        let mut text = String::new();

        // Add HTML content
        text.push_str(&page.html);
        text.push(' ');

        // Use metadata
        if let Some(title) = &page.metadata.title {
            text.push_str(title);
            text.push(' ');
        }

        if let Some(desc) = &page.metadata.description {
            text.push_str(desc);
            text.push(' ');
        }

        // Add inline content
        text.push_str(&page.all_css());
        text.push_str(&page.all_js());

        text
    }

    /// Extract core features from content
    fn extract_features(&self, text: &str, _words: &HashSet<&str>) -> Vec<String> {
        let mut features = Vec::new();

        // E-commerce features
        if text.contains("shopping cart") || text.contains("add to cart") {
            features.push("shopping_cart".to_string());
        }
        if text.contains("checkout") || text.contains("payment") {
            features.push("checkout".to_string());
        }
        if text.contains("product catalog") || text.contains("product list") {
            features.push("product_catalog".to_string());
        }
        if text.contains("search") && (text.contains("products") || text.contains("items")) {
            features.push("search".to_string());
        }
        if text.contains("user account") || text.contains("my account") {
            features.push("user_account".to_string());
        }

        // Blog features
        if text.contains("article") && text.contains("read") {
            features.push("article_reader".to_string());
        }
        if text.contains("comment") {
            features.push("comments".to_string());
        }
        if text.contains("subscribe") || text.contains("newsletter") {
            features.push("newsletter".to_string());
        }

        // Social features
        if text.contains("share") {
            features.push("sharing".to_string());
        }
        if text.contains("like") || text.contains("follow") {
            features.push("social_interaction".to_string());
        }

        // Authentication
        if text.contains("sign in") || text.contains("login") {
            features.push("authentication".to_string());
        }
        if text.contains("sign up") || text.contains("register") {
            features.push("registration".to_string());
        }

        // Contact
        if text.contains("contact") || text.contains("contact us") {
            features.push("contact".to_string());
        }

        features
    }

    /// Analyze design style
    fn analyze_design_style(&self, text: &str, _page: &PageContent) -> DesignStyle {
        let mut formality = 0.5;
        let colorfulness = 0.5;
        let mut minimalism = 0.5;
        let mut modernity = 0.5;

        // Check for formal keywords
        let formal_count = self
            .formal_keywords
            .iter()
            .filter(|kw| text.contains(*kw))
            .count();
        if formal_count > 2 {
            formality = 0.8;
        } else if formal_count > 0 {
            formality = 0.6;
        }

        // Check for minimalism
        let minimal_count = self
            .minimal_keywords
            .iter()
            .filter(|kw| text.contains(*kw))
            .count();
        if minimal_count > 1 {
            minimalism = 0.8;
        } else if minimal_count > 0 {
            minimalism = 0.6;
        }

        // Check for modernity
        let modern_count = self
            .modern_keywords
            .iter()
            .filter(|kw| text.contains(*kw))
            .count();
        if modern_count > 1 {
            modernity = 0.8;
        } else if modern_count > 0 {
            modernity = 0.6;
        }

        DesignStyle {
            formality,
            colorfulness,
            minimalism,
            modernity,
            primary_colors: None,
            layout_type: None,
        }
    }

    /// Analyze page structure
    fn analyze_structure(&self, page: &PageContent) -> PageStructure {
        let html_lower = page.html.to_lowercase();

        let has_header = html_lower.contains("<header");
        let has_navigation = html_lower.contains("<nav") || html_lower.contains("navigation");
        let has_sidebar = html_lower.contains("<aside");
        let has_main_content = html_lower.contains("<main");
        let has_footer = html_lower.contains("<footer");

        // Estimate section count
        let section_count =
            html_lower.matches("<section").count() + html_lower.matches("<div").count() / 10;

        // Determine layout type
        let layout_type = match (has_sidebar, has_navigation) {
            (true, true) => LayoutType::TwoColumn,
            (false, true) => LayoutType::SingleColumn,
            (true, false) => LayoutType::TwoColumn,
            _ => {
                if section_count > 10 {
                    LayoutType::Grid
                } else {
                    LayoutType::SingleColumn
                }
            }
        };

        // Determine complexity
        let complexity = if section_count > 50 {
            ComplexityLevel::VeryComplex
        } else if section_count > 20 {
            ComplexityLevel::Complex
        } else if section_count > 5 {
            ComplexityLevel::Moderate
        } else {
            ComplexityLevel::Simple
        };

        PageStructure {
            has_header,
            has_navigation,
            has_sidebar,
            has_main_content,
            has_footer,
            layout_type,
            section_count,
            complexity,
        }
    }

    /// Infer target audience
    fn infer_audience(&self, website_type: &str, features: &[String]) -> String {
        match website_type {
            "e-commerce" => {
                if features.contains(&"user_account".to_string()) {
                    "returning_customers".to_string()
                } else {
                    "general_consumers".to_string()
                }
            }
            "blog" | "news" => "readers".to_string(),
            "documentation" => "developers".to_string(),
            "social" => "community_members".to_string(),
            "corporate" => "business_professionals".to_string(),
            _ => "general_audience".to_string(),
        }
    }

    /// Infer business model
    fn infer_business_model(&self, website_type: &str) -> String {
        match website_type {
            "e-commerce" => "B2C_retail".to_string(),
            "blog" => {
                if website_type.contains("news") {
                    "advertising".to_string()
                } else {
                    "content_subscription".to_string()
                }
            }
            "documentation" => "B2B_or_open_source".to_string(),
            "social" => "advertising_or_data".to_string(),
            "corporate" => "B2B".to_string(),
            _ => "unknown".to_string(),
        }
    }
}

impl Default for IntentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = IntentAnalyzer::new();
        assert!(!analyzer.ecommerce_keywords.is_empty());
        assert!(!analyzer.blog_keywords.is_empty());
    }

    #[test]
    fn test_ecommerce_detection() {
        let analyzer = IntentAnalyzer::new();

        let page = PageContent::new(
            "https://shop.example.com".to_string(),
            r#"
            <html>
            <head><title>Shop - Best Products</title></head>
            <body>
                <p>Welcome to our online shop. Browse our products, add to cart, and checkout.</p>
                <p>Free shipping on orders over $50.</p>
                <p>Shop now!</p>
            </body>
            </html>
            "#
            .to_string(),
            HashMap::new(),
        );

        let intent = analyzer.analyze(&page);

        assert_eq!(intent.website_type, "e-commerce");
        assert!(intent.confidence > 0.0);
        assert!(intent.core_features.contains(&"shopping_cart".to_string()));
    }

    #[test]
    fn test_blog_detection() {
        let analyzer = IntentAnalyzer::new();

        let page = PageContent::new(
            "https://blog.example.com".to_string(),
            r#"
            <html>
            <head><title>My Blog - Latest Posts</title></head>
            <body>
                <p>Read our latest article about technology.</p>
                <p>Published on January 15, 2024 by the author.</p>
                <p>Leave a comment below.</p>
                <p>Subscribe to our newsletter.</p>
            </body>
            </html>
            "#
            .to_string(),
            HashMap::new(),
        );

        let intent = analyzer.analyze(&page);

        assert_eq!(intent.website_type, "blog");
        assert!(intent.core_features.contains(&"article_reader".to_string()));
    }

    #[test]
    fn test_structure_analysis() {
        let analyzer = IntentAnalyzer::new();

        let page = PageContent::new(
            "https://example.com".to_string(),
            r#"
            <html>
            <head><title>Test</title></head>
            <body>
                <header>Header</header>
                <nav>Navigation</nav>
                <main>Main Content</main>
                <aside>Sidebar</aside>
                <footer>Footer</footer>
            </body>
            </html>
            "#
            .to_string(),
            HashMap::new(),
        );

        let intent = analyzer.analyze(&page);

        assert!(intent.structure.has_header);
        assert!(intent.structure.has_navigation);
        assert!(intent.structure.has_sidebar);
        assert!(intent.structure.has_footer);
        assert_eq!(intent.structure.layout_type, LayoutType::TwoColumn);
    }

    #[test]
    fn test_unknown_page() {
        let analyzer = IntentAnalyzer::new();

        let page = PageContent::new(
            "https://example.com".to_string(),
            r#"<html><head><title>Test</title></head><body>Hello</body></html>"#.to_string(),
            HashMap::new(),
        );

        let intent = analyzer.analyze(&page);

        // Should still produce a valid intent
        assert!(!intent.website_type.is_empty());
        assert!(intent.type_scores.values().all(|s| *s >= 0.0));
    }
}
