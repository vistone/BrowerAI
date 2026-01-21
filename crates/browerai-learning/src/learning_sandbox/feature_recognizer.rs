//! Feature Recognizer
//!
//! Identifies and extracts features from web page content including forms,
//! navigation, interactive elements, and content sections.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Feature map containing all identified page features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMap {
    /// All identified features
    pub features: Vec<Feature>,

    /// Form fields found on the page
    pub form_fields: Vec<FormField>,

    /// Navigation items found on the page
    pub navigation_items: Vec<NavItem>,

    /// Interactive elements found on the page
    pub interactive_elements: Vec<InteractiveElement>,

    /// Content sections found on the page
    pub content_sections: Vec<ContentSection>,

    /// Metadata about the feature extraction
    pub metadata: FeatureMetadata,
}

/// A single feature identified on the page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature {
    /// Feature name/type
    pub name: String,

    /// Feature category
    pub category: FeatureCategory,

    /// Location in the page (CSS selector or approximate position)
    pub location: String,

    /// Confidence of detection (0.0 - 1.0)
    pub confidence: f32,

    /// Additional metadata about the feature
    pub metadata: HashMap<String, String>,
}

/// Categories of features
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FeatureCategory {
    Navigation,
    Content,
    Form,
    Interactive,
    Media,
    Social,
    Ecommerce,
    Authentication,
    Search,
    Filtering,
    Pagination,
    Modal,
    Animation,
    Responsive,
    Accessibility,
    Analytics,
    Other,
}

/// A form field identified on the page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormField {
    /// Field name/identifier
    pub name: String,

    /// Field type (text, email, password, etc.)
    pub field_type: String,

    /// HTML input type
    pub input_type: String,

    /// Whether the field is required
    pub required: bool,

    /// Field label (if available)
    pub label: Option<String>,

    /// Placeholder text
    pub placeholder: Option<String>,

    /// CSS selector for the field
    pub selector: String,

    /// Validation patterns (if any)
    pub validation_patterns: Vec<String>,
}

/// A navigation item identified on the page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavItem {
    /// Navigation item text
    pub text: String,

    /// Link URL
    pub url: String,

    /// Navigation type
    pub nav_type: NavType,

    /// Parent navigation item (if nested)
    pub parent: Option<String>,

    /// CSS selector
    pub selector: String,

    /// Whether it's a dropdown/has children
    pub has_children: bool,
}

/// Types of navigation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NavType {
    Primary,
    Secondary,
    Footer,
    Breadcrumb,
    Pagination,
    Social,
    Utility,
    Menu,
    Tab,
}

/// An interactive element on the page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveElement {
    /// Element type
    pub element_type: InteractiveType,

    /// Element text/description
    pub text: Option<String>,

    /// CSS selector
    pub selector: String,

    /// Associated actions
    pub actions: Vec<String>,

    /// Event handlers attached
    pub event_handlers: Vec<String>,
}

/// Types of interactive elements
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InteractiveType {
    Button,
    Link,
    Tab,
    Accordion,
    Carousel,
    Slider,
    Tooltip,
    Dropdown,
    Toggle,
    Modal,
    ScrollSpy,
    Lightbox,
    Rating,
    Counter,
    Filter,
    Sorter,
}

/// A content section on the page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSection {
    /// Section type
    pub section_type: SectionType,

    /// Section heading (if any)
    pub heading: Option<String>,

    /// Approximate position (order on page)
    pub position: usize,

    /// CSS selector
    pub selector: String,

    /// Content summary (first few elements)
    pub content_summary: String,

    /// Number of child elements
    pub child_count: usize,
}

/// Types of content sections
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SectionType {
    Hero,
    Feature,
    Testimonial,
    Pricing,
    FAQ,
    BlogPost,
    ProductList,
    Gallery,
    Team,
    Contact,
    About,
    Footer,
    Sidebar,
    CallToAction,
    Newsletter,
    Banner,
    Other,
}

/// Metadata about feature extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetadata {
    /// URL analyzed
    pub url: String,

    /// Extraction timestamp
    pub extracted_at: chrono::DateTime<chrono::Utc>,

    /// Time taken for extraction
    pub extraction_time_ms: u64,

    /// Number of elements scanned
    pub elements_scanned: usize,

    /// Features extracted
    pub features_extracted: usize,
}

/// Feature Recognizer Configuration
#[derive(Debug, Clone)]
pub struct FeatureRecognizerConfig {
    /// Minimum confidence threshold
    pub min_confidence: f32,

    /// Enable deep feature analysis
    pub deep_analysis: bool,

    /// Maximum elements to scan
    pub max_elements: usize,

    /// Enable form detection
    pub detect_forms: bool,

    /// Enable navigation detection
    pub detect_navigation: bool,

    /// Enable interactive element detection
    pub detect_interactive: bool,
}

impl Default for FeatureRecognizerConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            deep_analysis: true,
            max_elements: 1000,
            detect_forms: true,
            detect_navigation: true,
            detect_interactive: true,
        }
    }
}

/// Feature Recognizer
///
/// Identifies and extracts features from web page content.
#[derive(Debug, Clone)]
pub struct FeatureRecognizer {
    config: FeatureRecognizerConfig,
}

impl FeatureRecognizer {
    /// Create a new feature recognizer
    pub fn new() -> Self {
        Self::with_config(FeatureRecognizerConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: FeatureRecognizerConfig) -> Self {
        Self { config }
    }

    /// Recognize features in page content
    ///
    /// # Arguments
    /// * `page` - Page content to analyze
    ///
    /// # Returns
    /// * `FeatureMap` with all identified features
    pub async fn recognize(&self, page: &crate::data_models::PageContent) -> FeatureMap {
        let start_time = chrono::Utc::now();

        let mut features = Vec::new();
        let mut form_fields = Vec::new();
        let mut navigation_items = Vec::new();
        let mut interactive_elements = Vec::new();
        let mut content_sections = Vec::new();

        let html_lower = page.html.to_lowercase();

        // Detect forms and form fields
        if self.config.detect_forms {
            self.analyze_forms(&page.html, &mut form_fields, &mut features);
        }

        // Detect navigation
        if self.config.detect_navigation {
            self.analyze_navigation(&page.html, &mut navigation_items, &mut features);
        }

        // Detect interactive elements
        if self.config.detect_interactive {
            self.analyze_interactive_elements(&page.html, &mut interactive_elements, &mut features);
        }

        // Analyze content sections
        self.analyze_content_sections(&page.html, &mut content_sections, &mut features);

        // Detect specific features based on content patterns
        self.detect_specific_features(&html_lower, &mut features);

        let end_time = chrono::Utc::now();
        let extraction_time_ms = (end_time - start_time).num_milliseconds() as u64;

        FeatureMap {
            features,
            form_fields,
            navigation_items,
            interactive_elements,
            content_sections,
            metadata: FeatureMetadata {
                url: page.url.clone(),
                extracted_at: chrono::Utc::now(),
                extraction_time_ms,
                elements_scanned: page.html.len(),
                features_extracted: 0,
            },
        }
    }

    /// Analyze forms in HTML content
    fn analyze_forms(
        &self,
        html: &str,
        form_fields: &mut Vec<FormField>,
        features: &mut Vec<Feature>,
    ) {
        // Simple pattern-based form detection
        // In a full implementation, this would parse the HTML properly

        // Detect form elements
        let input_patterns = [
            ("search", r#"<input[^>]*type=["']?search["']?"#),
            ("email", r#"<input[^>]*type=["']?email["']?"#),
            ("password", r#"<input[^>]*type=["']?password["']?"#),
            ("text", r#"<input[^>]*type=["']?text["']?"#),
            ("checkbox", r#"<input[^>]*type=["']?checkbox["']?"#),
            ("radio", r#"<input[^>]*type=["']?radio["']?"#),
            ("submit", r#"<input[^>]*type=["']?submit["']?"#),
            ("button", r#"<button[^>]*"#),
            ("textarea", r#"<textarea[^>]*"#),
            ("select", r#"<select[^>]*"#),
        ];

        for (field_type, pattern) in &input_patterns {
            let matches: Vec<_> = html.matches(pattern).collect();
            if !matches.is_empty() {
                form_fields.push(FormField {
                    name: field_type.to_string(),
                    field_type: field_type.to_string(),
                    input_type: field_type.to_string(),
                    required: matches.iter().any(|m| m.contains("required")),
                    label: None,
                    placeholder: None,
                    selector: format!("input[type={}]", field_type),
                    validation_patterns: Vec::new(),
                });
            }
        }

        // Add form feature
        if html.contains("<form") {
            features.push(Feature {
                name: "Form".to_string(),
                category: FeatureCategory::Form,
                location: "form".to_string(),
                confidence: 0.9,
                metadata: HashMap::new(),
            });
        }
    }

    /// Analyze navigation elements
    fn analyze_navigation(
        &self,
        html: &str,
        navigation_items: &mut Vec<NavItem>,
        features: &mut Vec<Feature>,
    ) {
        // Detect navigation elements
        if html.contains("<nav") {
            let nav_type = if html.contains("navbar") {
                NavType::Primary
            } else if html.contains("footer") {
                NavType::Footer
            } else {
                NavType::Primary
            };

            features.push(Feature {
                name: "Navigation".to_string(),
                category: FeatureCategory::Navigation,
                location: "nav".to_string(),
                confidence: 0.9,
                metadata: HashMap::new(),
            });

            // Extract navigation links
            let link_pattern = r#"<a[^>]*href=["']([^"']*)["'][^>]*>([^<]*)</a>"#;
            let links: Vec<_> = html.matches(link_pattern).collect();

            for link in links.iter().take(20) {
                navigation_items.push(NavItem {
                    text: "Link".to_string(),
                    url: "#".to_string(),
                    nav_type,
                    parent: None,
                    selector: "a".to_string(),
                    has_children: false,
                });
            }
        }

        // Detect breadcrumbs
        if html.contains("breadcrumb") {
            features.push(Feature {
                name: "Breadcrumb Navigation".to_string(),
                category: FeatureCategory::Navigation,
                location: ".breadcrumb".to_string(),
                confidence: 0.85,
                metadata: HashMap::new(),
            });
        }

        // Detect pagination
        if html.contains("pagination") || html.contains("page-numbers") {
            features.push(Feature {
                name: "Pagination".to_string(),
                category: FeatureCategory::Pagination,
                location: ".pagination".to_string(),
                confidence: 0.8,
                metadata: HashMap::new(),
            });
        }
    }

    /// Analyze interactive elements
    fn analyze_interactive_elements(
        &self,
        html: &str,
        interactive_elements: &mut Vec<InteractiveElement>,
        features: &mut Vec<Feature>,
    ) {
        // Detect modal dialogs
        if html.contains("modal") || html.contains("dialog") {
            interactive_elements.push(InteractiveElement {
                element_type: InteractiveType::Modal,
                text: Some("Modal Dialog".to_string()),
                selector: ".modal".to_string(),
                actions: vec!["open".to_string(), "close".to_string()],
                event_handlers: vec!["click".to_string(), "shown.bs.modal".to_string()],
            });

            features.push(Feature {
                name: "Modal Dialog".to_string(),
                category: FeatureCategory::Modal,
                location: ".modal".to_string(),
                confidence: 0.85,
                metadata: HashMap::new(),
            });
        }

        // Detect tabs
        if html.contains("tab") && html.contains("nav-tab") {
            interactive_elements.push(InteractiveElement {
                element_type: InteractiveType::Tab,
                text: Some("Tab Navigation".to_string()),
                selector: ".nav-tabs".to_string(),
                actions: vec!["activate".to_string(), "deactivate".to_string()],
                event_handlers: vec!["shown.bs.tab".to_string()],
            });

            features.push(Feature {
                name: "Tab Navigation".to_string(),
                category: FeatureCategory::Interactive,
                location: ".nav-tabs".to_string(),
                confidence: 0.8,
                metadata: HashMap::new(),
            });
        }

        // Detect carousels/sliders
        if html.contains("carousel") || html.contains("slider") || html.contains("owl-carousel") {
            interactive_elements.push(InteractiveElement {
                element_type: InteractiveType::Carousel,
                text: Some("Carousel".to_string()),
                selector: ".carousel".to_string(),
                actions: vec!["next".to_string(), "prev".to_string(), "slide".to_string()],
                event_handlers: vec!["slide.bs.carousel".to_string()],
            });

            features.push(Feature {
                name: "Carousel/Slider".to_string(),
                category: FeatureCategory::Interactive,
                location: ".carousel".to_string(),
                confidence: 0.8,
                metadata: HashMap::new(),
            });
        }

        // Detect accordions
        if html.contains("accordion") || html.contains("collapse") {
            interactive_elements.push(InteractiveElement {
                element_type: InteractiveType::Accordion,
                text: Some("Accordion".to_string()),
                selector: ".accordion".to_string(),
                actions: vec!["expand".to_string(), "collapse".to_string()],
                event_handlers: vec!["shown.bs.collapse".to_string()],
            });

            features.push(Feature {
                name: "Accordion".to_string(),
                category: FeatureCategory::Interactive,
                location: ".accordion".to_string(),
                confidence: 0.75,
                metadata: HashMap::new(),
            });
        }

        // Detect tooltips
        if html.contains("tooltip") {
            features.push(Feature {
                name: "Tooltip".to_string(),
                category: FeatureCategory::Interactive,
                location: "[data-toggle=tooltip]".to_string(),
                confidence: 0.7,
                metadata: HashMap::new(),
            });
        }

        // Detect dropdowns
        if html.contains("dropdown-menu") {
            interactive_elements.push(InteractiveElement {
                element_type: InteractiveType::Dropdown,
                text: Some("Dropdown Menu".to_string()),
                selector: ".dropdown-menu".to_string(),
                actions: vec!["toggle".to_string(), "show".to_string(), "hide".to_string()],
                event_handlers: vec!["show.bs.dropdown".to_string()],
            });
        }
    }

    /// Analyze content sections
    fn analyze_content_sections(
        &self,
        html: &str,
        content_sections: &mut Vec<ContentSection>,
        features: &mut Vec<Feature>,
    ) {
        let section_patterns = [
            (SectionType::Hero, vec!["hero", "jumbotron", "banner"]),
            (SectionType::Feature, vec!["feature", "service"]),
            (SectionType::Testimonial, vec!["testimonial", "review"]),
            (SectionType::Pricing, vec!["pricing", "price"]),
            (SectionType::FAQ, vec!["faq", "question"]),
            (SectionType::Team, vec!["team", "member"]),
            (SectionType::Contact, vec!["contact"]),
            (SectionType::About, vec!["about"]),
            (SectionType::Newsletter, vec!["newsletter", "subscribe"]),
            (SectionType::CallToAction, vec!["cta", "call-to-action"]),
        ];

        for (section_type, keywords) in &section_patterns {
            for keyword in keywords {
                if html.contains(keyword) {
                    let selector = format!(".{}", keyword);
                    content_sections.push(ContentSection {
                        section_type: section_type.clone(),
                        heading: Some(keyword.to_string()),
                        position: content_sections.len(),
                        selector: selector.clone(),
                        content_summary: format!("Section with {}", keyword),
                        child_count: 0,
                    });

                    features.push(Feature {
                        name: format!("{:?}", section_type),
                        category: FeatureCategory::Content,
                        location: selector,
                        confidence: 0.7,
                        metadata: HashMap::new(),
                    });

                    break;
                }
            }
        }

        // Detect footer
        if html.contains("<footer") {
            content_sections.push(ContentSection {
                section_type: SectionType::Footer,
                heading: Some("Footer".to_string()),
                position: content_sections.len(),
                selector: "footer".to_string(),
                content_summary: "Page footer".to_string(),
                child_count: 0,
            });
        }
    }

    /// Detect specific features based on content patterns
    fn detect_specific_features(&self, html_lower: &str, features: &mut Vec<Feature>) {
        // Search functionality
        if html_lower.contains("search")
            && (html_lower.contains("input") || html_lower.contains("form"))
        {
            features.push(Feature {
                name: "Search Functionality".to_string(),
                category: FeatureCategory::Search,
                location: ".search".to_string(),
                confidence: 0.85,
                metadata: HashMap::new(),
            });
        }

        // Social sharing
        if html_lower.contains("share")
            || html_lower.contains("facebook")
            || html_lower.contains("twitter")
            || html_lower.contains("linkedin")
        {
            features.push(Feature {
                name: "Social Sharing".to_string(),
                category: FeatureCategory::Social,
                location: ".social-share".to_string(),
                confidence: 0.8,
                metadata: HashMap::new(),
            });
        }

        // E-commerce features
        if html_lower.contains("cart")
            || html_lower.contains("add to cart")
            || html_lower.contains("checkout")
        {
            features.push(Feature {
                name: "E-commerce".to_string(),
                category: FeatureCategory::Ecommerce,
                location: ".cart".to_string(),
                confidence: 0.9,
                metadata: HashMap::new(),
            });
        }

        // Authentication
        if html_lower.contains("sign in")
            || html_lower.contains("login")
            || html_lower.contains("sign up")
            || html_lower.contains("register")
        {
            features.push(Feature {
                name: "Authentication".to_string(),
                category: FeatureCategory::Authentication,
                location: ".auth".to_string(),
                confidence: 0.85,
                metadata: HashMap::new(),
            });
        }

        // Filtering
        if html_lower.contains("filter") || html_lower.contains("sort") {
            features.push(Feature {
                name: "Filtering/Sorting".to_string(),
                category: FeatureCategory::Filtering,
                location: ".filter".to_string(),
                confidence: 0.75,
                metadata: HashMap::new(),
            });
        }

        // Analytics
        if html_lower.contains("analytics")
            || html_lower.contains("google-analytics")
            || html_lower.contains("gtag")
        {
            features.push(Feature {
                name: "Analytics".to_string(),
                category: FeatureCategory::Analytics,
                location: "head".to_string(),
                confidence: 0.9,
                metadata: HashMap::new(),
            });
        }

        // Responsive design indicators
        if html_lower.contains("viewport") || html_lower.contains("media query") {
            features.push(Feature {
                name: "Responsive Design".to_string(),
                category: FeatureCategory::Responsive,
                location: "meta[name=viewport]".to_string(),
                confidence: 0.95,
                metadata: HashMap::new(),
            });
        }

        // Accessibility features
        if html_lower.contains("aria-") || html_lower.contains("accessibility") {
            features.push(Feature {
                name: "Accessibility".to_string(),
                category: FeatureCategory::Accessibility,
                location: "body".to_string(),
                confidence: 0.7,
                metadata: HashMap::new(),
            });
        }
    }
}

impl Default for FeatureRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recognizer_creation() {
        let recognizer = FeatureRecognizer::new();
        assert!(recognizer.config.min_confidence > 0.0);
    }

    #[test]
    fn test_form_detection() {
        let recognizer = FeatureRecognizer::new();
        let html = r#"
            <form>
                <input type="email" required>
                <input type="password" required>
                <button type="submit">Submit</button>
            </form>
        "#;
        let mut form_fields = Vec::new();
        let mut features = Vec::new();

        recognizer.analyze_forms(html, &mut form_fields, &mut features);

        assert!(!form_fields.is_empty());
        assert!(features.iter().any(|f| f.name == "Form"));
    }

    #[test]
    fn test_navigation_detection() {
        let recognizer = FeatureRecognizer::new();
        let html = r#"
            <nav class="navbar">
                <a href="/">Home</a>
                <a href="/about">About</a>
            </nav>
        "#;
        let mut navigation_items = Vec::new();
        let mut features = Vec::new();

        recognizer.analyze_navigation(html, &mut navigation_items, &mut features);

        assert!(features.iter().any(|f| f.name == "Navigation"));
    }

    #[test]
    fn test_interactive_detection() {
        let recognizer = FeatureRecognizer::new();
        let html = r#"
            <div class="modal">
                <button data-toggle="modal">Open</button>
            </div>
        "#;
        let mut interactive_elements = Vec::new();
        let mut features = Vec::new();

        recognizer.analyze_interactive_elements(html, &mut interactive_elements, &mut features);

        assert!(features.iter().any(|f| f.name == "Modal Dialog"));
    }

    #[test]
    fn test_specific_feature_detection() {
        let recognizer = FeatureRecognizer::new();
        let html_lower = "search functionality available with google analytics tracking";
        let mut features = Vec::new();

        recognizer.detect_specific_features(html_lower, &mut features);

        assert!(features.iter().any(|f| f.name == "Search Functionality"));
        assert!(features.iter().any(|f| f.name == "Analytics"));
    }

    #[tokio::test]
    async fn test_full_recognition() {
        let recognizer = FeatureRecognizer::new();

        let page = crate::data_models::PageContent {
            url: "https://example.com".to_string(),
            html: r#"
                <nav class="navbar">
                    <a href="/">Home</a>
                </nav>
                <form>
                    <input type="email">
                </form>
                <div class="modal"></div>
            "#
            .to_string(),
            ..Default::default()
        };

        let feature_map = recognizer.recognize(&page).await;

        assert!(!feature_map.features.is_empty());
    }
}
