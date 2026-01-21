//! Page Content Data Models
//!
//! Defines the core data structures for capturing and representing web page content.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Resource type enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceType {
    #[serde(rename = "css")]
    Css,
    #[serde(rename = "javascript")]
    JavaScript,
    #[serde(rename = "image")]
    Image,
    #[serde(rename = "font")]
    Font,
    #[serde(rename = "other")]
    Other,
}

/// External resource information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    /// Resource URL
    pub url: String,

    /// Resource type
    pub resource_type: ResourceType,

    /// Whether the resource is inline
    pub is_inline: bool,

    /// Original content (if inline resource)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// HTTP content type if known
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,

    /// File size in bytes (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
}

impl Resource {
    /// Create a new external resource
    pub fn new(url: String, resource_type: ResourceType) -> Self {
        Self {
            url,
            resource_type,
            is_inline: false,
            content: None,
            content_type: None,
            size: None,
        }
    }

    /// Create an inline resource
    pub fn inline(url: String, resource_type: ResourceType, content: String) -> Self {
        let size = content.len() as u64;
        Self {
            url,
            resource_type,
            is_inline: true,
            content: Some(content),
            content_type: None,
            size: Some(size),
        }
    }
}

/// Page metadata extracted from HTML head
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PageMetadata {
    /// Page title
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Meta description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Meta keywords
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,

    /// Content type from HTTP headers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,

    /// Server information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server: Option<String>,

    /// Character encoding
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding: Option<String>,

    /// Open Graph metadata
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub og_metadata: HashMap<String, String>,

    /// Viewport settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub viewport: Option<String>,

    /// Favicon URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub favicon: Option<String>,

    /// Canonical URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_url: Option<String>,
}

impl PageMetadata {
    /// Create empty metadata
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract metadata from title element
    pub fn with_title(mut self, title: Option<String>) -> Self {
        self.title = title;
        self
    }

    /// Extract metadata from meta tags
    pub fn with_description(mut self, description: Option<String>) -> Self {
        self.description = description;
        self
    }
}

/// Main page content structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageContent {
    /// Original URL
    pub url: String,

    /// Base URL for resolving relative links
    pub base_url: String,

    /// Original HTML string
    pub html: String,

    /// Parsed DOM tree (RcDom from html5ever)
    #[serde(skip)]
    pub dom: std::collections::HashMap<String, serde_json::Value>,

    /// Inline CSS blocks
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inline_css: Vec<InlineContent>,

    /// Inline JavaScript blocks
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inline_js: Vec<InlineContent>,

    /// External resources
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub external_resources: Vec<Resource>,

    /// Page metadata
    pub metadata: PageMetadata,

    /// HTTP status code
    pub status_code: u16,

    /// Final URL after redirects
    pub final_url: String,

    /// Fetch timestamp
    pub fetched_at: chrono::DateTime<chrono::Utc>,

    /// Response headers
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub response_headers: HashMap<String, String>,
}

impl PageContent {
    /// Create a new page content instance
    pub fn new(
        url: String,
        html: String,
        dom: std::collections::HashMap<String, serde_json::Value>,
    ) -> Self {
        let final_url = url.clone();
        let base_url = match url::Url::parse(&url) {
            Ok(parsed) => {
                let mut base = parsed;
                base.set_path("");
                base.set_query(None);
                base.to_string()
            }
            Err(_) => url.clone(),
        };

        Self {
            url: url.clone(),
            base_url,
            html,
            dom,
            inline_css: Vec::new(),
            inline_js: Vec::new(),
            external_resources: Vec::new(),
            metadata: PageMetadata::new(),
            status_code: 200,
            final_url,
            fetched_at: chrono::Utc::now(),
            response_headers: HashMap::new(),
        }
    }

    /// Add inline CSS
    pub fn add_inline_css(&mut self, content: String, source: String) {
        self.inline_css.push(InlineContent {
            content,
            source,
            is_obfuscated: false,
            obfuscation_techniques: Vec::new(),
        });
    }

    /// Add inline JavaScript
    pub fn add_inline_js(&mut self, content: String, source: String) {
        self.inline_js.push(InlineContent {
            content,
            source,
            is_obfuscated: false,
            obfuscation_techniques: Vec::new(),
        });
    }

    /// Add external resource
    pub fn add_resource(&mut self, resource: Resource) {
        self.external_resources.push(resource);
    }

    /// Get all CSS content
    pub fn all_css(&self) -> String {
        self.inline_css
            .iter()
            .map(|c| c.content.clone())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get all JavaScript content
    pub fn all_js(&self) -> String {
        self.inline_js
            .iter()
            .map(|c| c.content.clone())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Check if page appears to be a specific type
    pub fn appears_to_be(&self, page_type: &str) -> bool {
        let text = self.html.to_lowercase();
        let meta_desc = self
            .metadata
            .description
            .as_ref()
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        match page_type {
            "ecommerce" => {
                text.contains("shop")
                    || text.contains("cart")
                    || text.contains("product")
                    || text.contains("price")
                    || text.contains("checkout")
            }
            "blog" => {
                text.contains("article")
                    || text.contains("post")
                    || text.contains("blog")
                    || meta_desc.contains("blog")
            }
            "documentation" => {
                text.contains("documentation")
                    || text.contains("docs")
                    || text.contains("guide")
                    || text.contains("tutorial")
            }
            "social" => {
                text.contains("profile")
                    || text.contains("friend")
                    || text.contains("follow")
                    || text.contains("post")
            }
            _ => false,
        }
    }

    /// Extract text content from the page
    pub fn extract_text(&self) -> String {
        // Simple text extraction - in real implementation would traverse DOM
        let text_elements = ["h1", "h2", "h3", "p", "li", "a", "span", "div"];
        let mut text = String::new();

        for element in text_elements {
            let matches: Vec<_> = self.html.matches(element).collect();
            if !matches.is_empty() {
                // Simplified - actual implementation would parse HTML
                text.push_str(&format!("Found {} {} elements\n", matches.len(), element));
            }
        }

        text
    }
}

/// Inline content block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineContent {
    /// Content string
    pub content: String,

    /// Source location (e.g., "line 10", "inline script")
    pub source: String,

    /// Whether the content appears obfuscated
    #[serde(default)]
    pub is_obfuscated: bool,

    /// Obfuscation techniques detected (if any)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub obfuscation_techniques: Vec<String>,
}

impl InlineContent {
    /// Create new inline content
    pub fn new(content: String, source: String) -> Self {
        Self {
            content,
            source,
            is_obfuscated: false,
            obfuscation_techniques: Vec::new(),
        }
    }
}

/// Simplified DOM representation for serialization
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimplifiedDom {
    /// Root element
    pub root: SimplifiedNode,

    /// All elements by tag name
    #[serde(default)]
    pub elements_by_tag: HashMap<String, Vec<String>>,

    /// Elements with IDs
    #[serde(default)]
    pub elements_by_id: HashMap<String, String>,

    /// Form elements
    #[serde(default)]
    pub forms: Vec<FormInfo>,

    /// Navigation elements
    #[serde(default)]
    pub navigation: Vec<NavInfo>,
}

/// Simplified DOM node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedNode {
    /// Node type
    pub node_type: String,

    /// Tag name (for elements)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,

    /// Text content (for text nodes)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// Attributes (for elements)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub attributes: HashMap<String, String>,

    /// Child nodes
    #[serde(default)]
    pub children: Vec<SimplifiedNode>,

    /// Element ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Element classes
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub classes: Vec<String>,
}

impl Default for SimplifiedNode {
    fn default() -> Self {
        SimplifiedNode {
            node_type: "element".to_string(),
            tag: None,
            text: None,
            attributes: HashMap::new(),
            children: Vec::new(),
            id: None,
            classes: Vec::new(),
        }
    }
}

/// Form information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormInfo {
    /// Form element ID or selector
    pub selector: String,

    /// Form action URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,

    /// Form method
    #[serde(default)]
    pub method: String,

    /// Input fields
    #[serde(default)]
    pub inputs: Vec<InputInfo>,

    /// Form type inference
    #[serde(default)]
    pub form_type: String,
}

/// Input field information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputInfo {
    /// Input name
    pub name: String,

    /// Input type
    pub input_type: String,

    /// Is required
    #[serde(default)]
    pub required: bool,

    /// Placeholder text
    #[serde(skip_serializing_if = "Option::is_none")]
    pub placeholder: Option<String>,
}

/// Navigation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavInfo {
    /// Navigation selector
    pub selector: String,

    /// Navigation items
    #[serde(default)]
    pub items: Vec<NavItem>,

    /// Is dropdown menu
    #[serde(default)]
    pub has_dropdowns: bool,
}

/// Navigation item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavItem {
    /// Item text
    pub text: String,

    /// Item URL
    pub url: String,

    /// Child items (for dropdowns)
    #[serde(default)]
    pub children: Vec<NavItem>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_content_creation() {
        let page = PageContent::new(
            "https://example.com".to_string(),
            "<html><body>Hello</body></html>".to_string(),
            HashMap::new(),
        );

        assert_eq!(page.url, "https://example.com");
        assert!(page.inline_css.is_empty());
        assert!(page.inline_js.is_empty());
    }

    #[test]
    fn test_resource_creation() {
        let resource = Resource::new(
            "https://example.com/style.css".to_string(),
            ResourceType::Css,
        );

        assert_eq!(resource.url, "https://example.com/style.css");
        assert_eq!(resource.resource_type, ResourceType::Css);
        assert!(!resource.is_inline);
    }

    #[test]
    fn test_inline_content() {
        let content = InlineContent::new(
            ".class { color: red; }".to_string(),
            "inline style".to_string(),
        );

        assert!(!content.is_obfuscated);
        assert_eq!(content.source, "inline style");
    }

    #[test]
    fn test_page_type_detection() {
        let page = PageContent::new(
            "https://shop.example.com".to_string(),
            "<html><body>Shop for products and add to cart</body></html>".to_string(),
            HashMap::new(),
        );

        assert!(page.appears_to_be("ecommerce"));
        assert!(!page.appears_to_be("blog"));
    }
}
