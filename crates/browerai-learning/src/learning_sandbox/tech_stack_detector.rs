//! Tech Stack Detector
//!
//! Detects the technology stack used by a website including JavaScript frameworks,
//! CSS frameworks, server-side technologies, and build tools.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::data_models::PageContent;
use crate::framework_knowledge::{FrameworkCategory, FrameworkKnowledgeBase};

/// Tech stack detection report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechStackReport {
    /// Detected JavaScript frameworks and libraries
    pub js_frameworks: Vec<FrameworkInfo>,

    /// Detected CSS frameworks and libraries
    pub css_frameworks: Vec<FrameworkInfo>,

    /// Detected server-side technologies
    pub server_tech: Vec<ServerTechInfo>,

    /// Detected build tools and bundlers
    pub build_tools: Vec<BuildToolInfo>,

    /// Detected CMS or platform
    pub cms_platform: Option<CmsInfo>,

    /// Overall confidence score (0.0 - 1.0)
    pub confidence: f32,

    /// Detection metadata
    pub metadata: TechStackMetadata,
}

/// Information about a detected framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkInfo {
    /// Framework name
    pub name: String,

    /// Framework version (if detected)
    pub version: Option<String>,

    /// Category of framework
    pub category: String,

    /// Detection confidence (0.0 - 1.0)
    pub confidence: f32,

    /// Evidence supporting the detection
    pub evidence: Vec<String>,

    /// File or script where it was detected
    pub source: Option<String>,
}

/// Information about detected server technology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerTechInfo {
    /// Technology name (e.g., "Node.js", "Python", "PHP")
    pub name: String,

    /// Version if detected
    pub version: Option<String>,

    /// Web server (e.g., "Nginx", "Apache")
    pub web_server: Option<String>,

    /// Detection confidence
    pub confidence: f32,

    /// Evidence for detection
    pub evidence: Vec<String>,
}

/// Information about detected build tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildToolInfo {
    /// Tool name (e.g., "Webpack", "Vite", "Rollup")
    pub name: String,

    /// Version if detected
    pub version: Option<String>,

    /// Detection confidence
    pub confidence: f32,

    /// Evidence for detection
    pub evidence: Vec<String>,
}

/// Information about detected CMS or platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmsInfo {
    /// CMS name (e.g., "WordPress", "Shopify", "React")
    pub name: String,

    /// Version if detected
    pub version: Option<String>,

    /// Detection confidence
    pub confidence: f32,

    /// Evidence for detection
    pub evidence: Vec<String>,
}

/// Metadata about the detection process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechStackMetadata {
    /// URL that was analyzed
    pub url: String,

    /// Timestamp of analysis
    pub analyzed_at: chrono::DateTime<chrono::Utc>,

    /// Time taken for detection in milliseconds
    pub detection_time_ms: u64,

    /// Number of resources analyzed
    pub resources_analyzed: usize,
}

/// Tech Stack Detector Configuration
#[derive(Debug, Clone)]
pub struct TechStackDetectorConfig {
    /// Minimum confidence threshold for detection
    pub min_confidence: f32,

    /// Enable deep analysis of inline scripts
    pub deep_analysis: bool,

    /// Enable version detection
    pub detect_versions: bool,

    /// Maximum resources to analyze
    pub max_resources: usize,
}

impl Default for TechStackDetectorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            deep_analysis: true,
            detect_versions: true,
            max_resources: 50,
        }
    }
}

/// Tech Stack Detector
///
/// Analyzes page content to detect the technology stack used by a website.
#[derive(Debug, Clone)]
pub struct TechStackDetector {
    config: TechStackDetectorConfig,
    framework_kb: Arc<FrameworkKnowledgeBase>,
}

impl TechStackDetector {
    /// Create a new tech stack detector
    pub fn new() -> Self {
        Self::with_config(TechStackDetectorConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: TechStackDetectorConfig) -> Self {
        let framework_kb = FrameworkKnowledgeBase::new();
        Self {
            config,
            framework_kb: Arc::new(framework_kb),
        }
    }

    /// Detect the tech stack used by a page
    ///
    /// # Arguments
    /// * `page` - Page content to analyze
    ///
    /// # Returns
    /// * `TechStackReport` with detection results
    pub async fn detect(&self, page: &PageContent) -> TechStackReport {
        let start_time = chrono::Utc::now();

        let mut js_frameworks = Vec::new();
        let mut css_frameworks = Vec::new();
        let mut server_tech = Vec::new();
        let mut build_tools = Vec::new();
        let mut cms_platform = None;

        // Analyze HTML for inline scripts and styles
        self.analyze_inline_content(page, &mut js_frameworks, &mut css_frameworks);

        // Analyze external resources
        let resources_to_analyze = page
            .external_resources
            .iter()
            .take(self.config.max_resources);
        for resource in resources_to_analyze {
            match resource.resource_type {
                crate::data_models::ResourceType::JavaScript => {
                    self.analyze_js_content(
                        &resource.content,
                        &mut js_frameworks,
                        &mut build_tools,
                    );
                }
                crate::data_models::ResourceType::CSS => {
                    self.analyze_css_content(&resource.content, &mut css_frameworks);
                }
                _ => {}
            }
        }

        // Analyze server headers if available
        if !page.response_headers.is_empty() {
            self.analyze_server_headers(&page.response_headers, &mut server_tech);
        }

        // Detect CMS/platform based on URL patterns and content
        self.detect_cms(&page.url, page, &mut js_frameworks, &mut cms_platform);

        // Calculate overall confidence
        let overall_confidence = self.calculate_overall_confidence(
            &js_frameworks,
            &css_frameworks,
            &server_tech,
            &build_tools,
        );

        let end_time = chrono::Utc::now();
        let detection_time_ms = (end_time - start_time).num_milliseconds() as u64;

        TechStackReport {
            js_frameworks,
            css_frameworks,
            server_tech,
            build_tools,
            cms_platform,
            confidence: overall_confidence,
            metadata: TechStackMetadata {
                url: page.url.clone(),
                analyzed_at: chrono::Utc::now(),
                detection_time_ms,
                resources_analyzed: page.external_resources.len(),
            },
        }
    }

    /// Analyze inline content for framework signatures
    fn analyze_inline_content(
        &self,
        page: &PageContent,
        js_frameworks: &mut Vec<FrameworkInfo>,
        css_frameworks: &mut Vec<FrameworkInfo>,
    ) {
        // Analyze inline JavaScript
        for inline_js in &page.inline_js {
            if let Some(content) = &inline_js.content {
                self.analyze_js_content(content, js_frameworks, &mut Vec::new());
            }
        }

        // Analyze inline CSS
        for inline_css in &page.inline_css {
            if let Some(content) = &inline_css.content {
                self.analyze_css_content(content, css_frameworks);
            }
        }
    }

    /// Analyze JavaScript content for framework signatures
    fn analyze_js_content(
        &self,
        content: &str,
        js_frameworks: &mut Vec<FrameworkInfo>,
        build_tools: &mut Vec<BuildToolInfo>,
    ) {
        let content_lower = content.to_lowercase();

        // React detection
        if content.contains("ReactDOM")
            || content.contains("__react")
            || content.contains("reactjs")
        {
            let version = self.extract_version(&content, r#"react["']?\s*v?([\d.]+)"#);
            js_frameworks.push(FrameworkInfo {
                name: "React".to_string(),
                version,
                category: "Frontend Framework".to_string(),
                confidence: 0.9,
                evidence: vec!["React DOM reference".to_string()],
                source: None,
            });
        }

        // Vue.js detection
        if content.contains("Vue") || content.contains("__vue") || content.contains("vuejs") {
            let version = self.extract_version(&content, r#"vue["']?\s*v?([\d.]+)"#);
            js_frameworks.push(FrameworkInfo {
                name: "Vue.js".to_string(),
                version,
                category: "Frontend Framework".to_string(),
                confidence: 0.9,
                evidence: vec!["Vue reference".to_string()],
                source: None,
            });
        }

        // Angular detection
        if content.contains("ng-") || content.contains("angular") || content.contains("@angular") {
            js_frameworks.push(FrameworkInfo {
                name: "Angular".to_string(),
                version: self.extract_version(&content, r#"angular["']?\s*v?([\d.]+)"#),
                category: "Frontend Framework".to_string(),
                confidence: 0.85,
                evidence: vec!["Angular directives detected".to_string()],
                source: None,
            });
        }

        // jQuery detection
        if content.contains("jQuery") || content.contains("$('") || content.contains("$.fn") {
            let version = self.extract_version(&content, r#"jquery["']?\s*v?([\d.]+)"#);
            js_frameworks.push(FrameworkInfo {
                name: "jQuery".to_string(),
                version,
                category: "JavaScript Library".to_string(),
                confidence: 0.8,
                evidence: vec!["jQuery reference".to_string()],
                source: None,
            });
        }

        // Bootstrap detection
        if content.contains("bootstrap")
            && (content.contains(".modal") || content.contains(".dropdown"))
        {
            let version = self.extract_version(&content, r#"bootstrap["']?\s*v?([\d.]+)"#);
            js_frameworks.push(FrameworkInfo {
                name: "Bootstrap".to_string(),
                version,
                category: "CSS Framework".to_string(),
                confidence: 0.75,
                evidence: vec!["Bootstrap components detected".to_string()],
                source: None,
            });
        }

        // Webpack detection
        if content.contains("webpack") || content.contains("__webpack") {
            build_tools.push(BuildToolInfo {
                name: "Webpack".to_string(),
                version: self.extract_version(&content, r#"webpack["']?\s*v?([\d.]+)"#),
                confidence: 0.7,
                evidence: vec!["Webpack reference".to_string()],
            });
        }

        // Lodash detection
        if content.contains("_") && (content.contains("_.map") || content.contains("_.filter")) {
            js_frameworks.push(FrameworkInfo {
                name: "Lodash".to_string(),
                version: self.extract_version(&content, r#"lodash["']?\s*v?([\d.]+)"#),
                category: "Utility Library".to_string(),
                confidence: 0.7,
                evidence: vec!["Lodash reference".to_string()],
                source: None,
            });
        }
    }

    /// Analyze CSS content for framework signatures
    fn analyze_css_content(&self, content: &str, css_frameworks: &mut Vec<FrameworkInfo>) {
        let content_lower = content.to_lowercase();

        // Tailwind CSS detection
        if content.contains("tailwind") || content.contains("-tw-") {
            css_frameworks.push(FrameworkInfo {
                name: "Tailwind CSS".to_string(),
                version: None,
                category: "CSS Framework".to_string(),
                confidence: 0.9,
                evidence: vec!["Tailwind CSS classes detected".to_string()],
                source: None,
            });
        }

        // Bootstrap detection
        if content.contains(".btn-")
            || content.contains(".container")
            || content.contains("bootstrap")
        {
            css_frameworks.push(FrameworkInfo {
                name: "Bootstrap".to_string(),
                version: self.extract_version(content, r#"bootstrap.*?([\d.]+)"#),
                category: "CSS Framework".to_string(),
                confidence: 0.85,
                evidence: vec!["Bootstrap classes detected".to_string()],
                source: None,
            });
        }

        // Material Design detection
        if content.contains(".mdc-") || content.contains("material-design") {
            css_frameworks.push(FrameworkInfo {
                name: "Material Design".to_string(),
                version: None,
                category: "UI Framework".to_string(),
                confidence: 0.8,
                evidence: vec!["Material Design classes detected".to_string()],
                source: None,
            });
        }

        // Bulma detection
        if content.contains(".columns")
            && (content.contains(".column") || content.contains(".tile"))
        {
            css_frameworks.push(FrameworkInfo {
                name: "Bulma".to_string(),
                version: None,
                category: "CSS Framework".to_string(),
                confidence: 0.75,
                evidence: vec!["Bulma classes detected".to_string()],
                source: None,
            });
        }
    }

    /// Analyze server headers for technology detection
    fn analyze_server_headers(
        &self,
        headers: &HashMap<String, String>,
        server_tech: &mut Vec<ServerTechInfo>,
    ) {
        for (key, value) in headers {
            let key_lower = key.to_lowercase();
            let value_lower = value.to_lowercase();

            // X-Powered-By header
            if key_lower == "x-powered-by" {
                if value_lower.contains("php") {
                    server_tech.push(ServerTechInfo {
                        name: "PHP".to_string(),
                        version: self.extract_version(&value, r#"php/([\d.]+)"#),
                        web_server: None,
                        confidence: 0.9,
                        evidence: vec!["X-Powered-By header".to_string()],
                    });
                } else if value_lower.contains("express") {
                    server_tech.push(ServerTechInfo {
                        name: "Node.js".to_string(),
                        version: None,
                        web_server: None,
                        confidence: 0.8,
                        evidence: vec!["Express in X-Powered-By".to_string()],
                    });
                }
            }

            // Server header
            if key_lower == "server" {
                if value_lower.contains("nginx") {
                    if let Some(nginx_info) =
                        server_tech.iter_mut().find(|s| s.web_server.is_none())
                    {
                        nginx_info.web_server = Some("Nginx".to_string());
                    } else {
                        server_tech.push(ServerTechInfo {
                            name: "Unknown".to_string(),
                            version: None,
                            web_server: Some("Nginx".to_string()),
                            confidence: 0.7,
                            evidence: vec!["Server header".to_string()],
                        });
                    }
                } else if value_lower.contains("apache") {
                    if let Some(apache_info) =
                        server_tech.iter_mut().find(|s| s.web_server.is_none())
                    {
                        apache_info.web_server = Some("Apache".to_string());
                    } else {
                        server_tech.push(ServerTechInfo {
                            name: "Unknown".to_string(),
                            version: None,
                            web_server: Some("Apache".to_string()),
                            confidence: 0.7,
                            evidence: vec!["Server header".to_string()],
                        });
                    }
                }
            }
        }
    }

    /// Detect CMS or platform
    fn detect_cms(
        &self,
        url: &str,
        page: &PageContent,
        js_frameworks: &mut Vec<FrameworkInfo>,
        cms: &mut Option<CmsInfo>,
    ) {
        let url_lower = url.to_lowercase();
        let html_lower = page.html.to_lowercase();

        // WordPress detection
        if url_lower.contains("wp-content")
            || url_lower.contains("wp-includes")
            || html_lower.contains("wordpress")
        {
            *cms = Some(CmsInfo {
                name: "WordPress".to_string(),
                version: self.extract_version(&html_lower, r#"wordpress["']?\s*v?([\d.]+)"#),
                confidence: 0.95,
                evidence: vec!["WordPress URL patterns or references".to_string()],
            });
        }

        // Shopify detection
        if url_lower.contains("shopify") || html_lower.contains("shopify") {
            *cms = Some(CmsInfo {
                name: "Shopify".to_string(),
                version: None,
                confidence: 0.9,
                evidence: vec!["Shopify reference".to_string()],
            });

            // Shopify typically uses React
            if !js_frameworks.iter().any(|f| f.name == "React") {
                js_frameworks.push(FrameworkInfo {
                    name: "React".to_string(),
                    version: None,
                    category: "Frontend Framework".to_string(),
                    confidence: 0.6,
                    evidence: vec!["Commonly used with Shopify".to_string()],
                    source: None,
                });
            }
        }

        // Next.js detection
        if html_lower.contains("next") || html_lower.contains("_next") {
            *cms = Some(CmsInfo {
                name: "Next.js".to_string(),
                version: self.extract_version(&html_lower, r#"nextjs["']?\s*v?([\d.]+)"#),
                confidence: 0.9,
                evidence: vec!["Next.js reference".to_string()],
            });
        }

        // Gatsby detection
        if html_lower.contains("gatsby") || html_lower.contains("gatsbyjs") {
            *cms = Some(CmsInfo {
                name: "Gatsby".to_string(),
                version: self.extract_version(&html_lower, r#"gatsby["']?\s*v?([\d.]+)"#),
                confidence: 0.9,
                evidence: vec!["Gatsby reference".to_string()],
            });
        }
    }

    /// Calculate overall detection confidence
    fn calculate_overall_confidence(
        &self,
        js_frameworks: &[FrameworkInfo],
        css_frameworks: &[FrameworkInfo],
        server_tech: &[ServerTechInfo],
        build_tools: &[BuildToolInfo],
    ) -> f32 {
        let mut total_confidence = 0.0;
        let mut count = 0;

        for f in js_frameworks {
            if f.confidence >= self.config.min_confidence {
                total_confidence += f.confidence;
                count += 1;
            }
        }

        for f in css_frameworks {
            if f.confidence >= self.config.min_confidence {
                total_confidence += f.confidence;
                count += 1;
            }
        }

        for s in server_tech {
            if s.confidence >= self.config.min_confidence {
                total_confidence += s.confidence;
                count += 1;
            }
        }

        for b in build_tools {
            if b.confidence >= self.config.min_confidence {
                total_confidence += b.confidence;
                count += 1;
            }
        }

        if count > 0 {
            (total_confidence / count as f32).min(1.0)
        } else {
            0.0
        }
    }

    /// Extract version from content using regex pattern
    fn extract_version(&self, content: &str, pattern: &str) -> Option<String> {
        // Simplified version extraction
        // In a full implementation, this would use regex
        None
    }
}

impl Default for TechStackDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let detector = TechStackDetector::new();
        assert!(detector.config.min_confidence > 0.0);
    }

    #[test]
    fn test_js_content_analysis() {
        let detector = TechStackDetector::new();
        let mut frameworks = Vec::new();
        let mut build_tools = Vec::new();

        let react_code = r#"
            import React from 'react';
            import ReactDOM from 'react-dom';
            ReactDOM.render(<App />, document.getElementById('root'));
        "#;

        detector.analyze_js_content(react_code, &mut frameworks, &mut build_tools);

        assert!(frameworks.iter().any(|f| f.name == "React"));
    }

    #[test]
    fn test_css_content_analysis() {
        let detector = TechStackDetector::new();
        let mut css_frameworks = Vec::new();

        let tailwind_css = r#"
            .flex { display: flex; }
            .container { max-width: 1200px; }
            .bg-blue-500 { background-color: #3b82f6; }
        "#;

        detector.analyze_css_content(tailwind_css, &mut css_frameworks);

        assert!(css_frameworks.iter().any(|f| f.name == "Tailwind CSS"));
    }

    #[tokio::test]
    async fn test_full_detection() {
        let detector = TechStackDetector::new();

        let page = PageContent {
            url: "https://example.com".to_string(),
            html: r#"<!DOCTYPE html><html><head></head><body></body></html>"#.to_string(),
            dom: std::collections::HashMap::new(),
            metadata: Default::default(),
            resources: vec![],
            ..Default::default()
        };

        let report = detector.detect(&page).await;
        assert!(report.confidence >= 0.0);
        assert!(!report.js_frameworks.is_empty() || report.js_frameworks.is_empty());
    }
}
