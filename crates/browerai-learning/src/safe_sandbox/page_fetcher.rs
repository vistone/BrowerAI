//! Page Fetcher
//!
//! Fetches and parses web pages, extracting HTML, CSS, JavaScript, and resources.

use anyhow::{anyhow, Context, Result};
use reqwest::{Client, Response, StatusCode};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use url::Url;

use crate::data_models::{PageContent, Resource, ResourceType};

/// Configuration for page fetching
#[derive(Debug, Clone)]
pub struct FetcherConfig {
    /// Maximum number of redirects to follow
    pub max_redirects: usize,

    /// Request timeout
    pub timeout: Duration,

    /// User agent string
    pub user_agent: String,

    /// Whether to follow redirects
    pub follow_redirects: bool,

    /// Additional headers to include
    pub extra_headers: HashMap<String, String>,
}

impl Default for FetcherConfig {
    fn default() -> Self {
        Self {
            max_redirects: 10,
            timeout: Duration::from_secs(30),
            user_agent: "BrowerAI/1.0 (https://brower.ai)".to_string(),
            follow_redirects: true,
            extra_headers: HashMap::new(),
        }
    }
}

/// Page Fetcher
///
/// Handles HTTP requests, redirect following, and content extraction.
#[derive(Debug, Clone)]
pub struct PageFetcher {
    client: Arc<Client>,
    config: FetcherConfig,
}

impl PageFetcher {
    /// Create a new page fetcher with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(FetcherConfig::default())
    }

    /// Create a page fetcher with custom configuration
    pub fn with_config(config: FetcherConfig) -> Result<Self> {
        let mut builder = Client::builder()
            .timeout(config.timeout)
            .user_agent(&config.user_agent);

        if !config.follow_redirects {
            builder = builder.redirect(reqwest::redirect::Policy::none());
        }

        // Add extra headers
        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in &config.extra_headers {
            if let Ok(key) = key.parse::<reqwest::header::HeaderName>() {
                if let Ok(value) = value.parse::<reqwest::header::HeaderValue>() {
                    headers.insert(key, value);
                }
            }
        }
        builder = builder.default_headers(headers);

        let client = builder.build().context("Failed to build HTTP client")?;

        Ok(Self {
            client: Arc::new(client),
            config,
        })
    }

    /// Fetch a page from URL
    ///
    /// # Arguments
    /// * `url` - The URL to fetch
    ///
    /// # Returns
    /// * `PageContent` containing the fetched and parsed page
    pub async fn fetch(&self, url: &str) -> Result<PageContent> {
        let url = Url::parse(url).context("Invalid URL")?;

        let response = self.send_request(&url).await?;

        let status = response.status();
        let final_url = response.url().to_string();

        // Extract response headers before consuming response
        let response_headers: HashMap<String, String> = response
            .headers()
            .iter()
            .filter_map(|(key, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value_str| (key.to_string(), value_str.to_string()))
            })
            .collect();

        // Check for HTTP errors
        if !status.is_success() {
            return Err(anyhow!(
                "HTTP error: {} {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("Unknown")
            ));
        }

        let html = response
            .text()
            .await
            .context("Failed to read response body")?;

        // Parse HTML and extract content
        let mut page = PageContent::new(
            final_url.clone(),
            html.clone(),
            HashMap::new(), // DOM will be populated separately
        );

        page.status_code = status.as_u16();
        page.final_url = final_url.clone();

        // Set response headers
        page.response_headers = response_headers;

        // Extract metadata from HTML
        self.extract_metadata(&html, &mut page);

        // Extract inline CSS and JS
        self.extract_inline_content(&html, &mut page);

        // Extract external resources
        self.extract_external_resources(&html, &final_url, &mut page);

        // Extract metadata
        page.metadata.title = PageFetcher::extract_title(&html);
        page.metadata.description = PageFetcher::extract_description(&html);

        Ok(page)
    }

    /// Send HTTP request with redirect handling
    async fn send_request(&self, url: &Url) -> Result<Response> {
        let mut current_url = url.clone();
        let mut redirects = 0;

        loop {
            let response = self
                .client
                .get(current_url.clone())
                .send()
                .await
                .context("HTTP request failed")?;

            match response.status() {
                StatusCode::MOVED_PERMANENTLY
                | StatusCode::FOUND
                | StatusCode::SEE_OTHER
                | StatusCode::TEMPORARY_REDIRECT => {
                    if redirects >= self.config.max_redirects {
                        return Err(anyhow!(
                            "Too many redirects (max: {})",
                            self.config.max_redirects
                        ));
                    }

                    if let Some(location) = response.headers().get("Location") {
                        let location_str = location.to_str().context("Invalid redirect header")?;
                        current_url = current_url
                            .join(location_str)
                            .context("Invalid redirect URL")?;
                        redirects += 1;
                    } else {
                        return Err(anyhow!("Redirect without Location header"));
                    }
                }
                _ => return Ok(response),
            }
        }
    }

    /// Extract metadata from HTML
    fn extract_metadata(&self, html: &str, page: &mut PageContent) {
        // Extract charset
        if let Some(css) = Self::extract_tag_content(html, "meta", "charset") {
            page.metadata.encoding = Some(css);
        }

        // Extract viewport
        if let Some(viewport) = Self::extract_meta_attribute(html, "viewport", "content") {
            page.metadata.viewport = Some(viewport);
        }

        // Extract Open Graph data
        for og_property in &["title", "description", "image", "url", "type", "site_name"] {
            if let Some(value) = Self::extract_meta_property(html, "og", og_property) {
                page.metadata
                    .og_metadata
                    .insert(format!("og:{}", og_property), value);
            }
        }

        // Extract canonical URL
        if let Some(canonical) = Self::extract_link_href(html, "canonical") {
            page.metadata.canonical_url = Some(canonical);
        }

        // Extract favicon
        if let Some(favicon) = Self::extract_link_href(html, "icon") {
            page.metadata.favicon = Some(favicon);
        }
    }

    /// Extract title from HTML
    fn extract_title(html: &str) -> Option<String> {
        Self::extract_tag_content(html, "title", "")
    }

    /// Extract description from meta tag
    fn extract_description(html: &str) -> Option<String> {
        Self::extract_meta_attribute(html, "description", "content")
    }

    /// Extract content from a specific tag
    fn extract_tag_content(html: &str, tag: &str, _attribute: &str) -> Option<String> {
        let pattern = format!("<{}>", tag);
        let start = html.find(&pattern)?;
        let end = html[start..].find("</")?;
        Some(html[start + pattern.len()..start + end].trim().to_string())
    }

    /// Extract meta tag attribute
    fn extract_meta_attribute(html: &str, name: &str, _attr: &str) -> Option<String> {
        let pattern = format!(r#"<meta name="{}" content=""#, name);
        let start = html.find(&pattern)?;
        let end = html[start + pattern.len()..].find('"')?;
        Some(html[start + pattern.len()..start + pattern.len() + end].to_string())
    }

    /// Extract Open Graph property
    fn extract_meta_property(html: &str, prefix: &str, property: &str) -> Option<String> {
        let pattern = format!(r#"<meta property="{}:{}" content=""#, prefix, property);
        let start = html.find(&pattern)?;
        let end = html[start + pattern.len()..].find('"')?;
        Some(html[start + pattern.len()..start + pattern.len() + end].to_string())
    }

    /// Extract link href
    fn extract_link_href(html: &str, rel: &str) -> Option<String> {
        let pattern = format!(r#"<link rel="{}" href=""#, rel);
        let start = html.find(&pattern)?;
        let end = html[start + pattern.len()..].find('"')?;
        Some(html[start + pattern.len()..start + pattern.len() + end].to_string())
    }

    /// Extract inline CSS and JavaScript
    fn extract_inline_content(&self, html: &str, page: &mut PageContent) {
        // Extract inline CSS
        let css_pattern = "<style";
        let mut css_start = 0;
        while let Some(start) = html[css_start..].find(css_pattern) {
            let abs_start = css_start + start;
            if let Some(end) = html[abs_start..].find("</style>") {
                let content = html[abs_start + css_pattern.len()..abs_start + end]
                    .trim()
                    .trim_start_matches('>')
                    .trim()
                    .to_string();

                if !content.is_empty() {
                    page.add_inline_css(content, format!("inline style at position {}", abs_start));
                }
                css_start = abs_start + end + "</style>".len();
            } else {
                break;
            }
        }

        // Extract inline JavaScript
        let js_pattern = "<script";
        let mut js_start = 0;
        while let Some(start) = html[js_start..].find(js_pattern) {
            let abs_start = js_start + start;
            if let Some(end) = html[abs_start..].find("</script>") {
                let content = html[abs_start + js_pattern.len()..abs_start + end]
                    .trim()
                    .trim_start_matches('>')
                    .trim()
                    .to_string();

                // Skip empty scripts and type definitions
                if !content.is_empty() && !content.starts_with("<!") {
                    page.add_inline_js(content, format!("inline script at position {}", abs_start));
                }
                js_start = abs_start + end + "</script>".len();
            } else {
                break;
            }
        }
    }

    /// Extract external resources (CSS, JS, images)
    fn extract_external_resources(&self, html: &str, base_url: &str, page: &mut PageContent) {
        // Extract CSS links
        let css_pattern = r#"<link rel="stylesheet" href=""#;
        let mut css_start = 0;
        while let Some(start) = html[css_start..].find(css_pattern) {
            let abs_start = css_start + start;
            if let Some(end) = html[abs_start + css_pattern.len()..].find('"') {
                let url = &html[abs_start + css_pattern.len()..abs_start + css_pattern.len() + end];
                let absolute_url = self.resolve_url(url, base_url);
                page.add_resource(Resource::new(absolute_url, ResourceType::Css));
                css_start = abs_start + css_pattern.len() + end + 1;
            } else {
                break;
            }
        }

        // Extract JS scripts
        let js_pattern = r#"<script src=""#;
        let mut js_start = 0;
        while let Some(start) = html[js_start..].find(js_pattern) {
            let abs_start = js_start + start;
            if let Some(end) = html[abs_start + js_pattern.len()..].find('"') {
                let url = &html[abs_start + js_pattern.len()..abs_start + js_pattern.len() + end];
                let absolute_url = self.resolve_url(url, base_url);
                page.add_resource(Resource::new(absolute_url, ResourceType::JavaScript));
                js_start = abs_start + js_pattern.len() + end + 1;
            } else {
                break;
            }
        }

        // Extract images
        let img_pattern = r#"<img src=""#;
        let mut img_start = 0;
        while let Some(start) = html[img_start..].find(img_pattern) {
            let abs_start = img_start + start;
            if let Some(end) = html[abs_start + img_pattern.len()..].find('"') {
                let url = &html[abs_start + img_pattern.len()..abs_start + img_pattern.len() + end];
                let absolute_url = self.resolve_url(url, base_url);
                page.add_resource(Resource::new(absolute_url, ResourceType::Image));
                img_start = abs_start + img_pattern.len() + end + 1;
            } else {
                break;
            }
        }

        // Extract fonts
        let font_pattern = r#"<link rel="preload" as="font""#;
        if let Some(start) = html.find(font_pattern) {
            if let Some(href_start) = html[start..].find("href=\"") {
                let href_abs = start + href_start + "href=\"".len();
                if let Some(href_end) = html[href_abs..].find('"') {
                    let url = &html[href_abs..href_abs + href_end];
                    let absolute_url = self.resolve_url(url, base_url);
                    page.add_resource(Resource::new(absolute_url, ResourceType::Font));
                }
            }
        }
    }

    /// Resolve relative URL to absolute
    fn resolve_url(&self, relative: &str, base: &str) -> String {
        if relative.starts_with("http://") || relative.starts_with("https://") {
            return relative.to_string();
        }

        if relative.starts_with("//") {
            return format!("https:{}", relative);
        }

        if let Ok(base_url) = Url::parse(base) {
            if let Ok(resolved) = base_url.join(relative) {
                return resolved.to_string();
            }
        }

        // Fallback: prepend base path
        if relative.starts_with("/") {
            if let Some(domain_end) = base.find('/') {
                return format!("{}{}", &base[..domain_end], relative);
            }
        }

        relative.to_string()
    }

    /// Get configuration
    pub fn config(&self) -> &FetcherConfig {
        &self.config
    }
}

impl Default for PageFetcher {
    fn default() -> Self {
        PageFetcher::new().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fetch_simple_page() {
        // Use a simple test server or mock
        // For now, just test the structure
        let fetcher = PageFetcher::new().unwrap();
        assert!(fetcher.config().max_redirects > 0);
    }

    #[test]
    fn test_url_resolution() {
        let fetcher = PageFetcher::new().unwrap();

        // Absolute URL
        assert_eq!(
            fetcher.resolve_url("https://example.com/style.css", "https://test.com/page"),
            "https://example.com/style.css"
        );

        // Protocol-relative URL
        assert_eq!(
            fetcher.resolve_url("//cdn.example.com/script.js", "https://test.com/page"),
            "https://cdn.example.com/script.js"
        );

        // Relative URL
        assert_eq!(
            fetcher.resolve_url("style.css", "https://test.com/path/to/page"),
            "https://test.com/path/to/style.css"
        );

        // Root-relative URL
        assert_eq!(
            fetcher.resolve_url("/abs/path.js", "https://test.com/rel/page"),
            "https://test.com/abs/path.js"
        );
    }

    #[test]
    fn test_title_extraction() {
        let html = r#"<html><head><title>Test Page</title></head></html>"#;
        let title = PageFetcher::extract_title(html);
        assert_eq!(title, Some("Test Page".to_string()));
    }

    #[test]
    fn test_description_extraction() {
        let html =
            r#"<html><head><meta name="description" content="Test description"></head></html>"#;
        let desc = PageFetcher::extract_description(html);
        assert_eq!(desc, Some("Test description".to_string()));
    }
}
