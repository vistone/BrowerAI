use anyhow::{Context, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// HTTP request method
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    HEAD,
}

/// HTTP request structure
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body: Option<Vec<u8>>,
}

impl HttpRequest {
    /// Create a new GET request
    pub fn get(url: impl Into<String>) -> Self {
        Self {
            method: HttpMethod::GET,
            url: url.into(),
            headers: HashMap::new(),
            body: None,
        }
    }

    /// Add a header to the request
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Set the request body
    pub fn with_body(mut self, body: Vec<u8>) -> Self {
        self.body = Some(body);
        self
    }
}

/// HTTP response structure
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status_code: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
    pub response_time: Duration,
}

impl HttpResponse {
    /// Get response body as string
    pub fn text(&self) -> Result<String> {
        String::from_utf8(self.body.clone()).context("Failed to convert response body to UTF-8")
    }

    /// Check if the response is successful (2xx)
    pub fn is_success(&self) -> bool {
        self.status_code >= 200 && self.status_code < 300
    }
}

/// HTTP client for making requests
pub struct HttpClient {
    user_agent: String,
    timeout: Duration,
}

impl HttpClient {
    /// Create a new HTTP client
    pub fn new() -> Self {
        Self {
            user_agent: "BrowerAI/0.1.0".to_string(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Set custom user agent
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }

    /// Set request timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Execute an HTTP request (stub implementation)
    pub fn execute(&self, mut request: HttpRequest) -> Result<HttpResponse> {
        log::info!("Executing {:?} request to {}", request.method, request.url);

        // Add default headers
        if !request.headers.contains_key("User-Agent") {
            request
                .headers
                .insert("User-Agent".to_string(), self.user_agent.clone());
        }

        let start = Instant::now();

        // Stub implementation - in real implementation, this would use reqwest or similar
        let response = self.execute_stub(&request)?;

        let response_time = start.elapsed();
        log::info!("Request completed in {:?}", response_time);

        Ok(HttpResponse {
            status_code: response.0,
            headers: response.1,
            body: response.2,
            response_time,
        })
    }

    /// Stub implementation for testing
    fn execute_stub(
        &self,
        request: &HttpRequest,
    ) -> Result<(u16, HashMap<String, String>, Vec<u8>)> {
        // Simulate a successful response
        let mut headers = HashMap::new();
        headers.insert(
            "Content-Type".to_string(),
            "text/html; charset=utf-8".to_string(),
        );
        headers.insert("Content-Length".to_string(), "100".to_string());

        let body = format!(
            "<html><body><h1>Stub Response for {}</h1></body></html>",
            request.url
        );

        Ok((200, headers, body.into_bytes()))
    }

    /// Convenience method for GET requests
    pub fn get(&self, url: impl Into<String>) -> Result<HttpResponse> {
        let request = HttpRequest::get(url);
        self.execute(request)
    }
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_request_creation() {
        let request = HttpRequest::get("https://example.com");
        assert_eq!(request.method, HttpMethod::GET);
        assert_eq!(request.url, "https://example.com");
    }

    #[test]
    fn test_http_request_with_headers() {
        let request = HttpRequest::get("https://example.com").with_header("Accept", "text/html");

        assert_eq!(
            request.headers.get("Accept"),
            Some(&"text/html".to_string())
        );
    }

    #[test]
    fn test_http_client_creation() {
        let client = HttpClient::new();
        assert_eq!(client.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_http_client_execute() {
        let client = HttpClient::new();
        let request = HttpRequest::get("https://example.com");

        let result = client.execute(request);
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.status_code, 200);
        assert!(response.is_success());
    }

    #[test]
    fn test_http_response_text() {
        let client = HttpClient::new();
        let response = client.get("https://example.com").unwrap();

        let text = response.text();
        assert!(text.is_ok());
        assert!(text.unwrap().contains("Stub Response"));
    }

    #[test]
    fn test_http_client_with_custom_user_agent() {
        let client = HttpClient::new().with_user_agent("CustomAgent/1.0");

        assert_eq!(client.user_agent, "CustomAgent/1.0");
    }
}
