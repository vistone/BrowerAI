use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("Browser error: {kind}")]
pub struct BrowserError {
    pub kind: ErrorKind,
    #[source]
    pub source: Option<anyhow::Error>,
    pub context: HashMap<String, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Parse,
    Render,
    Network,
    Ai,
    Validation,
    Io,
    Config,
    Security,
    Timeout,
    ResourceLimit,
    Unknown,
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorKind::Parse => write!(f, "Parse"),
            ErrorKind::Render => write!(f, "Render"),
            ErrorKind::Network => write!(f, "Network"),
            ErrorKind::Ai => write!(f, "AI"),
            ErrorKind::Validation => write!(f, "Validation"),
            ErrorKind::Io => write!(f, "IO"),
            ErrorKind::Config => write!(f, "Config"),
            ErrorKind::Security => write!(f, "Security"),
            ErrorKind::Timeout => write!(f, "Timeout"),
            ErrorKind::ResourceLimit => write!(f, "ResourceLimit"),
            ErrorKind::Unknown => write!(f, "Unknown"),
        }
    }
}

impl BrowserError {
    pub fn new(kind: ErrorKind, _message: impl Into<String>) -> Self {
        Self {
            kind,
            source: None,
            context: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn with_source(mut self, source: impl Into<anyhow::Error>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn with_context(mut self, key: impl Into<String>, value: impl ToString) -> Self {
        self.context.insert(key.into(), value.to_string());
        self
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn is_parse_error(&self) -> bool {
        self.kind == ErrorKind::Parse
    }

    pub fn is_render_error(&self) -> bool {
        self.kind == ErrorKind::Render
    }

    pub fn is_network_error(&self) -> bool {
        self.kind == ErrorKind::Network
    }

    pub fn is_ai_error(&self) -> bool {
        self.kind == ErrorKind::Ai
    }

    pub fn context(&self) -> &HashMap<String, String> {
        &self.context
    }

    pub fn add_context(&mut self, key: impl Into<String>, value: impl ToString) {
        self.context.insert(key.into(), value.to_string());
    }
}

pub type Result<T> = std::result::Result<T, anyhow::Error>;

#[macro_export]
macro_rules! context {
    ($err:expr, $kind:expr) => {
        $err.context(stringify!($kind))
    };
}

pub mod parse {
    use super::*;

    #[derive(Debug, Error)]
    #[error("HTML parse error at {line}:{col}: {message}")]
    pub struct HtmlParseError {
        pub message: String,
        pub line: usize,
        pub col: usize,
    }

    impl HtmlParseError {
        pub fn new(message: impl Into<String>, line: usize, col: usize) -> Self {
            Self {
                message: message.into(),
                line,
                col,
            }
        }
    }

    #[derive(Debug, Error)]
    #[error("CSS parse error: {message}")]
    pub struct CssParseError {
        pub message: String,
        pub line: Option<usize>,
    }

    impl CssParseError {
        pub fn new(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                line: None,
            }
        }

        pub fn at_line(mut self, line: usize) -> Self {
            self.line = Some(line);
            self
        }
    }

    #[derive(Debug, Error)]
    #[error("JavaScript parse error: {message}")]
    pub struct JsParseError {
        pub message: String,
        pub line: Option<usize>,
        pub column: Option<usize>,
    }

    impl JsParseError {
        pub fn new(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                line: None,
                column: None,
            }
        }

        pub fn at_location(mut self, line: usize, col: usize) -> Self {
            self.line = Some(line);
            self.column = Some(col);
            self
        }
    }
}

pub mod render {
    use super::*;

    #[derive(Debug, Error)]
    #[error("Layout error: {message}")]
    pub struct LayoutError {
        pub message: String,
    }

    impl LayoutError {
        pub fn new(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
            }
        }
    }

    #[derive(Debug, Error)]
    #[error("Paint error: {message}")]
    pub struct PaintError {
        pub message: String,
    }

    impl PaintError {
        pub fn new(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
            }
        }
    }
}

pub mod network {
    use super::*;

    #[derive(Debug, Error)]
    pub struct NetworkError {
        pub message: String,
        pub status_code: Option<u16>,
        pub url: Option<PathBuf>,
        pub method: Option<String>,
    }

    impl std::fmt::Display for NetworkError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Network error: {}", self.message)?;
            if let Some(code) = self.status_code {
                write!(f, " (status: {})", code)?;
            }
            Ok(())
        }
    }

    impl NetworkError {
        pub fn new(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                status_code: None,
                url: None,
                method: None,
            }
        }

        pub fn with_status_code(mut self, code: u16) -> Self {
            self.status_code = Some(code);
            self
        }

        pub fn for_url(mut self, url: impl Into<PathBuf>) -> Self {
            self.url = Some(url.into());
            self
        }
    }

    #[derive(Debug, Error)]
    #[error("Timeout: {operation} exceeded {timeout_ms}ms")]
    pub struct TimeoutError {
        pub operation: String,
        pub timeout_ms: u64,
    }

    impl TimeoutError {
        pub fn new(operation: impl Into<String>, timeout_ms: u64) -> Self {
            Self {
                operation: operation.into(),
                timeout_ms,
            }
        }
    }
}

pub mod ai {
    use super::*;

    #[derive(Debug, Error)]
    pub struct AiError {
        pub message: String,
        pub model: Option<String>,
        pub recoverable: bool,
    }

    impl std::fmt::Display for AiError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "AI error: {}", self.message)?;
            if let Some(ref model) = self.model {
                write!(f, " [model: {}]", model)?;
            }
            Ok(())
        }
    }

    impl AiError {
        pub fn new(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                model: None,
                recoverable: true,
            }
        }

        pub fn for_model(mut self, model: impl Into<String>) -> Self {
            self.model = Some(model.into());
            self
        }

        pub fn unrecoverable(mut self) -> Self {
            self.recoverable = false;
            self
        }
    }

    #[derive(Debug, Error)]
    #[error("Model loading error: {message}")]
    pub struct ModelLoadError {
        pub message: String,
        pub model_path: Option<PathBuf>,
    }

    impl ModelLoadError {
        pub fn new(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                model_path: None,
            }
        }

        pub fn at_path(mut self, path: impl Into<PathBuf>) -> Self {
            self.model_path = Some(path.into());
            self
        }
    }
}

pub mod config {
    use super::*;

    #[derive(Debug, Error)]
    #[error("Configuration error: {message}")]
    pub struct ConfigError {
        pub message: String,
        pub config_key: Option<String>,
    }

    impl ConfigError {
        pub fn new(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                config_key: None,
            }
        }

        pub fn for_key(mut self, key: impl Into<String>) -> Self {
            self.config_key = Some(key.into());
            self
        }
    }

    #[derive(Debug, Error)]
    #[error("Validation error: {field}: {message}")]
    pub struct ValidationError {
        pub field: String,
        pub message: String,
    }

    impl ValidationError {
        pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
            Self {
                field: field.into(),
                message: message.into(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = BrowserError::new(ErrorKind::Parse, "test message");
        assert_eq!(error.kind(), ErrorKind::Parse);
    }

    #[test]
    fn test_error_with_context() {
        let error = BrowserError::new(ErrorKind::Network, "connection failed")
            .with_context("url", "https://example.com");
        assert_eq!(error.context().len(), 1);
    }

    #[test]
    fn test_error_kind_display() {
        assert_eq!(ErrorKind::Parse.to_string(), "Parse");
        assert_eq!(ErrorKind::Render.to_string(), "Render");
    }

    #[test]
    fn test_html_parse_error() {
        let error = parse::HtmlParseError::new("Unexpected token", 10, 5);
        assert!(error.to_string().contains("10:5"));
    }

    #[test]
    fn test_network_error() {
        let error = network::NetworkError::new("Connection refused").with_status_code(500);
        assert!(error.to_string().contains("500"));
    }

    #[test]
    fn test_ai_error() {
        let error = ai::AiError::new("Model inference failed")
            .for_model("bert-base")
            .unrecoverable();
        assert!(!error.recoverable);
        assert!(error.to_string().contains("bert-base"));
    }

    #[test]
    fn test_validation_error() {
        let error = config::ValidationError::new("email", "Invalid format");
        assert_eq!(error.field, "email");
    }

    #[test]
    fn test_timeout_error() {
        let error = network::TimeoutError::new("API call", 5000);
        assert!(error.to_string().contains("5000ms"));
    }
}
