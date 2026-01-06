use thiserror::Error;

/// Core error types for BrowerAI
#[derive(Error, Debug)]
pub enum BrowserError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Render error: {0}")]
    RenderError(String),

    #[error("AI model error: {0}")]
    ModelError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Result type alias for BrowerAI operations
pub type Result<T> = std::result::Result<T, BrowserError>;
