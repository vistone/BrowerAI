//! 错误处理系统
//!
//! 提供 BrowerAI 的统一错误类型，支持：
//! - 详细的错误分类
//! - 错误链追踪
//! - 用户友好的错误消息

use std::fmt;
use std::path::PathBuf;

/// BrowerAI 统一的 Result 类型
pub type Result<T> = std::result::Result<T, BrowserError>;

/// 错误分类
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// 解析错误（HTML/CSS/JS）
    Parse,
    /// 渲染错误
    Render,
    /// AI/ML 错误
    Ai,
    /// 网络错误
    Network,
    /// IO 错误
    Io,
    /// 配置错误
    Config,
    /// 反混淆错误
    Deobfuscation,
    /// 学习错误
    Learning,
    /// 验证错误
    Validation,
    /// 未知错误
    Unknown,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::Parse => write!(f, "Parse"),
            ErrorKind::Render => write!(f, "Render"),
            ErrorKind::Ai => write!(f, "AI"),
            ErrorKind::Network => write!(f, "Network"),
            ErrorKind::Io => write!(f, "IO"),
            ErrorKind::Config => write!(f, "Config"),
            ErrorKind::Deobfuscation => write!(f, "Deobfuscation"),
            ErrorKind::Learning => write!(f, "Learning"),
            ErrorKind::Validation => write!(f, "Validation"),
            ErrorKind::Unknown => write!(f, "Unknown"),
        }
    }
}

/// BrowerAI 统一的错误类型
#[derive(Debug, Clone)]
pub struct BrowserError {
    /// 错误分类
    pub kind: ErrorKind,
    /// 错误消息
    pub message: String,
    /// 错误来源（文件路径等）
    pub source: Option<String>,
    /// 行号（如果有）
    pub line: Option<usize>,
    /// 列号（如果有）
    pub column: Option<usize>,
}

impl BrowserError {
    /// 创建新的错误
    #[allow(dead_code)]
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            source: None,
            line: None,
            column: None,
        }
    }

    /// 添加来源信息
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// 添加位置信息
    pub fn with_location(mut self, line: usize, column: usize) -> Self {
        self.line = Some(line);
        self.column = Some(column);
        self
    }

    /// 创建解析错误
    #[allow(dead_code)]
    pub fn parse(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Parse, message)
    }

    /// 创建渲染错误
    #[allow(dead_code)]
    pub fn render(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Render, message)
    }

    /// 创建 AI 错误
    #[allow(dead_code)]
    pub fn ai(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Ai, message)
    }

    /// 创建网络错误
    #[allow(dead_code)]
    pub fn network(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Network, message)
    }

    /// 创建 IO 错误
    #[allow(dead_code)]
    pub fn io(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Io, message)
    }

    /// 创建配置错误
    #[allow(dead_code)]
    pub fn config(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Config, message)
    }

    /// 创建反混淆错误
    pub fn deobfuscation(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Deobfuscation, message)
    }

    /// 创建学习错误
    #[allow(dead_code)]
    pub fn learning(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Learning, message)
    }

    /// 创建验证错误
    #[allow(dead_code)]
    pub fn validation(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Validation, message)
    }
}

impl fmt::Display for BrowserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.kind, self.message)?;

        if let Some(ref source) = self.source {
            write!(f, " (source: {}", source)?;
            if let (Some(line), Some(column)) = (self.line, self.column) {
                write!(f, ":{}:{}", line, column)?;
            }
            write!(f, ")")?;
        }

        Ok(())
    }
}

impl std::error::Error for BrowserError {}

// 从标准 IO 错误转换
impl From<std::io::Error> for BrowserError {
    fn from(err: std::io::Error) -> Self {
        BrowserError::io(err.to_string())
    }
}

// 从 serde_json 错误转换
impl From<serde_json::Error> for BrowserError {
    fn from(err: serde_json::Error) -> Self {
        BrowserError::parse(format!("JSON error: {}", err)).with_location(err.line(), err.column())
    }
}

/// 解析错误子类型
pub mod parse {
    use super::*;

    /// HTML 解析错误
    #[derive(Debug, Clone)]
    pub struct HtmlParseError {
        /// 人类可读的错误信息。
        pub message: String,
        /// 错误所在行号（1-based）。
        pub line: Option<usize>,
        /// 错误所在列号（1-based）。
        pub column: Option<usize>,
    }

    impl HtmlParseError {
        /// 使用错误消息创建 HTML 解析错误。
        pub fn new(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                line: None,
                column: None,
            }
        }

        /// 附加错误位置信息（行、列）。
        pub fn with_location(mut self, line: usize, column: usize) -> Self {
            self.line = Some(line);
            self.column = Some(column);
            self
        }
    }

    impl fmt::Display for HtmlParseError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "HTML parse error: {}", self.message)?;
            if let (Some(line), Some(column)) = (self.line, self.column) {
                write!(f, " at {}:{}", line, column)?;
            }
            Ok(())
        }
    }

    impl std::error::Error for HtmlParseError {}

    /// CSS 解析错误
    #[derive(Debug, Clone)]
    pub struct CssParseError {
        /// 人类可读的错误信息。
        pub message: String,
        /// 错误位置描述（如 selector/offset）。
        pub location: Option<String>,
    }

    impl CssParseError {
        /// 使用错误消息创建 CSS 解析错误。
        pub fn new(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                location: None,
            }
        }
    }

    /// JS 解析错误
    #[derive(Debug, Clone)]
    pub struct JsParseError {
        /// 人类可读的错误信息。
        pub message: String,
        /// 错误所在行号（1-based）。
        pub line: Option<usize>,
        /// 错误所在列号（1-based）。
        pub column: Option<usize>,
    }
}

/// AI 错误子类型
pub mod ai {
    use super::*;

    /// 模型加载错误
    #[derive(Debug, Clone)]
    pub struct ModelLoadError {
        /// 模型文件路径。
        pub model_path: PathBuf,
        /// 加载失败原因。
        pub reason: String,
    }

    impl ModelLoadError {
        /// 创建模型加载错误。
        pub fn new(path: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
            Self {
                model_path: path.into(),
                reason: reason.into(),
            }
        }
    }

    impl fmt::Display for ModelLoadError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "Failed to load model from {:?}: {}",
                self.model_path, self.reason
            )
        }
    }

    impl std::error::Error for ModelLoadError {}

    /// 推理错误
    #[derive(Debug, Clone)]
    pub struct InferenceError {
        /// 人类可读的错误信息。
        pub message: String,
        /// 可选的模型名称。
        pub model_name: Option<String>,
    }

    impl InferenceError {
        /// 使用错误消息创建推理错误。
        pub fn new(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                model_name: None,
            }
        }

        /// 设置产生该错误的模型名称。
        pub fn with_model(mut self, name: impl Into<String>) -> Self {
            self.model_name = Some(name.into());
            self
        }
    }
}

/// 网络错误子类型
pub mod network {
    use super::*;

    /// 超时错误
    #[derive(Debug, Clone)]
    pub struct TimeoutError {
        /// 请求 URL。
        pub url: String,
        /// 超时时长（秒）。
        pub timeout_secs: u64,
    }

    impl TimeoutError {
        /// 创建超时错误。
        pub fn new(url: impl Into<String>, secs: u64) -> Self {
            Self {
                url: url.into(),
                timeout_secs: secs,
            }
        }
    }

    impl fmt::Display for TimeoutError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "Request to {} timed out after {}s",
                self.url, self.timeout_secs
            )
        }
    }

    impl std::error::Error for TimeoutError {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = BrowserError::parse("Invalid HTML");
        assert_eq!(err.kind, ErrorKind::Parse);
        assert!(err.message.contains("Invalid HTML"));
    }

    #[test]
    fn test_error_with_location() {
        let err = BrowserError::parse("Syntax error").with_location(10, 5);

        assert_eq!(err.line, Some(10));
        assert_eq!(err.column, Some(5));
    }

    #[test]
    fn test_error_display() {
        let err = BrowserError::parse("Test error")
            .with_source("test.html")
            .with_location(1, 10);

        let display = format!("{}", err);
        assert!(display.contains("Parse"));
        assert!(display.contains("Test error"));
        assert!(display.contains("test.html"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let browser_err: BrowserError = io_err.into();

        assert_eq!(browser_err.kind, ErrorKind::Io);
    }
}
