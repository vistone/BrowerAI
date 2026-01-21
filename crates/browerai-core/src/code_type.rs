/// Unified code type enumeration for the entire BrowerAI project
///
/// This module provides a centralized CodeType enum used across all code generation,
/// parsing, and analysis modules to avoid duplication and ensure consistency.
use serde::{Deserialize, Serialize};

/// Represents the type of source code or content
///
/// This enum unifies code type definitions that were previously duplicated in:
/// - `browerai-learning/src/code_generator.rs` (CodeType)
/// - `browerai-learning/src/pipeline/learning_pipeline.rs` (FileType)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CodeType {
    /// HTML markup content
    Html,
    /// CSS stylesheet content
    Css,
    /// JavaScript/ECMAScript source code
    #[default]
    JavaScript,
    /// JSON data format
    Json,
    /// TypeScript source code
    TypeScript,
    /// Markdown documentation
    Markdown,
    /// Plain text content
    Text,
}

impl std::fmt::Display for CodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodeType::Html => write!(f, "HTML"),
            CodeType::Css => write!(f, "CSS"),
            CodeType::JavaScript => write!(f, "JavaScript"),
            CodeType::Json => write!(f, "JSON"),
            CodeType::TypeScript => write!(f, "TypeScript"),
            CodeType::Markdown => write!(f, "Markdown"),
            CodeType::Text => write!(f, "Text"),
        }
    }
}

impl From<&str> for CodeType {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "html" | ".html" => CodeType::Html,
            "css" | ".css" => CodeType::Css,
            "javascript" | "js" | ".js" => CodeType::JavaScript,
            "json" | ".json" => CodeType::Json,
            "typescript" | "ts" | ".ts" => CodeType::TypeScript,
            "markdown" | "md" | ".md" => CodeType::Markdown,
            _ => CodeType::Text,
        }
    }
}

impl From<std::path::PathBuf> for CodeType {
    fn from(path: std::path::PathBuf) -> Self {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let path_str = path.to_string_lossy().to_lowercase();

        // Check full path first (e.g., "index.html")
        if path_str.ends_with(".html") {
            return CodeType::Html;
        }
        if path_str.ends_with(".css") {
            return CodeType::Css;
        }
        if path_str.ends_with(".js") {
            return CodeType::JavaScript;
        }

        // Fall back to extension
        extension.as_str().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_type_display() {
        assert_eq!(CodeType::Html.to_string(), "HTML");
        assert_eq!(CodeType::Css.to_string(), "CSS");
        assert_eq!(CodeType::JavaScript.to_string(), "JavaScript");
    }

    #[test]
    fn test_code_type_from_str() {
        assert_eq!(CodeType::from("html"), CodeType::Html);
        assert_eq!(CodeType::from(".css"), CodeType::Css);
        assert_eq!(CodeType::from("js"), CodeType::JavaScript);
        assert_eq!(CodeType::from("unknown"), CodeType::Text);
    }

    #[test]
    fn test_code_type_from_path() {
        assert_eq!(
            CodeType::from(std::path::PathBuf::from("index.html")),
            CodeType::Html
        );
        assert_eq!(
            CodeType::from(std::path::PathBuf::from("style.css")),
            CodeType::Css
        );
        assert_eq!(
            CodeType::from(std::path::PathBuf::from("script.js")),
            CodeType::JavaScript
        );
    }
}
