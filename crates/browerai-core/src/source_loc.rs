/// Unified source location information for the entire BrowerAI project
///
/// This module provides centralized types for representing source code locations,
/// unifying previously duplicated definitions across different modules.
use serde::{Deserialize, Serialize};

/// Represents a location in source code
///
/// This struct unifies location definitions that were previously duplicated in:
/// - `browerai-js-analyzer/src/types.rs` (LocationInfo)
/// - `browerai-learning/src/behavior_record.rs` (CodeLocation)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct SourceLocation {
    /// File path or identifier
    #[serde(default)]
    pub file: String,
    /// Line number (1-based)
    #[serde(default)]
    pub line: usize,
    /// Column number (0-based)
    #[serde(default)]
    pub column: usize,
    /// Character offset from start of file
    #[serde(default)]
    pub start: usize,
    /// Character offset to end of range
    #[serde(default)]
    pub end: usize,
}

impl SourceLocation {
    /// Create a new source location
    pub fn new(
        file: impl Into<String>,
        line: usize,
        column: usize,
        start: usize,
        end: usize,
    ) -> Self {
        Self {
            file: file.into(),
            line,
            column,
            start,
            end,
        }
    }

    /// Create a location from line and column only
    pub fn from_line_column(file: impl Into<String>, line: usize, column: usize) -> Self {
        Self {
            file: file.into(),
            line,
            column,
            start: 0,
            end: 0,
        }
    }

    /// Get the byte range as a Rust range
    pub fn byte_range(&self) -> std::ops::Range<usize> {
        self.start..self.end
    }

    /// Get the line range (inclusive start, exclusive end)
    pub fn line_range(&self) -> std::ops::Range<usize> {
        self.line..self.line + 1
    }

    /// Check if this location is valid (has non-zero positions)
    pub fn is_valid(&self) -> bool {
        self.line > 0 || self.column > 0 || self.start > 0 || self.end > 0
    }

    /// Calculate the length in characters
    pub fn length(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Format as a display string (e.g., "file.rs:10:5")
    pub fn to_display(&self) -> String {
        format!("{}:{}:{}", self.file, self.line, self.column)
    }
}

impl std::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_display())
    }
}

/// Represents a span of source code (start and end locations)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct SourceSpan {
    /// Starting location
    #[serde(default)]
    pub start: SourceLocation,
    /// Ending location
    #[serde(default)]
    pub end: SourceLocation,
}

impl SourceSpan {
    /// Create a new span from start and end locations
    pub fn new(start: SourceLocation, end: SourceLocation) -> Self {
        Self { start, end }
    }

    /// Create a span covering a single location
    pub fn at(location: SourceLocation) -> Self {
        Self {
            start: location.clone(),
            end: location,
        }
    }

    /// Get the total length of the span
    pub fn length(&self) -> usize {
        self.end.length().max(self.start.length())
    }

    /// Check if the span is valid
    pub fn is_valid(&self) -> bool {
        self.start.is_valid() || self.end.is_valid()
    }

    /// Check if this span contains another location
    pub fn contains(&self, location: &SourceLocation) -> bool {
        self.start.start <= location.start && location.end <= self.end.end
    }
}

/// Kind of source (file, generated, memory, etc.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum SourceKind {
    /// Physical file on disk
    File,
    /// Generated code
    Generated,
    /// In-memory code (eval, dynamic)
    Memory,
    /// Standard input
    Stdin,
    /// Unknown origin
    #[default]
    Unknown,
}

/// Container for source code with its location information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SourceInfo {
    /// Location of the source
    #[serde(default)]
    pub location: SourceLocation,
    /// Kind of source
    #[serde(default)]
    pub kind: SourceKind,
    /// The source code content
    #[serde(default)]
    pub content: String,
    /// Hash of the content for change detection
    #[serde(default)]
    pub content_hash: String,
}

impl SourceInfo {
    /// Create source info from content with default location
    pub fn from_content(content: impl Into<String>) -> Self {
        let content = content.into();
        let content_hash = format!("{:x}", md5::compute(&content));
        Self {
            location: SourceLocation::default(),
            kind: SourceKind::Unknown,
            content,
            content_hash,
        }
    }

    /// Create source info with full location
    pub fn new(location: SourceLocation, kind: SourceKind, content: impl Into<String>) -> Self {
        let content = content.into();
        let content_hash = format!("{:x}", md5::compute(&content));
        Self {
            location,
            kind,
            content,
            content_hash,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_location_new() {
        let loc = SourceLocation::new("test.rs", 10, 5, 100, 150);
        assert_eq!(loc.file, "test.rs");
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, 5);
        assert_eq!(loc.start, 100);
        assert_eq!(loc.end, 150);
    }

    #[test]
    fn test_source_location_is_valid() {
        assert!(!SourceLocation::default().is_valid());
        assert!(SourceLocation::new("test.rs", 1, 0, 0, 0).is_valid());
    }

    #[test]
    fn test_source_location_length() {
        let loc = SourceLocation::new("test.rs", 1, 0, 100, 150);
        assert_eq!(loc.length(), 50);
    }

    #[test]
    fn test_source_span() {
        let start = SourceLocation::new("test.rs", 10, 5, 100, 150);
        let end = SourceLocation::new("test.rs", 15, 10, 200, 250);
        let span = SourceSpan::new(start, end);
        assert_eq!(span.length(), 50);
    }

    #[test]
    fn test_source_info_content_hash() {
        let info = SourceInfo::from_content("test content");
        assert_eq!(info.content, "test content");
        assert!(!info.content_hash.is_empty());
    }
}
