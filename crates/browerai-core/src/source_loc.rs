//! 源代码位置信息
//!
//! 提供代码位置、范围等信息的类型

use serde::{Deserialize, Serialize};

/// 源代码位置
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct SourceLocation {
    /// 行号（1-based）
    pub line: usize,
    /// 列号（1-based）
    pub column: usize,
}

impl SourceLocation {
    /// 创建新的位置
    pub fn new(line: usize, column: usize) -> Self {
        Self { line, column }
    }

    /// 位置是否在另一个位置之前
    pub fn is_before(&self, other: &Self) -> bool {
        self.line < other.line || (self.line == other.line && self.column < other.column)
    }

    /// 位置是否在另一个位置之后
    pub fn is_after(&self, other: &Self) -> bool {
        self.line > other.line || (self.line == other.line && self.column > other.column)
    }
}

impl std::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// 源代码范围
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct SourceSpan {
    /// 起始位置
    pub start: SourceLocation,
    /// 结束位置
    pub end: SourceLocation,
}

impl SourceSpan {
    /// 创建新的范围
    pub fn new(start: SourceLocation, end: SourceLocation) -> Self {
        Self { start, end }
    }

    /// 从行列创建
    pub fn from_lines(
        start_line: usize,
        start_col: usize,
        end_line: usize,
        end_col: usize,
    ) -> Self {
        Self {
            start: SourceLocation::new(start_line, start_col),
            end: SourceLocation::new(end_line, end_col),
        }
    }

    /// 检查位置是否在范围内
    pub fn contains(&self, loc: &SourceLocation) -> bool {
        !loc.is_before(&self.start) && !loc.is_after(&self.end)
    }

    /// 检查两个范围是否重叠
    pub fn overlaps(&self, other: &Self) -> bool {
        !(self.end.is_before(&other.start) || self.start.is_after(&other.end))
    }

    /// 合并两个范围
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            start: if self.start.is_before(&other.start) {
                self.start
            } else {
                other.start
            },
            end: if self.end.is_after(&other.end) {
                self.end
            } else {
                other.end
            },
        }
    }
}

impl std::fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} - {}", self.start, self.end)
    }
}

/// 源代码信息
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceInfo {
    /// 文件名
    pub filename: Option<String>,
    /// 源代码内容
    pub source: Option<String>,
    /// 位置
    pub span: SourceSpan,
}

impl SourceInfo {
    /// 创建新的源信息
    pub fn new(span: SourceSpan) -> Self {
        Self {
            filename: None,
            source: None,
            span,
        }
    }

    /// 设置文件名
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    /// 设置源代码
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// 获取指定范围的源代码片段
    pub fn snippet(&self) -> Option<String> {
        let source = self.source.as_ref()?;
        let lines: Vec<&str> = source.lines().collect();
        
        if self.span.start.line == 0 || self.span.start.line > lines.len() {
            return None;
        }

        let start_line = self.span.start.line - 1;
        let end_line = self.span.end.line.min(lines.len()) - 1;

        if start_line == end_line {
            // 单行
            let line = lines[start_line];
            let start_col = self.span.start.column.saturating_sub(1);
            let end_col = self.span.end.column.min(line.len());
            if start_col < end_col {
                Some(line[start_col..end_col].to_string())
            } else {
                Some(line.to_string())
            }
        } else {
            // 多行
            let mut result = String::new();
            for i in start_line..=end_line {
                result.push_str(lines[i]);
                result.push('\n');
            }
            Some(result)
        }
    }

    /// 格式化错误消息
    pub fn format_error(&self, message: &str) -> String {
        let mut result = String::new();
        
        if let Some(ref filename) = self.filename {
            result.push_str(&format!("{}:{}:{}\n", 
                filename, 
                self.span.start.line, 
                self.span.start.column
            ));
        }
        
        if let Some(snippet) = self.snippet() {
            result.push_str(&format!("  | {}\n", snippet.trim()));
            result.push_str(&format!("  | {}^ {}\n", 
                " ".repeat(self.span.start.column.saturating_sub(1)),
                message
            ));
        } else {
            result.push_str(message);
        }
        
        result
    }
}

/// 源代码种类
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceKind {
    /// HTML
    Html,
    /// CSS
    Css,
    /// JavaScript
    JavaScript,
    /// TypeScript
    TypeScript,
    /// 其他
    Other,
}

impl SourceKind {
    /// 获取文件扩展名
    pub fn extension(&self) -> &'static str {
        match self {
            SourceKind::Html => "html",
            SourceKind::Css => "css",
            SourceKind::JavaScript => "js",
            SourceKind::TypeScript => "ts",
            SourceKind::Other => "txt",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_location() {
        let loc1 = SourceLocation::new(10, 5);
        let loc2 = SourceLocation::new(10, 10);
        let loc3 = SourceLocation::new(11, 1);

        assert!(loc1.is_before(&loc2));
        assert!(loc2.is_before(&loc3));
        assert!(!loc2.is_before(&loc1));
    }

    #[test]
    fn test_source_span() {
        let span1 = SourceSpan::from_lines(1, 1, 3, 10);
        let span2 = SourceSpan::from_lines(2, 5, 4, 8);

        assert!(span1.overlaps(&span2));

        let merged = span1.merge(&span2);
        assert_eq!(merged.start.line, 1);
        assert_eq!(merged.end.line, 4);
    }

    #[test]
    fn test_source_info_snippet() {
        let source = "line 1\nline 2\nline 3".to_string();
        let info = SourceInfo::new(SourceSpan::from_lines(2, 1, 2, 6))
            .with_source(source);

        assert_eq!(info.snippet(), Some("line 2".to_string()));
    }

    #[test]
    fn test_format_error() {
        let source = "function test() {".to_string();
        let info = SourceInfo::new(SourceSpan::from_lines(1, 1, 1, 17))
            .with_filename("test.js")
            .with_source(source);

        let formatted = info.format_error("missing closing brace");
        assert!(formatted.contains("test.js"));
        assert!(formatted.contains("function test()"));
        assert!(formatted.contains("missing closing brace"));
    }
}
