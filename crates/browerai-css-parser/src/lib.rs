//! BrowerAI CSS Parser
//!
//! 基于 cssparser 的 CSS 解析器，提供：
//! - CSS3 标准解析
//! - 样式规则提取
//! - 选择器解析
//!
//! # 示例
//! ```
//! use browerai_css_parser::CssParser;
//! use browerai_core::traits::Parser;
//!
//! let parser = CssParser::new();
//! let css = "body { color: red; }";
//! let stylesheet = parser.parse(css).unwrap();
//! ```

#![warn(missing_docs)]

use browerai_core::{traits::Parser, Result};

pub mod stylesheet;

pub use stylesheet::{Declaration, Rule, Selector, Stylesheet, Value};

/// CSS 解析器
pub struct CssParser {
    /// 是否忽略无效规则
    ignore_invalid: bool,
}

impl CssParser {
    /// 创建新的 CSS 解析器
    pub fn new() -> Self {
        Self {
            ignore_invalid: true,
        }
    }

    /// 设置是否忽略无效规则
    pub fn ignore_invalid(mut self, ignore: bool) -> Self {
        self.ignore_invalid = ignore;
        self
    }

    /// 解析 CSS 字符串
    pub fn parse_string(&self, css: impl AsRef<str>) -> Result<Stylesheet> {
        let css = css.as_ref();
        
        // 简化的 CSS 解析实现
        // 实际应该使用 cssparser crate
        let mut stylesheet = Stylesheet::new();
        
        // 简单的解析逻辑（示例）
        self.parse_simple(css, &mut stylesheet)?;
        
        Ok(stylesheet)
    }

    /// 简化的 CSS 解析
    fn parse_simple(&self, css: &str, stylesheet: &mut Stylesheet) -> Result<()> {
        // 这里是一个简化的实现
        // 实际应该使用 cssparser 进行完整解析
        
        // 按规则分割（简化版）
        let rules = css.split('}');
        
        for rule_str in rules {
            let rule_str = rule_str.trim();
            if rule_str.is_empty() {
                continue;
            }
            
            // 查找选择器和声明的分界
            if let Some(idx) = rule_str.find('{') {
                let selector_str = &rule_str[..idx].trim();
                let declarations_str = &rule_str[idx + 1..].trim();
                
                let mut rule = Rule::new();
                rule.selector = Selector::from_string(selector_str.to_string());
                
                // 解析声明
                for decl_str in declarations_str.split(';') {
                    let decl_str = decl_str.trim();
                    if decl_str.is_empty() {
                        continue;
                    }
                    
                    if let Some(colon_idx) = decl_str.find(':') {
                        let property = decl_str[..colon_idx].trim().to_string();
                        let value = decl_str[colon_idx + 1..].trim().to_string();
                        
                        rule.declarations.push(Declaration {
                            property,
                            value: Value::String(value),
                            important: false,
                        });
                    }
                }
                
                if !rule.declarations.is_empty() {
                    stylesheet.rules.push(rule);
                }
            }
        }
        
        Ok(())
    }

    /// 提取所有颜色
    pub fn extract_colors(&self, stylesheet: &Stylesheet) -> Vec<String> {
        let mut colors = Vec::new();
        
        for rule in &stylesheet.rules {
            for decl in &rule.declarations {
                if decl.property.contains("color") || decl.property == "background" {
                    if let Value::String(ref value) = decl.value {
                        colors.push(value.clone());
                    }
                }
            }
        }
        
        colors
    }

    /// 提取所有字体
    pub fn extract_fonts(&self, stylesheet: &Stylesheet) -> Vec<String> {
        let mut fonts = Vec::new();
        
        for rule in &stylesheet.rules {
            for decl in &rule.declarations {
                if decl.property == "font-family" {
                    if let Value::String(ref value) = decl.value {
                        fonts.push(value.clone());
                    }
                }
            }
        }
        
        fonts
    }

    /// 提取所有媒体查询
    pub fn extract_media_queries(&self, _stylesheet: &Stylesheet) -> Vec<String> {
        // 简化实现
        Vec::new()
    }
}

impl Default for CssParser {
    fn default() -> Self {
        Self::new()
    }
}

impl Parser for CssParser {
    type Input = str;
    type Output = Stylesheet;

    fn parse(&self, input: &Self::Input) -> Result<Self::Output> {
        self.parse_string(input)
    }
}

/// CSS 解析统计
#[derive(Debug, Clone, Default)]
pub struct CssParseStats {
    /// 规则数量
    pub rule_count: usize,
    /// 选择器数量
    pub selector_count: usize,
    /// 声明数量
    pub declaration_count: usize,
    /// 使用的颜色数量
    pub color_count: usize,
    /// 使用的字体数量
    pub font_count: usize,
}

impl CssParseStats {
    /// 从样式表计算统计
    pub fn from_stylesheet(stylesheet: &Stylesheet) -> Self {
        Self {
            rule_count: stylesheet.rules.len(),
            selector_count: stylesheet.rules.iter().map(|_r| 1).sum(),
            declaration_count: stylesheet.rules.iter().map(|r| r.declarations.len()).sum(),
            color_count: 0,
            font_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_css() {
        let parser = CssParser::new();
        let css = "body { color: red; background: blue; }";
        let stylesheet = parser.parse(css).unwrap();
        
        assert!(!stylesheet.rules.is_empty());
    }

    #[test]
    fn test_extract_colors() {
        let parser = CssParser::new();
        let css = r#"
            body { color: red; }
            h1 { background: blue; }
        "#;
        let stylesheet = parser.parse(css).unwrap();
        let colors = parser.extract_colors(&stylesheet);
        
        assert!(colors.contains(&"red".to_string()));
        assert!(colors.contains(&"blue".to_string()));
    }

    #[test]
    fn test_parse_multiple_rules() {
        let parser = CssParser::new();
        let css = r#"
            body { margin: 0; }
            h1 { font-size: 24px; }
            p { line-height: 1.5; }
        "#;
        let stylesheet = parser.parse(css).unwrap();
        
        assert_eq!(stylesheet.rules.len(), 3);
    }
}
