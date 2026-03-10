//! BrowerAI HTML5 Parser
//!
//! 基于 html5ever 的 HTML5 解析器，提供：
//! - 标准兼容的 HTML5 解析
//! - DOM 树构建
//! - AI 增强支持（可选）
//!
//! # 示例
//! ```
//! use browerai_html_parser::HtmlParser;
//! use browerai_core::traits::Parser;
//!
//! let parser = HtmlParser::new();
//! let html = "<html><body><h1>Hello</h1></body></html>";
//! let document = parser.parse(html).unwrap();
//! ```

#![warn(missing_docs)]

use browerai_core::{
    traits::{AiModel, Parser},
    BrowserError, Result,
};
use html5ever::{
    parse_document,
    tendril::TendrilSink,
};
use markup5ever_rcdom::{Handle, NodeData, RcDom};

pub mod dom;

pub use dom::{Document, Element, Node, NodeType};

/// HTML 解析器
///
/// 支持传统解析和可选的 AI 增强
pub struct HtmlParser {
    /// 基础解析器
    base_parser: RcDom,
    /// AI 增强器（可选）
    ai_enhancer: Option<Box<dyn AiModel<Input = String, Output = Document>>>,
    /// 是否忽略解析错误
    ignore_errors: bool,
}

impl HtmlParser {
    /// 创建新的 HTML 解析器
    pub fn new() -> Self {
        Self {
            base_parser: RcDom::default(),
            ai_enhancer: None,
            ignore_errors: true,
        }
    }

    /// 创建带 AI 增强的解析器
    pub fn with_ai(
        mut self,
        enhancer: impl AiModel<Input = String, Output = Document> + 'static,
    ) -> Self {
        self.ai_enhancer = Some(Box::new(enhancer));
        self
    }

    /// 设置是否忽略解析错误
    pub fn ignore_errors(mut self, ignore: bool) -> Self {
        self.ignore_errors = ignore;
        self
    }

    /// 解析 HTML 字符串
    ///
    /// # 参数
    /// - `html`: HTML 字符串
    ///
    /// # 返回
    /// - `Ok(Document)`: 解析成功，返回 DOM 文档
    /// - `Err(BrowserError)`: 解析失败
    pub fn parse_string(&self, html: impl AsRef<str>) -> Result<Document> {
        let html = html.as_ref();
        
        // 基础解析
        let dom = parse_document(RcDom::default(), Default::default())
            .from_utf8()
            .read_from(&mut html.as_bytes())
            .map_err(|e| BrowserError::parse(format!("HTML parse error: {}", e)))?;

        // 转换为内部 Document 表示
        let document = self.convert_dom(&dom);

        // 如果 AI 增强器可用，尝试增强
        if let Some(ref ai) = self.ai_enhancer {
            if ai.is_available() {
                match ai.infer(&html.to_string()) {
                    Ok(enhanced) => {
                        log::debug!("AI enhancement applied to HTML");
                        return Ok(enhanced);
                    }
                    Err(e) => {
                        log::warn!("AI enhancement failed: {}, using base result", e);
                    }
                }
            }
        }

        Ok(document)
    }

    /// 将 RcDom 转换为内部 Document
    fn convert_dom(&self, dom: &RcDom) -> Document {
        let mut document = Document::new();
        
        // 处理 document 的所有子节点（通常是 <html> 元素）
        for child in dom.document.children.borrow().iter() {
            self.convert_node(child, &mut document.root);
        }

        document
    }

    /// 递归转换节点
    fn convert_node(&self, handle: &Handle, parent: &mut Node) {
        match &handle.data {
            NodeData::Document => {
                // Document 节点不应该在这里处理，但保留递归逻辑
                for child in handle.children.borrow().iter() {
                    self.convert_node(child, parent);
                }
            }
            NodeData::Element { name, attrs, .. } => {
                let mut element_node = Node::element(&*name.local);
                
                // 获取元素引用以设置属性
                if let NodeType::Element(ref mut element) = element_node.node_type {
                    for attr in attrs.borrow().iter() {
                        element.set_attribute(
                            &*attr.name.local,
                            &*attr.value,
                        );
                    }
                }

                // 递归处理子节点
                for child in handle.children.borrow().iter() {
                    self.convert_node(child, &mut element_node);
                }

                parent.append_child(element_node);
            }
            NodeData::Text { contents } => {
                let text = contents.borrow();
                let text_content = &*text;
                if !text_content.trim().is_empty() {
                    parent.append_child(Node::text(text_content));
                }
            }
            NodeData::Comment { contents } => {
                parent.append_child(Node::comment(contents.clone()));
            }
            _ => {}
        }
    }

    /// 提取所有元素
    pub fn extract_elements(&self, document: &Document, tag_name: &str) -> Vec<Element> {
        document.query_selector_all(tag_name)
    }

    /// 提取所有脚本
    pub fn extract_scripts(&self, document: &Document) -> Vec<String> {
        self.extract_elements(document, "script")
            .iter()
            .filter_map(|el| el.text_content())
            .collect()
    }

    /// 提取所有样式
    pub fn extract_styles(&self, document: &Document) -> Vec<String> {
        self.extract_elements(document, "style")
            .iter()
            .filter_map(|el| el.text_content())
            .collect()
    }

    /// 提取外部资源链接
    pub fn extract_resources(&self, document: &Document) -> Vec<String> {
        let mut resources = Vec::new();

        // CSS 链接
        for link in self.extract_elements(document, "link") {
            if link.get_attribute("rel").as_deref() == Some("stylesheet") {
                if let Some(href) = link.get_attribute("href") {
                    resources.push(href);
                }
            }
        }

        // JS 脚本
        for script in self.extract_elements(document, "script") {
            if let Some(src) = script.get_attribute("src") {
                resources.push(src);
            }
        }

        // 图片
        for img in self.extract_elements(document, "img") {
            if let Some(src) = img.get_attribute("src") {
                resources.push(src);
            }
        }

        resources
    }
}

impl Default for HtmlParser {
    fn default() -> Self {
        Self::new()
    }
}

impl Parser for HtmlParser {
    type Input = str;
    type Output = Document;

    fn parse(&self, input: &Self::Input) -> Result<Self::Output> {
        self.parse_string(input)
    }
}

/// HTML 解析统计
#[derive(Debug, Clone, Default)]
pub struct HtmlParseStats {
    /// 元素数量
    pub element_count: usize,
    /// 文本节点数量
    pub text_node_count: usize,
    /// 注释数量
    pub comment_count: usize,
    /// 最大深度
    pub max_depth: usize,
    /// 解析时间（毫秒）
    pub parse_time_ms: u64,
}

impl HtmlParseStats {
    /// 从文档计算统计
    pub fn from_document(document: &Document) -> Self {
        Self {
            element_count: document.element_count(),
            text_node_count: document.text_node_count(),
            comment_count: document.comment_count(),
            max_depth: document.max_depth(),
            parse_time_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_html() {
        let parser = HtmlParser::new();
        let html = "<html><body><h1>Hello</h1></body></html>";
        let document = parser.parse(html).unwrap();
        
        assert!(!document.root.children().is_empty());
    }

    #[test]
    fn test_extract_scripts() {
        let parser = HtmlParser::new();
        let html = r#"
            <html>
                <head>
                    <script>console.log("test");</script>
                </head>
            </html>
        "#;
        let document = parser.parse(html).unwrap();
        let script_elements = parser.extract_elements(&document, "script");
        
        // 验证至少找到了script元素
        assert!(!script_elements.is_empty(), "Should find at least one script element");
    }

    #[test]
    fn test_extract_resources() {
        let parser = HtmlParser::new();
        let html = r#"
            <html>
                <head>
                    <link rel="stylesheet" href="style.css">
                    <script src="app.js"></script>
                </head>
                <body>
                    <img src="image.png">
                </body>
            </html>
        "#;
        let document = parser.parse(html).unwrap();
        
        // 验证能找到各种资源元素
        let links = parser.extract_elements(&document, "link");
        let scripts = parser.extract_elements(&document, "script");
        let images = parser.extract_elements(&document, "img");
        
        assert!(!links.is_empty(), "Should find link elements");
        assert!(!scripts.is_empty(), "Should find script elements");
        assert!(!images.is_empty(), "Should find img elements");
        
        // 验证能提取href/src属性
        let resources = parser.extract_resources(&document);
        assert!(!resources.is_empty(), "Should extract some resources");
    }

    #[test]
    fn test_parse_malformed_html() {
        let parser = HtmlParser::new().ignore_errors(true);
        let html = "<div><span>Unclosed";  // 故意不完整的 HTML
        let result = parser.parse(html);
        
        // html5ever 会自动修复错误
        assert!(result.is_ok());
    }
}
