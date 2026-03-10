//! DOM 类型定义
//!
//! 提供 BrowerAI 的 DOM 表示

use std::collections::HashMap;

/// DOM 文档
#[derive(Debug, Clone)]
pub struct Document {
    /// 文档根节点
    pub root: Node,
    /// 文档标题
    pub title: Option<String>,
    /// 文档类型
    pub doctype: Option<String>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl Document {
    /// 创建新的文档
    pub fn new() -> Self {
        Self {
            root: Node::element("html"),
            title: None,
            doctype: Some("html".to_string()),
            metadata: HashMap::new(),
        }
    }

    /// 设置文档标题
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.title = Some(title.into());
    }

    /// 获取文档标题
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// 查询选择器
    pub fn query_selector(&self, selector: &str) -> Option<Element> {
        self.root.query_selector(selector)
    }

    /// 查询所有匹配的元素
    pub fn query_selector_all(&self, selector: &str) -> Vec<Element> {
        self.root.query_selector_all(selector)
    }

    /// 获取所有元素数量
    pub fn element_count(&self) -> usize {
        self.root.element_count()
    }

    /// 获取文本节点数量
    pub fn text_node_count(&self) -> usize {
        self.root.text_node_count()
    }

    /// 获取注释数量
    pub fn comment_count(&self) -> usize {
        self.root.comment_count()
    }

    /// 获取最大深度
    pub fn max_depth(&self) -> usize {
        self.root.max_depth()
    }

    /// 序列化为 HTML 字符串
    pub fn to_html(&self) -> String {
        let mut result = String::new();
        
        if let Some(ref doctype) = self.doctype {
            result.push_str(&format!("<!DOCTYPE {}>\n", doctype));
        }
        
        result.push_str(&self.root.to_html());
        result
    }
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

/// DOM 节点
#[derive(Debug, Clone)]
pub struct Node {
    /// 节点类型
    pub node_type: NodeType,
    /// 子节点
    children: Vec<Node>,
}

impl Node {
    /// 创建元素节点
    pub fn element(tag_name: impl Into<String>) -> Self {
        Self {
            node_type: NodeType::Element(Element {
                tag_name: tag_name.into(),
                attributes: HashMap::new(),
                classes: Vec::new(),
                id: None,
            }),
            children: Vec::new(),
        }
    }

    /// 创建文本节点
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            node_type: NodeType::Text(content.into()),
            children: Vec::new(),
        }
    }

    /// 创建注释节点
    pub fn comment(content: impl Into<String>) -> Self {
        Self {
            node_type: NodeType::Comment(content.into()),
            children: Vec::new(),
        }
    }

    /// 添加子节点
    pub fn append_child(&mut self, child: Node) {
        self.children.push(child);
    }

    /// 获取子节点
    pub fn children(&self) -> &[Node] {
        &self.children
    }

    /// 获取子节点（可变）
    pub fn children_mut(&mut self) -> &mut Vec<Node> {
        &mut self.children
    }

    /// 查询选择器（简化版，只支持标签名）
    pub fn query_selector(&self, selector: &str) -> Option<Element> {
        for child in &self.children {
            if let NodeType::Element(ref el) = child.node_type {
                if el.tag_name == selector {
                    return Some(el.clone());
                }
            }
            if let Some(found) = child.query_selector(selector) {
                return Some(found);
            }
        }
        None
    }

    /// 查询所有匹配的元素
    pub fn query_selector_all(&self, selector: &str) -> Vec<Element> {
        let mut results = Vec::new();
        self.query_selector_all_recursive(selector, &mut results);
        results
    }

    fn query_selector_all_recursive(&self, selector: &str, results: &mut Vec<Element>) {
        for child in &self.children {
            if let NodeType::Element(ref el) = child.node_type {
                if el.tag_name == selector {
                    results.push(el.clone());
                }
            }
            child.query_selector_all_recursive(selector, results);
        }
    }

    /// 获取元素数量
    pub fn element_count(&self) -> usize {
        let mut count = 0;
        for child in &self.children {
            if let NodeType::Element(_) = child.node_type {
                count += 1;
            }
            count += child.element_count();
        }
        count
    }

    /// 获取文本节点数量
    pub fn text_node_count(&self) -> usize {
        let mut count = 0;
        for child in &self.children {
            if let NodeType::Text(_) = child.node_type {
                count += 1;
            }
            count += child.text_node_count();
        }
        count
    }

    /// 获取注释数量
    pub fn comment_count(&self) -> usize {
        let mut count = 0;
        for child in &self.children {
            if let NodeType::Comment(_) = child.node_type {
                count += 1;
            }
            count += child.comment_count();
        }
        count
    }

    /// 获取最大深度
    pub fn max_depth(&self) -> usize {
        let mut max_child_depth = 0;
        for child in &self.children {
            max_child_depth = max_child_depth.max(child.max_depth());
        }
        1 + max_child_depth
    }

    /// 序列化为 HTML
    pub fn to_html(&self) -> String {
        match &self.node_type {
            NodeType::Element(el) => {
                let mut result = format!("<{}", el.tag_name);
                
                // 添加 id
                if let Some(ref id) = el.id {
                    result.push_str(&format!(" id=\"{}\"", id));
                }
                
                // 添加 classes
                if !el.classes.is_empty() {
                    result.push_str(&format!(" class=\"{}\"", el.classes.join(" ")));
                }
                
                // 添加其他属性
                for (key, value) in &el.attributes {
                    if key != "id" && key != "class" {
                        result.push_str(&format!(" {}=\"{}\"", key, value));
                    }
                }
                
                if self.children.is_empty() {
                    // 自闭合标签
                    if is_void_element(&el.tag_name) {
                        result.push_str(" />");
                    } else {
                        result.push_str(&format!("></{}>", el.tag_name));
                    }
                } else {
                    result.push('>');
                    for child in &self.children {
                        result.push_str(&child.to_html());
                    }
                    result.push_str(&format!("</{}>", el.tag_name));
                }
                
                result
            }
            NodeType::Text(text) => html_escape(text),
            NodeType::Comment(comment) => format!("<!--{}-->", comment),
            NodeType::Document => {
                let mut result = String::new();
                for child in &self.children {
                    result.push_str(&child.to_html());
                }
                result
            }
        }
    }

    /// 转换为 Element（如果是元素节点）
    pub fn as_element(&self) -> Option<&Element> {
        match &self.node_type {
            NodeType::Element(el) => Some(el),
            _ => None,
        }
    }

    /// 转换为 Element（可变）
    pub fn as_element_mut(&mut self) -> Option<&mut Element> {
        match &mut self.node_type {
            NodeType::Element(el) => Some(el),
            _ => None,
        }
    }
}

/// 节点类型
#[derive(Debug, Clone)]
pub enum NodeType {
    /// 元素节点
    Element(Element),
    /// 文本节点
    Text(String),
    /// 注释节点
    Comment(String),
    /// 文档节点
    Document,
}

/// 元素
#[derive(Debug, Clone)]
pub struct Element {
    /// 标签名
    pub tag_name: String,
    /// 属性
    pub attributes: HashMap<String, String>,
    /// 类名列表
    pub classes: Vec<String>,
    /// ID
    pub id: Option<String>,
}

impl Element {
    /// 创建新元素
    pub fn new(tag_name: impl Into<String>) -> Self {
        Self {
            tag_name: tag_name.into(),
            attributes: HashMap::new(),
            classes: Vec::new(),
            id: None,
        }
    }

    /// 设置属性
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into();
        
        // 特殊处理 id 和 class
        if key == "id" {
            self.id = Some(value);
        } else if key == "class" {
            self.classes = value.split_whitespace().map(String::from).collect();
        } else {
            self.attributes.insert(key, value);
        }
    }

    /// 获取属性
    pub fn get_attribute(&self, key: &str) -> Option<String> {
        if key == "id" {
            self.id.clone()
        } else if key == "class" {
            Some(self.classes.join(" "))
        } else {
            self.attributes.get(key).cloned()
        }
    }

    /// 获取文本内容（递归）
    pub fn text_content(&self) -> Option<String> {
        // 这里简化处理，实际应该递归获取子节点的文本
        None
    }

    /// 转换为 Node
    pub fn as_node(&self) -> Node {
        Node {
            node_type: NodeType::Element(self.clone()),
            children: Vec::new(),
        }
    }

    /// 是否有某个类
    pub fn has_class(&self, class: &str) -> bool {
        self.classes.contains(&class.to_string())
    }

    /// 添加类
    pub fn add_class(&mut self, class: impl Into<String>) {
        let class = class.into();
        if !self.classes.contains(&class) {
            self.classes.push(class);
        }
    }

    /// 移除类
    pub fn remove_class(&mut self, class: &str) {
        self.classes.retain(|c| c != class);
    }
}

/// 检查是否是 void 元素（自闭合）
fn is_void_element(tag_name: &str) -> bool {
    matches!(
        tag_name,
        "area" | "base" | "br" | "col" | "embed" | "hr" | "img" | "input"
            | "link" | "meta" | "param" | "source" | "track" | "wbr"
    )
}

/// HTML 转义
fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_creation() {
        let mut el = Element::new("div");
        el.set_attribute("id", "test");
        el.set_attribute("class", "foo bar");
        
        assert_eq!(el.tag_name, "div");
        assert_eq!(el.id, Some("test".to_string()));
        assert_eq!(el.classes, vec!["foo", "bar"]);
    }

    #[test]
    fn test_node_html_serialization() {
        let mut node = Node::element("div");
        node.append_child(Node::text("Hello"));
        
        let html = node.to_html();
        assert_eq!(html, "<div>Hello</div>");
    }

    #[test]
    fn test_void_element() {
        let img = Node::element("img");
        let html = img.to_html();
        assert!(html.contains("/>"));
    }

    #[test]
    fn test_html_escape() {
        let text = Node::text("<script>alert('xss')</script>");
        let html = text.to_html();
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn test_document_html() {
        let doc = Document::new();
        let html = doc.to_html();
        assert!(html.starts_with("<!DOCTYPE html>"));
    }
}
