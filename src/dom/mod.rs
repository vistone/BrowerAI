// DOM API implementation for JavaScript execution
// Provides a Document Object Model interface for programmatic page manipulation

pub mod events;

use markup5ever_rcdom::{Handle, NodeData, RcDom};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub use events::{Event, EventListener, EventListeners, EventPhase, EventType};

/// Represents a DOM element with attributes and children
#[derive(Debug, Clone)]
pub struct DomElement {
    pub tag_name: String,
    pub attributes: HashMap<String, String>,
    pub children: Vec<Arc<RwLock<DomNode>>>,
    pub text_content: Option<String>,
}

/// Represents different types of DOM nodes
#[derive(Debug, Clone)]
pub enum DomNode {
    Element(DomElement),
    Text(String),
    Comment(String),
    Document,
}

/// The main Document interface
#[derive(Debug, Clone)]
pub struct Document {
    root: Arc<RwLock<DomNode>>,
    elements_by_id: HashMap<String, Arc<RwLock<DomNode>>>,
    elements_by_tag: HashMap<String, Vec<Arc<RwLock<DomNode>>>>,
}

impl Document {
    /// Create a new empty document
    pub fn new() -> Self {
        Document {
            root: Arc::new(RwLock::new(DomNode::Document)),
            elements_by_id: HashMap::new(),
            elements_by_tag: HashMap::new(),
        }
    }

    /// Create a document from an HTML5ever RcDom
    pub fn from_rcdom(rcdom: &RcDom) -> Self {
        let mut doc = Document::new();
        
        // Convert the document and its children
        // html5ever's document node has children that are the actual HTML elements
        let dom_node = Self::convert_handle_to_dom(&rcdom.document);
        doc.root = Arc::new(RwLock::new(dom_node));
        
        // Index all elements in the tree
        doc.index_elements();
        
        doc
    }

    /// Convert html5ever Handle to DOM node
    fn convert_handle_to_dom(handle: &Handle) -> DomNode {
        match &handle.data {
            NodeData::Element { name, attrs, .. } => {
                let tag_name = name.local.to_string();
                let mut attributes = HashMap::new();
                
                for attr in attrs.borrow().iter() {
                    attributes.insert(
                        attr.name.local.to_string(),
                        attr.value.to_string(),
                    );
                }

                let children: Vec<Arc<RwLock<DomNode>>> = handle
                    .children
                    .borrow()
                    .iter()
                    .map(|child| Arc::new(RwLock::new(Self::convert_handle_to_dom(child))))
                    .collect();

                DomNode::Element(DomElement {
                    tag_name,
                    attributes,
                    children,
                    text_content: None,
                })
            }
            NodeData::Text { contents } => {
                DomNode::Text(contents.borrow().to_string())
            }
            NodeData::Comment { contents } => {
                DomNode::Comment(contents.to_string())
            }
            NodeData::Document => {
                // For Document nodes, wrap children in a synthetic element
                let children: Vec<Arc<RwLock<DomNode>>> = handle
                    .children
                    .borrow()
                    .iter()
                    .map(|child| Arc::new(RwLock::new(Self::convert_handle_to_dom(child))))
                    .collect();
                
                // Return a document-like element node
                DomNode::Element(DomElement {
                    tag_name: "#document".to_string(),
                    attributes: HashMap::new(),
                    children,
                    text_content: None,
                })
            }
            _ => DomNode::Document, // Other node types become Document
        }
    }

    /// Index elements by ID and tag name for quick lookups
    fn index_elements(&mut self) {
        self.elements_by_id.clear();
        self.elements_by_tag.clear();
        self.index_node_recursive(&self.root.clone());
    }

    /// Recursively index a node and its children
    fn index_node_recursive(&mut self, node: &Arc<RwLock<DomNode>>) {
        let node_read = node.read().unwrap();
        
        if let DomNode::Element(element) = &*node_read {
            // Index by ID
            if let Some(id) = element.attributes.get("id") {
                self.elements_by_id.insert(id.clone(), node.clone());
            }

            // Index by tag name
            let tag_lower = element.tag_name.to_lowercase();
            self.elements_by_tag
                .entry(tag_lower)
                .or_insert_with(Vec::new)
                .push(node.clone());

            // Recurse into children
            let children = element.children.clone();
            drop(node_read); // Release read lock before recursing
            for child in &children {
                self.index_node_recursive(child);
            }
        }
    }

    /// Get element by ID
    pub fn get_element_by_id(&self, id: &str) -> Option<Arc<RwLock<DomNode>>> {
        self.elements_by_id.get(id).cloned()
    }

    /// Get elements by tag name
    pub fn get_elements_by_tag_name(&self, tag_name: &str) -> Vec<Arc<RwLock<DomNode>>> {
        let tag_lower = tag_name.to_lowercase();
        self.elements_by_tag.get(&tag_lower).cloned().unwrap_or_default()
    }

    /// Get the document root
    pub fn get_root(&self) -> Arc<RwLock<DomNode>> {
        self.root.clone()
    }

    /// Create a new element
    pub fn create_element(&self, tag_name: &str) -> Arc<RwLock<DomNode>> {
        Arc::new(RwLock::new(DomNode::Element(DomElement {
            tag_name: tag_name.to_string(),
            attributes: HashMap::new(),
            children: Vec::new(),
            text_content: None,
        })))
    }

    /// Create a text node
    pub fn create_text_node(&self, text: &str) -> Arc<RwLock<DomNode>> {
        Arc::new(RwLock::new(DomNode::Text(text.to_string())))
    }

    /// Query selector - find the first element matching the CSS selector
    pub fn query_selector(&self, selector: &str) -> Option<Arc<RwLock<DomNode>>> {
        self.query_selector_all(selector).into_iter().next()
    }

    /// Query selector all - find all elements matching the CSS selector
    pub fn query_selector_all(&self, selector: &str) -> Vec<Arc<RwLock<DomNode>>> {
        // Simple selector matching - supports basic selectors like tag, .class, #id, [attr]
        let mut results = Vec::new();
        
        // Parse the selector
        if selector.starts_with('#') {
            // ID selector
            let id = &selector[1..];
            if let Some(elem) = self.get_element_by_id(id) {
                results.push(elem);
            }
        } else if selector.starts_with('.') {
            // Class selector
            let class_name = &selector[1..];
            self.query_by_class_recursive(&self.root, class_name, &mut results);
        } else if selector.contains('[') && selector.contains(']') {
            // Attribute selector [attr] or [attr=value]
            self.query_by_attribute_recursive(&self.root, selector, &mut results);
        } else {
            // Tag selector
            results = self.get_elements_by_tag_name(selector);
        }
        
        results
    }

    /// Recursively search for elements with a specific class
    fn query_by_class_recursive(
        &self,
        node: &Arc<RwLock<DomNode>>,
        class_name: &str,
        results: &mut Vec<Arc<RwLock<DomNode>>>,
    ) {
        let node_read = node.read().unwrap();
        
        if let DomNode::Element(element) = &*node_read {
            // Check if element has the class
            if let Some(classes) = element.attributes.get("class") {
                if classes.split_whitespace().any(|c| c == class_name) {
                    results.push(node.clone());
                }
            }
            
            // Recurse into children
            let children = element.children.clone();
            drop(node_read);
            for child in &children {
                self.query_by_class_recursive(child, class_name, results);
            }
        }
    }

    /// Recursively search for elements with a specific attribute
    fn query_by_attribute_recursive(
        &self,
        node: &Arc<RwLock<DomNode>>,
        selector: &str,
        results: &mut Vec<Arc<RwLock<DomNode>>>,
    ) {
        let node_read = node.read().unwrap();
        
        if let DomNode::Element(element) = &*node_read {
            // Parse attribute selector: [attr] or [attr=value]
            let attr_part = selector.trim_matches(|c| c == '[' || c == ']');
            
            let matches = if attr_part.contains('=') {
                // [attr=value]
                let parts: Vec<&str> = attr_part.splitn(2, '=').collect();
                if parts.len() == 2 {
                    let attr_name = parts[0].trim();
                    let attr_value = parts[1].trim().trim_matches('"').trim_matches('\'');
                    element.attributes.get(attr_name).map(|v| v == attr_value).unwrap_or(false)
                } else {
                    false
                }
            } else {
                // [attr]
                element.attributes.contains_key(attr_part.trim())
            };
            
            if matches {
                results.push(node.clone());
            }
            
            // Recurse into children
            let children = element.children.clone();
            drop(node_read);
            for child in &children {
                self.query_by_attribute_recursive(child, selector, results);
            }
        }
    }
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

impl DomElement {
    /// Get an attribute value
    pub fn get_attribute(&self, name: &str) -> Option<&String> {
        self.attributes.get(name)
    }

    /// Set an attribute value
    pub fn set_attribute(&mut self, name: String, value: String) {
        self.attributes.insert(name, value);
    }

    /// Remove an attribute
    pub fn remove_attribute(&mut self, name: &str) -> Option<String> {
        self.attributes.remove(name)
    }

    /// Append a child node
    pub fn append_child(&mut self, child: Arc<RwLock<DomNode>>) {
        self.children.push(child);
    }

    /// Remove a child node
    pub fn remove_child(&mut self, child: &Arc<RwLock<DomNode>>) -> bool {
        if let Some(pos) = self.children.iter().position(|c| Arc::ptr_eq(c, child)) {
            self.children.remove(pos);
            true
        } else {
            false
        }
    }

    /// Get text content of element and its descendants
    pub fn get_text_content(&self) -> String {
        let mut text = String::new();
        
        if let Some(content) = &self.text_content {
            text.push_str(content);
        }

        for child in &self.children {
            let child_read = child.read().unwrap();
            match &*child_read {
                DomNode::Text(content) => text.push_str(content),
                DomNode::Element(elem) => text.push_str(&elem.get_text_content()),
                _ => {}
            }
        }

        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::html::HtmlParser;

    #[test]
    fn test_create_document() {
        let doc = Document::new();
        let root = doc.get_root();
        let root_read = root.read().unwrap();
        matches!(*root_read, DomNode::Document);
    }

    #[test]
    fn test_create_element() {
        let doc = Document::new();
        let element = doc.create_element("div");
        let elem_read = element.read().unwrap();
        
        if let DomNode::Element(elem) = &*elem_read {
            assert_eq!(elem.tag_name, "div");
        } else {
            panic!("Expected Element node");
        }
    }

    #[test]
    fn test_create_text_node() {
        let doc = Document::new();
        let text_node = doc.create_text_node("Hello, World!");
        let text_read = text_node.read().unwrap();
        
        if let DomNode::Text(text) = &*text_read {
            assert_eq!(text, "Hello, World!");
        } else {
            panic!("Expected Text node");
        }
    }

    #[test]
    fn test_element_attributes() {
        let mut element = DomElement {
            tag_name: "div".to_string(),
            attributes: HashMap::new(),
            children: Vec::new(),
            text_content: None,
        };

        element.set_attribute("id".to_string(), "test-id".to_string());
        assert_eq!(element.get_attribute("id"), Some(&"test-id".to_string()));

        element.remove_attribute("id");
        assert_eq!(element.get_attribute("id"), None);
    }

    #[test]
    fn test_from_rcdom() {
        let html = r#"<html><body><div id="test">Hello</div></body></html>"#;
        let parser = HtmlParser::new();
        let rcdom = parser.parse(html).unwrap();
        
        let doc = Document::from_rcdom(&rcdom);
        
        // Verify we have a DOM structure
        let root = doc.get_root();
        let root_read = root.read().unwrap();
        matches!(*root_read, DomNode::Document | DomNode::Element(_));
    }

    #[test]
    fn test_get_text_content() {
        let element = DomElement {
            tag_name: "div".to_string(),
            attributes: HashMap::new(),
            children: vec![
                Arc::new(RwLock::new(DomNode::Text("Hello ".to_string()))),
                Arc::new(RwLock::new(DomNode::Text("World!".to_string()))),
            ],
            text_content: None,
        };

        assert_eq!(element.get_text_content(), "Hello World!");
    }

    #[test]
    fn test_element_manipulation() {
        let doc = Document::new();
        let parent = doc.create_element("div");
        let child = doc.create_text_node("Test");
        
        {
            let mut parent_write = parent.write().unwrap();
            if let DomNode::Element(ref mut elem) = *parent_write {
                elem.append_child(child.clone());
                assert_eq!(elem.children.len(), 1);
            }
        }
        
        // Verify text content
        let parent_read = parent.read().unwrap();
        if let DomNode::Element(ref elem) = *parent_read {
            assert_eq!(elem.get_text_content(), "Test");
        }
    }

    #[test]
    fn test_query_selector_by_id() {
        let html = r#"<html><body><div id="main">Content</div><div id="sidebar">Side</div></body></html>"#;
        let parser = HtmlParser::new();
        let rcdom = parser.parse(html).unwrap();
        let doc = Document::from_rcdom(&rcdom);
        
        // Query by ID
        let result = doc.query_selector("#main");
        assert!(result.is_some());
        
        if let Some(node) = result {
            let node_read = node.read().unwrap();
            if let DomNode::Element(elem) = &*node_read {
                assert_eq!(elem.tag_name, "div");
                assert_eq!(elem.get_attribute("id"), Some(&"main".to_string()));
            }
        }
    }

    #[test]
    fn test_query_selector_by_class() {
        let html = r#"<html><body><div class="active">One</div><div class="active">Two</div><div>Three</div></body></html>"#;
        let parser = HtmlParser::new();
        let rcdom = parser.parse(html).unwrap();
        let doc = Document::from_rcdom(&rcdom);
        
        // Query by class
        let results = doc.query_selector_all(".active");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_selector_by_tag() {
        let html = r#"<html><body><div>One</div><div>Two</div><span>Three</span></body></html>"#;
        let parser = HtmlParser::new();
        let rcdom = parser.parse(html).unwrap();
        let doc = Document::from_rcdom(&rcdom);
        
        // Query by tag name
        let results = doc.query_selector_all("div");
        assert_eq!(results.len(), 2);
        
        let span_results = doc.query_selector_all("span");
        assert_eq!(span_results.len(), 1);
    }

    #[test]
    fn test_query_selector_by_attribute() {
        let html = r#"<html><body><div data-test="value">One</div><div>Two</div><div data-test="other">Three</div></body></html>"#;
        let parser = HtmlParser::new();
        let rcdom = parser.parse(html).unwrap();
        let doc = Document::from_rcdom(&rcdom);
        
        // Query by attribute presence
        let results = doc.query_selector_all("[data-test]");
        assert_eq!(results.len(), 2);
        
        // Query by attribute value
        let value_results = doc.query_selector_all("[data-test=value]");
        assert_eq!(value_results.len(), 1);
    }

    #[test]
    fn test_query_selector_single() {
        let html = r#"<html><body><div class="item">One</div><div class="item">Two</div></body></html>"#;
        let parser = HtmlParser::new();
        let rcdom = parser.parse(html).unwrap();
        let doc = Document::from_rcdom(&rcdom);
        
        // querySelector returns first match
        let result = doc.query_selector(".item");
        assert!(result.is_some());
        
        // querySelectorAll returns all matches
        let all_results = doc.query_selector_all(".item");
        assert_eq!(all_results.len(), 2);
    }
}
