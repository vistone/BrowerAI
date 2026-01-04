// DOM API implementation for JavaScript execution
// Provides a Document Object Model interface for programmatic page manipulation

use markup5ever_rcdom::{Handle, NodeData, RcDom};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

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
        
        // Build DOM tree from RcDom
        let dom_node = Self::convert_handle_to_dom(&rcdom.document);
        doc.root = Arc::new(RwLock::new(dom_node));
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
            NodeData::Document => DomNode::Document,
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
}
