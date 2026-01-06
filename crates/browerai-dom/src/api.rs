/// Enhanced DOM API for JavaScript execution
///
/// Provides JavaScript-compatible DOM manipulation methods
use super::{Document, DomNode};
use std::sync::{Arc, RwLock};

/// JavaScript-compatible DOM API extensions
pub trait DomApiExtensions {
    /// Get element by ID (JavaScript-style)
    fn get_element_by_id_js(&self, id: &str) -> Option<ElementHandle>;

    /// Get elements by class name
    fn get_elements_by_class_name(&self, class_name: &str) -> Vec<ElementHandle>;

    /// Get elements by tag name (JavaScript-style)
    fn get_elements_by_tag_name_js(&self, tag_name: &str) -> Vec<ElementHandle>;
}

/// Handle to a DOM element for safe JavaScript interaction
#[derive(Debug, Clone)]
pub struct ElementHandle {
    pub(crate) node: Arc<RwLock<DomNode>>,
}

impl ElementHandle {
    pub fn new(node: Arc<RwLock<DomNode>>) -> Self {
        Self { node }
    }

    /// Get the tag name
    pub fn tag_name(&self) -> Option<String> {
        let node_read = self.node.read().ok()?;
        if let DomNode::Element(elem) = &*node_read {
            Some(elem.tag_name.clone())
        } else {
            None
        }
    }

    /// Get an attribute value
    pub fn get_attribute(&self, name: &str) -> Option<String> {
        let node_read = self.node.read().ok()?;
        if let DomNode::Element(elem) = &*node_read {
            elem.get_attribute(name).cloned()
        } else {
            None
        }
    }

    /// Set an attribute value
    pub fn set_attribute(&self, name: &str, value: &str) -> Result<(), String> {
        let mut node_write = self.node.write().map_err(|e| e.to_string())?;
        if let DomNode::Element(ref mut elem) = *node_write {
            elem.set_attribute(name.to_string(), value.to_string());
            Ok(())
        } else {
            Err("Not an element node".to_string())
        }
    }

    /// Remove an attribute
    pub fn remove_attribute(&self, name: &str) -> Result<Option<String>, String> {
        let mut node_write = self.node.write().map_err(|e| e.to_string())?;
        if let DomNode::Element(ref mut elem) = *node_write {
            Ok(elem.remove_attribute(name))
        } else {
            Err("Not an element node".to_string())
        }
    }

    /// Check if element has an attribute
    pub fn has_attribute(&self, name: &str) -> bool {
        self.get_attribute(name).is_some()
    }

    /// Get inner HTML (simplified - returns text content)
    pub fn inner_text(&self) -> Option<String> {
        let node_read = self.node.read().ok()?;
        match &*node_read {
            DomNode::Element(elem) => Some(elem.get_text_content()),
            DomNode::Text(text) => Some(text.clone()),
            _ => None,
        }
    }

    /// Set inner text
    pub fn set_inner_text(&self, text: &str) -> Result<(), String> {
        let mut node_write = self.node.write().map_err(|e| e.to_string())?;
        if let DomNode::Element(ref mut elem) = *node_write {
            elem.text_content = Some(text.to_string());
            elem.children.clear();
            Ok(())
        } else {
            Err("Not an element node".to_string())
        }
    }

    /// Get class list
    pub fn get_class_list(&self) -> Vec<String> {
        if let Some(classes) = self.get_attribute("class") {
            classes.split_whitespace().map(|s| s.to_string()).collect()
        } else {
            Vec::new()
        }
    }

    /// Add a class
    pub fn add_class(&self, class_name: &str) -> Result<(), String> {
        let mut classes = self.get_class_list();
        if !classes.contains(&class_name.to_string()) {
            classes.push(class_name.to_string());
            self.set_attribute("class", &classes.join(" "))
        } else {
            Ok(())
        }
    }

    /// Remove a class
    pub fn remove_class(&self, class_name: &str) -> Result<(), String> {
        let mut classes = self.get_class_list();
        classes.retain(|c| c != class_name);
        self.set_attribute("class", &classes.join(" "))
    }

    /// Toggle a class
    pub fn toggle_class(&self, class_name: &str) -> Result<bool, String> {
        let classes = self.get_class_list();
        if classes.contains(&class_name.to_string()) {
            self.remove_class(class_name)?;
            Ok(false)
        } else {
            self.add_class(class_name)?;
            Ok(true)
        }
    }

    /// Check if element has a class
    pub fn has_class(&self, class_name: &str) -> bool {
        self.get_class_list().contains(&class_name.to_string())
    }

    /// Get all children
    pub fn children(&self) -> Vec<ElementHandle> {
        let node_read = match self.node.read() {
            Ok(guard) => guard,
            Err(_) => return Vec::new(),
        };

        if let DomNode::Element(elem) = &*node_read {
            elem.children
                .iter()
                .map(|c| ElementHandle::new(c.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Append a child element
    pub fn append_child(&self, child: &ElementHandle) -> Result<(), String> {
        let mut node_write = self.node.write().map_err(|e| e.to_string())?;
        if let DomNode::Element(ref mut elem) = *node_write {
            elem.append_child(child.node.clone());
            Ok(())
        } else {
            Err("Not an element node".to_string())
        }
    }

    /// Remove a child element
    pub fn remove_child(&self, child: &ElementHandle) -> Result<bool, String> {
        let mut node_write = self.node.write().map_err(|e| e.to_string())?;
        if let DomNode::Element(ref mut elem) = *node_write {
            Ok(elem.remove_child(&child.node))
        } else {
            Err("Not an element node".to_string())
        }
    }

    /// Get the ID of the element
    pub fn id(&self) -> Option<String> {
        self.get_attribute("id")
    }

    /// Set the ID of the element
    pub fn set_id(&self, id: &str) -> Result<(), String> {
        self.set_attribute("id", id)
    }
}

impl DomApiExtensions for Document {
    fn get_element_by_id_js(&self, id: &str) -> Option<ElementHandle> {
        self.get_element_by_id(id).map(ElementHandle::new)
    }

    fn get_elements_by_class_name(&self, class_name: &str) -> Vec<ElementHandle> {
        self.query_selector_all(&format!(".{}", class_name))
            .into_iter()
            .map(ElementHandle::new)
            .collect()
    }

    fn get_elements_by_tag_name_js(&self, tag_name: &str) -> Vec<ElementHandle> {
        self.get_elements_by_tag_name(tag_name)
            .into_iter()
            .map(ElementHandle::new)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use browerai_html_parser::HtmlParser;

    #[test]
    fn test_element_handle_tag_name() {
        let doc = Document::new();
        let elem = doc.create_element("div");
        let handle = ElementHandle::new(elem);

        assert_eq!(handle.tag_name(), Some("div".to_string()));
    }

    #[test]
    fn test_element_handle_attributes() {
        let doc = Document::new();
        let elem = doc.create_element("div");
        let handle = ElementHandle::new(elem);

        handle.set_attribute("id", "test").unwrap();
        assert_eq!(handle.get_attribute("id"), Some("test".to_string()));
        assert!(handle.has_attribute("id"));

        handle.remove_attribute("id").unwrap();
        assert!(!handle.has_attribute("id"));
    }

    #[test]
    fn test_element_handle_classes() {
        let doc = Document::new();
        let elem = doc.create_element("div");
        let handle = ElementHandle::new(elem);

        handle.add_class("active").unwrap();
        assert!(handle.has_class("active"));

        handle.add_class("visible").unwrap();
        let classes = handle.get_class_list();
        assert_eq!(classes.len(), 2);

        handle.remove_class("active").unwrap();
        assert!(!handle.has_class("active"));
        assert!(handle.has_class("visible"));
    }

    #[test]
    fn test_element_handle_toggle_class() {
        let doc = Document::new();
        let elem = doc.create_element("div");
        let handle = ElementHandle::new(elem);

        let added = handle.toggle_class("active").unwrap();
        assert!(added);
        assert!(handle.has_class("active"));

        let removed = handle.toggle_class("active").unwrap();
        assert!(!removed);
        assert!(!handle.has_class("active"));
    }

    #[test]
    fn test_element_handle_inner_text() {
        let doc = Document::new();
        let elem = doc.create_element("div");
        let handle = ElementHandle::new(elem);

        handle.set_inner_text("Hello, World!").unwrap();
        assert_eq!(handle.inner_text(), Some("Hello, World!".to_string()));
    }

    #[test]
    fn test_element_handle_children() {
        let doc = Document::new();
        let parent = doc.create_element("div");
        let child = doc.create_element("span");

        let parent_handle = ElementHandle::new(parent);
        let child_handle = ElementHandle::new(child);

        parent_handle.append_child(&child_handle).unwrap();

        let children = parent_handle.children();
        assert_eq!(children.len(), 1);
    }

    #[test]
    fn test_element_handle_id() {
        let doc = Document::new();
        let elem = doc.create_element("div");
        let handle = ElementHandle::new(elem);

        handle.set_id("main").unwrap();
        assert_eq!(handle.id(), Some("main".to_string()));
    }

    #[test]
    fn test_dom_api_extensions() {
        let html = r#"<html><body><div id="main" class="container">Content</div></body></html>"#;
        let parser = HtmlParser::new();
        let rcdom = parser.parse(html).unwrap();
        let doc = Document::from_rcdom(&rcdom);

        // Test get_element_by_id_js
        let elem = doc.get_element_by_id_js("main");
        assert!(elem.is_some());

        // Test get_elements_by_class_name
        let elems = doc.get_elements_by_class_name("container");
        assert_eq!(elems.len(), 1);

        // Test get_elements_by_tag_name_js
        let divs = doc.get_elements_by_tag_name_js("div");
        assert!(divs.len() >= 1);
    }

    #[test]
    fn test_element_handle_remove_child() {
        let doc = Document::new();
        let parent = doc.create_element("div");
        let child = doc.create_element("span");

        let parent_handle = ElementHandle::new(parent);
        let child_handle = ElementHandle::new(child);

        parent_handle.append_child(&child_handle).unwrap();
        assert_eq!(parent_handle.children().len(), 1);

        let removed = parent_handle.remove_child(&child_handle).unwrap();
        assert!(removed);
        assert_eq!(parent_handle.children().len(), 0);
    }
}
