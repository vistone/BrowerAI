// DOM 数据模型（从 HTML parser 接收）

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomNode {
    Document {
        children: Vec<DomNode>,
    },
    Element {
        tag: String,
        attributes: std::collections::HashMap<String, String>,
        children: Vec<DomNode>,
    },
    Text(String),
    Comment(String),
}

impl DomNode {
    pub fn tag_name(&self) -> Option<&str> {
        match self {
            DomNode::Element { tag, .. } => Some(tag),
            _ => None,
        }
    }

    pub fn text_content(&self) -> String {
        match self {
            DomNode::Text(s) => s.clone(),
            DomNode::Element { children, .. } => children
                .iter()
                .map(|c| c.text_content())
                .collect::<Vec<_>>()
                .join(""),
            DomNode::Document { children } => children
                .iter()
                .map(|c| c.text_content())
                .collect::<Vec<_>>()
                .join(""),
            DomNode::Comment(_) => String::new(),
        }
    }
}

/// 样式数据结构（从 CSS 匹配）
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Display {
    Block,
    Inline,
    InlineBlock,
    None,
}

impl Default for Display {
    fn default() -> Self {
        Display::Block
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComputedStyle {
    pub display: Display,
    pub width: Option<f32>,
    pub height: Option<f32>,
    pub margin_top: Option<f32>,
    pub margin_bottom: Option<f32>,
    pub margin_left: Option<f32>,
    pub margin_right: Option<f32>,
    pub padding_top: Option<f32>,
    pub padding_bottom: Option<f32>,
    pub padding_left: Option<f32>,
    pub padding_right: Option<f32>,
    pub color: Option<String>,
    pub background_color: Option<String>,
    pub font_size: Option<f32>,
}
