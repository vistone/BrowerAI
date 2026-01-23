// 布局引擎：计算各元素的几何信息

#[allow(unused_imports)]
use crate::dom::{ComputedStyle, Display, DomNode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutBox {
    pub node_tag: String,
    pub rect: Rect,
    pub children: Vec<LayoutBox>,
    pub box_type: BoxType,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BoxType {
    Block,
    Inline,
    Text,
}

pub struct LayoutEngine;

impl LayoutEngine {
    pub fn compute(
        dom: &DomNode,
        styles: &HashMap<String, ComputedStyle>,
    ) -> anyhow::Result<LayoutBox> {
        let mut layout_ctx = LayoutContext::new();
        Self::compute_recursive(dom, styles, &mut layout_ctx, 0.0, 0.0)
    }

    fn compute_recursive(
        node: &DomNode,
        styles: &HashMap<String, ComputedStyle>,
        ctx: &mut LayoutContext,
        x: f32,
        y: f32,
    ) -> anyhow::Result<LayoutBox> {
        match node {
            DomNode::Element { tag, children, .. } => {
                let style = styles.get(tag).cloned().unwrap_or_default();
                let (width, height) = (style.width.unwrap_or(100.0), style.height.unwrap_or(50.0));

                let mut layout_children = Vec::new();
                let mut child_y = y;

                for child in children {
                    let child_layout = Self::compute_recursive(child, styles, ctx, x, child_y)?;
                    child_y += child_layout.rect.height;
                    layout_children.push(child_layout);
                }

                Ok(LayoutBox {
                    node_tag: tag.clone(),
                    rect: Rect {
                        x,
                        y,
                        width,
                        height,
                    },
                    children: layout_children,
                    box_type: BoxType::Block,
                })
            }
            DomNode::Text(content) => {
                let text_height = 16.0; // 估算
                Ok(LayoutBox {
                    node_tag: "text".to_string(),
                    rect: Rect {
                        x,
                        y,
                        width: content.len() as f32 * 8.0, // 粗估
                        height: text_height,
                    },
                    children: Vec::new(),
                    box_type: BoxType::Text,
                })
            }
            DomNode::Document { children } => {
                let mut layout_children = Vec::new();
                let mut child_y = y;

                for child in children {
                    let child_layout = Self::compute_recursive(child, styles, ctx, x, child_y)?;
                    child_y += child_layout.rect.height;
                    layout_children.push(child_layout);
                }

                Ok(LayoutBox {
                    node_tag: "document".to_string(),
                    rect: Rect {
                        x,
                        y,
                        width: 800.0,
                        height: child_y,
                    },
                    children: layout_children,
                    box_type: BoxType::Block,
                })
            }
            DomNode::Comment(_) => Ok(LayoutBox {
                node_tag: "comment".to_string(),
                rect: Rect::default(),
                children: Vec::new(),
                box_type: BoxType::Block,
            }),
        }
    }
}

#[allow(dead_code)]
struct LayoutContext {
    viewport_width: f32,
}

impl LayoutContext {
    fn new() -> Self {
        Self {
            viewport_width: 800.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_simple_block() {
        let dom = DomNode::Element {
            tag: "div".to_string(),
            attributes: Default::default(),
            children: vec![],
        };

        let mut styles = HashMap::new();
        styles.insert(
            "div".to_string(),
            ComputedStyle {
                display: Display::Block,
                width: Some(200.0),
                height: Some(100.0),
                ..Default::default()
            },
        );

        let layout = LayoutEngine::compute(&dom, &styles).unwrap();
        assert_eq!(layout.rect.width, 200.0);
        assert_eq!(layout.rect.height, 100.0);
    }
}
