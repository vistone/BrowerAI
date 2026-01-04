use anyhow::Result;
use markup5ever_rcdom::{Handle, NodeData, RcDom};
use std::collections::HashMap;

/// Represents a box in the CSS box model
#[derive(Debug, Clone)]
pub struct LayoutBox {
    pub box_type: BoxType,
    pub dimensions: Dimensions,
    pub children: Vec<LayoutBox>,
    pub element_type: String,
}

/// Type of layout box
#[derive(Debug, Clone, PartialEq)]
pub enum BoxType {
    Block,
    Inline,
    InlineBlock,
    Flex,
    Grid,
    Anonymous,
}

/// CSS box model dimensions
#[derive(Debug, Clone, Default)]
pub struct Dimensions {
    pub content: Rect,
    pub padding: EdgeSizes,
    pub border: EdgeSizes,
    pub margin: EdgeSizes,
}

/// Rectangle with position and size
#[derive(Debug, Clone, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Edge sizes (top, right, bottom, left)
#[derive(Debug, Clone, Default)]
pub struct EdgeSizes {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn expanded_by(&self, edge: &EdgeSizes) -> Self {
        Self {
            x: self.x - edge.left,
            y: self.y - edge.top,
            width: self.width + edge.left + edge.right,
            height: self.height + edge.top + edge.bottom,
        }
    }
}

impl Dimensions {
    /// Get the total width including padding and border
    pub fn padding_box(&self) -> Rect {
        self.content.expanded_by(&self.padding)
    }

    /// Get the total width including padding, border, and margin
    pub fn border_box(&self) -> Rect {
        self.padding_box().expanded_by(&self.border)
    }

    /// Get the total area including margin
    pub fn margin_box(&self) -> Rect {
        self.border_box().expanded_by(&self.margin)
    }
}

/// Layout engine for CSS box model
pub struct LayoutEngine {
    viewport_width: f32,
    viewport_height: f32,
}

impl LayoutEngine {
    /// Create a new layout engine with viewport dimensions
    pub fn new(viewport_width: f32, viewport_height: f32) -> Self {
        Self {
            viewport_width,
            viewport_height,
        }
    }

    /// Build layout tree from DOM and styles
    pub fn build_layout_tree(
        &self,
        dom: &RcDom,
        _styles: &HashMap<String, HashMap<String, String>>,
    ) -> Result<LayoutBox> {
        log::info!("Building layout tree");

        // Start with root element
        let root_box = self.build_layout_box(&dom.document, BoxType::Block)?;

        log::info!(
            "Layout tree built with {} children",
            root_box.children.len()
        );
        Ok(root_box)
    }

    /// Build a layout box for a DOM node
    fn build_layout_box(&self, handle: &Handle, box_type: BoxType) -> Result<LayoutBox> {
        let element_type = match &handle.data {
            NodeData::Element { name, .. } => name.local.to_string(),
            NodeData::Document => "document".to_string(),
            NodeData::Text { .. } => "text".to_string(),
            _ => "unknown".to_string(),
        };

        let mut layout_box = LayoutBox {
            box_type,
            dimensions: Dimensions::default(),
            children: Vec::new(),
            element_type,
        };

        // Recursively build children
        for child in handle.children.borrow().iter() {
            let child_type = self.determine_box_type(child);
            let child_box = self.build_layout_box(child, child_type)?;
            layout_box.children.push(child_box);
        }

        Ok(layout_box)
    }

    /// Determine the box type for a node
    fn determine_box_type(&self, handle: &Handle) -> BoxType {
        match &handle.data {
            NodeData::Element { name, .. } => {
                // Determine based on element type
                match name.local.as_ref() {
                    "div" | "p" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "header" | "footer"
                    | "section" | "article" | "nav" => BoxType::Block,
                    "span" | "a" | "strong" | "em" | "b" | "i" => BoxType::Inline,
                    _ => BoxType::Block,
                }
            }
            NodeData::Text { .. } => BoxType::Inline,
            _ => BoxType::Anonymous,
        }
    }

    /// Calculate layout for the box tree
    pub fn calculate_layout(&self, layout_box: &mut LayoutBox, containing_block: Rect) {
        log::debug!(
            "Calculating layout for {} in containing block {:?}",
            layout_box.element_type,
            containing_block
        );

        match layout_box.box_type {
            BoxType::Block => self.layout_block(layout_box, containing_block),
            BoxType::Inline => self.layout_inline(layout_box, containing_block),
            BoxType::Flex => self.layout_flex(layout_box, containing_block),
            BoxType::Grid => self.layout_grid(layout_box, containing_block),
            _ => self.layout_block(layout_box, containing_block),
        }
    }

    /// Layout a block box
    fn layout_block(&self, layout_box: &mut LayoutBox, containing_block: Rect) {
        // Set width to containing block width (default block behavior)
        layout_box.dimensions.content.width = containing_block.width;
        layout_box.dimensions.content.x = containing_block.x;
        layout_box.dimensions.content.y = containing_block.y;

        let mut y_offset = 0.0;

        // Layout children vertically
        for child in &mut layout_box.children {
            let child_containing_block = Rect {
                x: layout_box.dimensions.content.x,
                y: layout_box.dimensions.content.y + y_offset,
                width: layout_box.dimensions.content.width,
                height: 0.0, // Will be calculated
            };

            self.calculate_layout(child, child_containing_block);

            // Update height based on children
            y_offset += child.dimensions.margin_box().height;
        }

        layout_box.dimensions.content.height = y_offset;
    }

    /// Layout an inline box
    fn layout_inline(&self, layout_box: &mut LayoutBox, containing_block: Rect) {
        // Inline boxes flow horizontally
        layout_box.dimensions.content.x = containing_block.x;
        layout_box.dimensions.content.y = containing_block.y;

        // Default inline height
        layout_box.dimensions.content.height = 16.0; // Approximate line height
        layout_box.dimensions.content.width = 100.0; // Will be calculated based on content
    }

    /// Layout a flexbox container
    fn layout_flex(&self, layout_box: &mut LayoutBox, containing_block: Rect) {
        log::debug!("Laying out flexbox");

        layout_box.dimensions.content.width = containing_block.width;
        layout_box.dimensions.content.x = containing_block.x;
        layout_box.dimensions.content.y = containing_block.y;

        // Simple horizontal flex layout
        let mut x_offset = 0.0;
        let child_count = layout_box.children.len() as f32;
        let child_width = if child_count > 0.0 {
            containing_block.width / child_count
        } else {
            0.0
        };

        for child in &mut layout_box.children {
            let child_containing_block = Rect {
                x: layout_box.dimensions.content.x + x_offset,
                y: layout_box.dimensions.content.y,
                width: child_width,
                height: containing_block.height,
            };

            self.calculate_layout(child, child_containing_block);
            x_offset += child_width;
        }

        layout_box.dimensions.content.height = containing_block.height;
    }

    /// Layout a grid container
    fn layout_grid(&self, layout_box: &mut LayoutBox, containing_block: Rect) {
        log::debug!("Laying out grid");

        layout_box.dimensions.content.width = containing_block.width;
        layout_box.dimensions.content.x = containing_block.x;
        layout_box.dimensions.content.y = containing_block.y;

        // Simple 2-column grid layout
        let columns = 2;
        let column_width = containing_block.width / columns as f32;
        let mut row = 0;
        let mut col = 0;
        let row_height = 100.0; // Fixed row height for simplicity

        for child in &mut layout_box.children {
            let child_containing_block = Rect {
                x: layout_box.dimensions.content.x + (col as f32 * column_width),
                y: layout_box.dimensions.content.y + (row as f32 * row_height),
                width: column_width,
                height: row_height,
            };

            self.calculate_layout(child, child_containing_block);

            col += 1;
            if col >= columns {
                col = 0;
                row += 1;
            }
        }

        layout_box.dimensions.content.height = ((row + 1) as f32) * row_height;
    }

    /// Get viewport dimensions
    pub fn viewport(&self) -> Rect {
        Rect::new(0.0, 0.0, self.viewport_width, self.viewport_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_creation() {
        let rect = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(rect.x, 10.0);
        assert_eq!(rect.y, 20.0);
        assert_eq!(rect.width, 100.0);
        assert_eq!(rect.height, 50.0);
    }

    #[test]
    fn test_rect_expansion() {
        let rect = Rect::new(10.0, 10.0, 100.0, 50.0);
        let edge = EdgeSizes {
            top: 5.0,
            right: 10.0,
            bottom: 5.0,
            left: 10.0,
        };

        let expanded = rect.expanded_by(&edge);
        assert_eq!(expanded.x, 0.0);
        assert_eq!(expanded.y, 5.0);
        assert_eq!(expanded.width, 120.0);
        assert_eq!(expanded.height, 60.0);
    }

    #[test]
    fn test_dimensions_padding_box() {
        let mut dims = Dimensions::default();
        dims.content = Rect::new(0.0, 0.0, 100.0, 50.0);
        dims.padding = EdgeSizes {
            top: 10.0,
            right: 10.0,
            bottom: 10.0,
            left: 10.0,
        };

        let padding_box = dims.padding_box();
        assert_eq!(padding_box.width, 120.0);
        assert_eq!(padding_box.height, 70.0);
    }

    #[test]
    fn test_layout_engine_creation() {
        let engine = LayoutEngine::new(800.0, 600.0);
        let viewport = engine.viewport();
        assert_eq!(viewport.width, 800.0);
        assert_eq!(viewport.height, 600.0);
    }

    #[test]
    fn test_box_type_determination() {
        // This would need proper DOM nodes to test fully
        let engine = LayoutEngine::new(800.0, 600.0);
        assert_eq!(engine.viewport_width, 800.0);
    }
}
