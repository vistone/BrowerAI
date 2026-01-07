use anyhow::Result;
use browerai_css_parser::CssRule;
use markup5ever_rcdom::RcDom;
use std::collections::HashMap;

use super::layout::{LayoutBox, LayoutEngine};
use super::paint::{Color, PaintEngine};

/// Rendering engine with AI-powered optimizations
pub struct RenderEngine {
    layout_engine: LayoutEngine,
    paint_engine: PaintEngine,
}

impl RenderEngine {
    /// Create a new render engine with default viewport
    pub fn new() -> Self {
        Self::with_viewport(800.0, 600.0)
    }

    /// Create a new render engine with custom viewport
    pub fn with_viewport(width: f32, height: f32) -> Self {
        Self {
            layout_engine: LayoutEngine::new(width, height),
            paint_engine: PaintEngine::with_viewport(width, height),
        }
    }

    /// Render HTML DOM with CSS rules
    pub fn render(&mut self, dom: &RcDom, styles: &[CssRule]) -> Result<RenderTree> {
        log::info!("Starting render process");

        // Convert styles to HashMap for easier lookup
        let style_map = self.build_style_map(styles);

        // Build layout tree
        let mut layout_tree = self.layout_engine.build_layout_tree(dom, &style_map)?;

        // Calculate layout
        let viewport = self.layout_engine.viewport();
        self.layout_engine
            .calculate_layout(&mut layout_tree, viewport);

        // Paint
        self.paint_engine.clear();
        self.paint_engine.paint_layout_tree(&layout_tree)?;

        // Build render tree from layout and paint
        let render_tree = self.build_render_tree(&layout_tree)?;

        log::info!("Render tree created with {} nodes", render_tree.nodes.len());
        Ok(render_tree)
    }

    /// Build style map from CSS rules
    fn build_style_map(&self, _styles: &[CssRule]) -> HashMap<String, HashMap<String, String>> {
        // Placeholder: Convert CSS rules to a map
        HashMap::new()
    }

    /// Build render tree from layout tree
    fn build_render_tree(&self, layout_box: &LayoutBox) -> Result<RenderTree> {
        let mut nodes = Vec::new();

        // Convert layout box to render nodes
        self.collect_render_nodes(layout_box, &mut nodes);

        Ok(RenderTree { nodes })
    }

    /// Recursively collect render nodes from layout tree
    fn collect_render_nodes(&self, layout_box: &LayoutBox, nodes: &mut Vec<RenderNode>) {
        // Collect computed styles from layout box dimensions
        let mut styles = Vec::new();

        // Add dimensions as style properties
        let dims = &layout_box.dimensions;
        styles.push(format!("width: {}px", dims.content.width));
        styles.push(format!("height: {}px", dims.content.height));
        styles.push(format!(
            "margin: {}px {}px {}px {}px",
            dims.margin.top, dims.margin.right, dims.margin.bottom, dims.margin.left
        ));
        styles.push(format!(
            "padding: {}px {}px {}px {}px",
            dims.padding.top, dims.padding.right, dims.padding.bottom, dims.padding.left
        ));
        styles.push(format!(
            "border-width: {}px {}px {}px {}px",
            dims.border.top, dims.border.right, dims.border.bottom, dims.border.left
        ));

        // Add box type as display property
        let display = match layout_box.box_type {
            super::layout::BoxType::Block => "display: block",
            super::layout::BoxType::Inline => "display: inline",
            super::layout::BoxType::InlineBlock => "display: inline-block",
            super::layout::BoxType::Flex => "display: flex",
            super::layout::BoxType::Grid => "display: grid",
            super::layout::BoxType::Anonymous => "display: block",
        };
        styles.push(display.to_string());

        nodes.push(RenderNode {
            element_type: layout_box.element_type.clone(),
            styles,
        });

        for child in &layout_box.children {
            self.collect_render_nodes(child, nodes);
        }
    }

    /// Optimize rendering with AI
    #[allow(dead_code)]
    pub fn optimize_layout(&self, _render_tree: &mut RenderTree) -> Result<()> {
        log::debug!("AI optimization not available, using basic layout");
        Ok(())
    }

    /// Set background color
    #[allow(dead_code)]
    pub fn set_background_color(&mut self, color: Color) {
        self.paint_engine.set_background_color(color);
    }

    /// Get paint commands for rendering
    #[allow(dead_code)]
    pub fn get_paint_commands(&self) -> Vec<String> {
        self.paint_engine.generate_commands()
    }
}

impl Default for RenderEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a render tree node
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RenderNode {
    pub element_type: String,
    pub styles: Vec<String>,
}

/// Represents the complete render tree
#[derive(Debug, Clone)]
pub struct RenderTree {
    pub nodes: Vec<RenderNode>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use browerai_html_parser::HtmlParser;

    #[test]
    fn test_render_engine_creation() {
        let _engine = RenderEngine::new();
        // AI path removed; ensure object creates successfully
    }

    #[test]
    fn test_render_with_viewport() {
        let _engine = RenderEngine::with_viewport(1024.0, 768.0);
        // AI path removed; ensure object creates successfully
    }

    #[test]
    fn test_basic_render() {
        let parser = HtmlParser::new();
        let html = "<html><body><h1>Test</h1></body></html>";
        let dom = parser.parse(html).unwrap();

        let mut engine = RenderEngine::new();
        let result = engine.render(&dom, &[]);
        assert!(result.is_ok());

        let render_tree = result.unwrap();
        assert!(!render_tree.nodes.is_empty());
    }

    #[test]
    fn test_optimize_layout() {
        let mut render_tree = RenderTree {
            nodes: vec![RenderNode {
                element_type: "div".to_string(),
                styles: vec![],
            }],
        };

        let engine = RenderEngine::new();
        let result = engine.optimize_layout(&mut render_tree);
        assert!(result.is_ok());
    }

    #[test]
    fn test_paint_commands() {
        let parser = HtmlParser::new();
        let html = "<html><body><div>Test</div></body></html>";
        let dom = parser.parse(html).unwrap();

        let mut engine = RenderEngine::new();
        engine.render(&dom, &[]).unwrap();

        let commands = engine.get_paint_commands();
        assert!(!commands.is_empty());
    }

    #[test]
    fn test_background_color() {
        let mut engine = RenderEngine::new();
        engine.set_background_color(Color::rgb(255, 0, 0));
        // Background color is set - full test would require rendering
    }
}
