use anyhow::Result;

use super::layout::{LayoutBox, Rect};

/// Represents a color in RGBA format
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r, g, b, 255)
    }

    #[allow(dead_code)]
    pub fn black() -> Self {
        Self::rgb(0, 0, 0)
    }

    pub fn white() -> Self {
        Self::rgb(255, 255, 255)
    }

    pub fn transparent() -> Self {
        Self::new(0, 0, 0, 0)
    }
}

/// Paint operation to be rendered
#[derive(Debug, Clone)]
pub enum PaintOperation {
    SolidRect {
        rect: Rect,
        color: Color,
    },
    Border {
        rect: Rect,
        color: Color,
        width: f32,
    },
    Text {
        position: (f32, f32),
        text: String,
        color: Color,
        font_size: f32,
    },
    Image {
        rect: Rect,
        url: String,
    },
}

/// Paint engine for rendering visual output
pub struct PaintEngine {
    operations: Vec<PaintOperation>,
    background_color: Color,
    viewport_width: f32,
    viewport_height: f32,
}

impl PaintEngine {
    /// Create a new paint engine with viewport dimensions
    pub fn new() -> Self {
        Self::with_viewport(800.0, 600.0)
    }

    /// Create a new paint engine with custom viewport
    pub fn with_viewport(width: f32, height: f32) -> Self {
        Self {
            operations: Vec::new(),
            background_color: Color::white(),
            viewport_width: width,
            viewport_height: height,
        }
    }

    /// Set background color
    #[allow(dead_code)]
    pub fn set_background_color(&mut self, color: Color) {
        self.background_color = color;
    }

    /// Paint a layout box tree
    pub fn paint_layout_tree(&mut self, layout_box: &LayoutBox) -> Result<()> {
        log::info!("Painting layout tree");

        // Paint background using actual viewport dimensions
        let viewport_rect = Rect::new(0.0, 0.0, self.viewport_width, self.viewport_height);
        self.operations.push(PaintOperation::SolidRect {
            rect: viewport_rect,
            color: self.background_color,
        });

        // Paint the layout box and its children
        self.paint_box(layout_box)?;

        log::info!("Paint operations generated: {}", self.operations.len());
        Ok(())
    }

    /// Paint a single layout box
    fn paint_box(&mut self, layout_box: &LayoutBox) -> Result<()> {
        // Paint background
        self.paint_background(layout_box)?;

        // Paint border
        self.paint_border(layout_box)?;

        // Paint children
        for child in &layout_box.children {
            self.paint_box(child)?;
        }

        Ok(())
    }

    /// Paint background of a box
    fn paint_background(&mut self, layout_box: &LayoutBox) -> Result<()> {
        // Use light gray for block elements, transparent for inline
        let bg_color = match layout_box.box_type {
            super::layout::BoxType::Block => Color::rgb(240, 240, 240),
            _ => Color::transparent(),
        };

        if bg_color.a > 0 {
            self.operations.push(PaintOperation::SolidRect {
                rect: layout_box.dimensions.border_box(),
                color: bg_color,
            });
        }

        Ok(())
    }

    /// Paint border of a box
    fn paint_border(&mut self, layout_box: &LayoutBox) -> Result<()> {
        let border_width = layout_box.dimensions.border.top;

        if border_width > 0.0 {
            self.operations.push(PaintOperation::Border {
                rect: layout_box.dimensions.border_box(),
                color: Color::rgb(200, 200, 200),
                width: border_width,
            });
        }

        Ok(())
    }

    /// Get all paint operations
    #[allow(dead_code)]
    pub fn operations(&self) -> &[PaintOperation] {
        &self.operations
    }

    /// Clear all paint operations
    pub fn clear(&mut self) {
        self.operations.clear();
    }

    /// Render to a simple text representation (for testing)
    #[allow(dead_code)]
    pub fn render_to_text(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("Background: {:?}\n", self.background_color));
        output.push_str(&format!("Operations: {}\n", self.operations.len()));

        for (i, op) in self.operations.iter().enumerate() {
            output.push_str(&format!("  {}. {:?}\n", i + 1, op));
        }

        output
    }

    /// Generate paint commands optimized for rendering
    #[allow(dead_code)]
    pub fn generate_commands(&self) -> Vec<String> {
        let mut commands = Vec::new();

        for op in &self.operations {
            match op {
                PaintOperation::SolidRect { rect, color } => {
                    commands.push(format!(
                        "FILL_RECT({}, {}, {}, {}) COLOR({}, {}, {}, {})",
                        rect.x, rect.y, rect.width, rect.height, color.r, color.g, color.b, color.a
                    ));
                }
                PaintOperation::Border { rect, color, width } => {
                    commands.push(format!(
                        "STROKE_RECT({}, {}, {}, {}) COLOR({}, {}, {}, {}) WIDTH({})",
                        rect.x,
                        rect.y,
                        rect.width,
                        rect.height,
                        color.r,
                        color.g,
                        color.b,
                        color.a,
                        width
                    ));
                }
                PaintOperation::Text {
                    position,
                    text,
                    color,
                    font_size,
                } => {
                    commands.push(format!(
                        "DRAW_TEXT({}, {}) TEXT(\"{}\") COLOR({}, {}, {}, {}) SIZE({})",
                        position.0, position.1, text, color.r, color.g, color.b, color.a, font_size
                    ));
                }
                PaintOperation::Image { rect, url } => {
                    commands.push(format!(
                        "DRAW_IMAGE({}, {}, {}, {}) URL(\"{}\")",
                        rect.x, rect.y, rect.width, rect.height, url
                    ));
                }
            }
        }

        commands
    }
}

impl Default for PaintEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use browerai_renderer_core::layout::{BoxType, Dimensions, LayoutBox};

    #[test]
    fn test_color_creation() {
        let color = Color::rgb(255, 0, 0);
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 0);
        assert_eq!(color.b, 0);
        assert_eq!(color.a, 255);
    }

    #[test]
    fn test_color_transparent() {
        let color = Color::transparent();
        assert_eq!(color.a, 0);
    }

    #[test]
    fn test_paint_engine_creation() {
        let engine = PaintEngine::new();
        assert_eq!(engine.operations().len(), 0);
    }

    #[test]
    fn test_paint_simple_box() {
        let mut engine = PaintEngine::new();

        let layout_box = LayoutBox {
            box_type: BoxType::Block,
            dimensions: Dimensions::default(),
            children: Vec::new(),
            element_type: "div".to_string(),
        };

        let result = engine.paint_layout_tree(&layout_box);
        assert!(result.is_ok());
        assert!(!engine.operations().is_empty());
    }

    #[test]
    fn test_paint_commands_generation() {
        let mut engine = PaintEngine::new();

        engine.operations.push(PaintOperation::SolidRect {
            rect: Rect::new(0.0, 0.0, 100.0, 50.0),
            color: Color::rgb(255, 0, 0),
        });

        let commands = engine.generate_commands();
        assert_eq!(commands.len(), 1);
        assert!(commands[0].contains("FILL_RECT"));
    }

    #[test]
    fn test_render_to_text() {
        let mut engine = PaintEngine::new();
        engine.set_background_color(Color::rgb(255, 255, 255));

        let text = engine.render_to_text();
        assert!(text.contains("Background"));
        assert!(text.contains("Operations"));
    }
}
