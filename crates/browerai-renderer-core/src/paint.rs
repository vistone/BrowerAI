//! Paint System - 绘制系统
//!
//! 生成绘制命令，包括：
//! - 背景绘制
//! - 文本渲染
//! - 边框绘制
//! - 阴影效果
//! - 图像绘制

use browerai_core::{BrowserError, Result};
use crate::layout::ComputedLayout;
use crate::Rect;

/// 绘制引擎
#[derive(Debug, Clone)]
pub struct PaintEngine {
    /// 配置
    config: PaintConfig,
}

impl PaintEngine {
    /// 创建新的绘制引擎
    pub fn new(config: PaintConfig) -> Self {
        Self { config }
    }

    /// 生成绘制记录
    pub fn generate_paints(&self, layout: &ComputedLayout) -> Result<Vec<PaintRecord>> {
        let mut records = Vec::new();
        
        // 简化实现：为每个布局节点生成绘制命令
        for (node_id, computed_box) in &layout.nodes {
            let mut commands = Vec::new();
            
            // 背景绘制
            commands.push(PaintCommand::DrawRect {
                rect: computed_box.rect,
                color: Color::white(),
            });
            
            // 边框绘制
            if computed_box.box_model.border != crate::layout::EdgeInsets::zero() {
                commands.push(PaintCommand::DrawBorder {
                    rect: computed_box.rect,
                    border: computed_box.box_model.border,
                    color: Color::black(),
                });
            }
            
            records.push(PaintRecord {
                node_id: node_id.clone(),
                commands,
                z_index: 0,
                clip_rect: None,
            });
        }
        
        Ok(records)
    }

    /// 获取配置
    pub fn config(&self) -> &PaintConfig {
        &self.config
    }
}

impl Default for PaintEngine {
    fn default() -> Self {
        Self::new(PaintConfig::default())
    }
}

/// 绘制记录
#[derive(Debug, Clone)]
pub struct PaintRecord {
    /// 节点ID
    pub node_id: String,
    /// 绘制命令列表
    pub commands: Vec<PaintCommand>,
    /// Z轴顺序
    pub z_index: i32,
    /// 裁剪区域
    pub clip_rect: Option<Rect>,
}

/// 绘制命令
#[derive(Debug, Clone)]
pub enum PaintCommand {
    /// 绘制矩形
    DrawRect {
        /// 矩形区域
        rect: Rect,
        /// 填充颜色
        color: Color,
    },
    /// 绘制边框
    DrawBorder {
        /// 矩形区域
        rect: Rect,
        /// 边框宽度
        border: crate::layout::EdgeInsets,
        /// 边框颜色
        color: Color,
    },
    /// 绘制文本
    DrawText {
        /// 文本内容
        text: String,
        /// 位置
        x: f32,
        /// 位置Y
        y: f32,
        /// 字体大小
        font_size: f32,
        /// 颜色
        color: Color,
    },
    /// 绘制图像
    DrawImage {
        /// 图像URL
        url: String,
        /// 目标矩形
        dest_rect: Rect,
        /// 源矩形（可选，用于裁剪）
        source_rect: Option<Rect>,
    },
    /// 绘制阴影
    DrawShadow {
        /// 矩形区域
        rect: Rect,
        /// 阴影颜色
        color: Color,
        /// 模糊半径
        blur_radius: f32,
        /// 偏移
        offset_x: f32,
        /// 垂直偏移
        offset_y: f32,
    },
    /// 裁剪
    Clip {
        /// 裁剪区域
        rect: Rect,
    },
    /// 恢复裁剪
    RestoreClip,
}

/// 颜色
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    /// 红色通道
    pub r: u8,
    /// 绿色通道
    pub g: u8,
    /// 蓝色通道
    pub b: u8,
    /// 透明度
    pub a: u8,
}

impl Color {
    /// 创建新颜色
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// 从RGB创建（不透明）
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// 白色
    pub fn white() -> Self {
        Self::rgb(255, 255, 255)
    }

    /// 黑色
    pub fn black() -> Self {
        Self::rgb(0, 0, 0)
    }

    /// 透明
    pub fn transparent() -> Self {
        Self { r: 0, g: 0, b: 0, a: 0 }
    }

    /// 从十六进制字符串解析
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.trim_start_matches('#');
        
        match hex.len() {
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16)
                    .map_err(|_| BrowserError::parse("Invalid hex color"))?;
                let g = u8::from_str_radix(&hex[2..4], 16)
                    .map_err(|_| BrowserError::parse("Invalid hex color"))?;
                let b = u8::from_str_radix(&hex[4..6], 16)
                    .map_err(|_| BrowserError::parse("Invalid hex color"))?;
                Ok(Self::rgb(r, g, b))
            }
            8 => {
                let r = u8::from_str_radix(&hex[0..2], 16)
                    .map_err(|_| BrowserError::parse("Invalid hex color"))?;
                let g = u8::from_str_radix(&hex[2..4], 16)
                    .map_err(|_| BrowserError::parse("Invalid hex color"))?;
                let b = u8::from_str_radix(&hex[4..6], 16)
                    .map_err(|_| BrowserError::parse("Invalid hex color"))?;
                let a = u8::from_str_radix(&hex[6..8], 16)
                    .map_err(|_| BrowserError::parse("Invalid hex color"))?;
                Ok(Self::new(r, g, b, a))
            }
            _ => Err(BrowserError::parse("Invalid hex color length")),
        }
    }

    /// 转换为CSS字符串
    pub fn to_css_string(&self) -> String {
        if self.a == 255 {
            format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
        } else {
            format!("rgba({}, {}, {}, {})", self.r, self.g, self.b, self.a as f32 / 255.0)
        }
    }
}

/// 绘制配置
#[derive(Debug, Clone)]
pub struct PaintConfig {
    /// 抗锯齿
    pub antialias: bool,
    /// 子像素渲染
    pub subpixel_rendering: bool,
    /// 最大纹理大小
    pub max_texture_size: u32,
    /// 默认字体
    pub default_font: String,
}

impl Default for PaintConfig {
    fn default() -> Self {
        Self {
            antialias: true,
            subpixel_rendering: true,
            max_texture_size: 4096,
            default_font: "Arial".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paint_engine_creation() {
        let engine = PaintEngine::new(PaintConfig::default());
        assert!(engine.config().antialias);
    }

    #[test]
    fn test_color_creation() {
        let color = Color::rgb(255, 128, 64);
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 128);
        assert_eq!(color.b, 64);
        assert_eq!(color.a, 255);
    }

    #[test]
    fn test_color_from_hex() {
        let color = Color::from_hex("#FF8040").unwrap();
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 128);
        assert_eq!(color.b, 64);
        
        let color_with_alpha = Color::from_hex("FF804080").unwrap();
        assert_eq!(color_with_alpha.a, 128);
    }

    #[test]
    fn test_color_to_css() {
        let color = Color::rgb(255, 128, 64);
        assert_eq!(color.to_css_string(), "#ff8040");
        
        let transparent = Color::new(255, 128, 64, 128);
        assert!(transparent.to_css_string().starts_with("rgba"));
    }

    #[test]
    fn test_paint_commands() {
        let commands = [
            PaintCommand::DrawRect {
                rect: Rect::new(0.0, 0.0, 100.0, 100.0),
                color: Color::white(),
            },
            PaintCommand::DrawText {
                text: "Hello".to_string(),
                x: 10.0,
                y: 20.0,
                font_size: 16.0,
                color: Color::black(),
            },
        ];
        
        assert_eq!(commands.len(), 2);
    }
}
