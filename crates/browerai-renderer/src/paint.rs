// 绘制引擎：生成绘制命令

use crate::layout::LayoutBox;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaintCommand {
    FillRect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        color: String,
    },
    DrawText {
        x: f32,
        y: f32,
        text: String,
        font_size: f32,
        color: String,
    },
    StrokeRect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        color: String,
        width_px: f32,
    },
}

pub struct PaintEngine;

impl PaintEngine {
    pub fn generate(layout: &LayoutBox) -> anyhow::Result<Vec<PaintCommand>> {
        let mut commands = Vec::new();
        Self::traverse(layout, &mut commands)?;
        Ok(commands)
    }

    fn traverse(box_: &LayoutBox, commands: &mut Vec<PaintCommand>) -> anyhow::Result<()> {
        // 绘制此盒子背景
        commands.push(PaintCommand::FillRect {
            x: box_.rect.x,
            y: box_.rect.y,
            width: box_.rect.width,
            height: box_.rect.height,
            color: "#ffffff".to_string(),
        });

        // 如果是文本节点，绘制文本
        if box_.node_tag == "text" {
            // 提取文本内容（简化处理）
            commands.push(PaintCommand::DrawText {
                x: box_.rect.x,
                y: box_.rect.y,
                text: box_.node_tag.clone(),
                font_size: 16.0,
                color: "#000000".to_string(),
            });
        }

        // 递归处理子节点
        for child in &box_.children {
            Self::traverse(child, commands)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{BoxType, Rect};

    #[test]
    fn test_paint_simple() {
        let layout = LayoutBox {
            node_tag: "div".to_string(),
            rect: Rect {
                x: 10.0,
                y: 10.0,
                width: 100.0,
                height: 50.0,
            },
            children: vec![],
            box_type: BoxType::Block,
        };

        let commands = PaintEngine::generate(&layout).unwrap();
        assert!(!commands.is_empty(), "Should generate paint commands");
    }
}
