//! 样式推断器
//! 从视觉分析中推断CSS样式

use crate::*;
use anyhow::Result;
use image::DynamicImage;

/// 样式推断器
pub struct StyleInferencer {
    config: VisualLearningConfig,
}

impl StyleInferencer {
    pub fn new(config: &VisualLearningConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// 推断样式系统
    pub fn infer_styles(
        &self,
        image: &DynamicImage,
        components: &[VisualComponent],
    ) -> Result<(TypographySystem, SpacingSystem)> {
        let typography = self.infer_typography(image, components)?;
        let spacing = self.infer_spacing(image, components)?;

        Ok((typography, spacing))
    }

    /// 推断排版系统
    fn infer_typography(
        &self,
        _image: &DynamicImage,
        components: &[VisualComponent],
    ) -> Result<TypographySystem> {
        let mut font_sizes = Vec::new();
        let mut font_weights = Vec::new();

        // 从组件推断字体大小
        for component in components {
            match component.component_type {
                ComponentType::Header => {
                    font_sizes.push(32);
                    font_weights.push(700);
                }
                ComponentType::Button => {
                    font_sizes.push(14);
                    font_weights.push(600);
                }
                ComponentType::Text => {
                    font_sizes.push(16);
                    font_weights.push(400);
                }
                ComponentType::Navigation => {
                    font_sizes.push(14);
                    font_weights.push(500);
                }
                _ => {}
            }
        }

        // 去重并排序
        font_sizes.sort_unstable();
        font_sizes.dedup();

        font_weights.sort_unstable();
        font_weights.dedup();

        Ok(TypographySystem {
            font_families: vec!["system-ui".to_string(), "sans-serif".to_string()],
            font_sizes,
            font_weights,
            line_heights: vec![1.5, 1.6, 1.75],
            letter_spacings: vec![-0.02, 0.0, 0.02],
        })
    }

    /// 推断间距系统
    fn infer_spacing(
        &self,
        _image: &DynamicImage,
        components: &[VisualComponent],
    ) -> Result<SpacingSystem> {
        let mut gaps = Vec::new();
        let mut paddings = Vec::new();
        let mut margins = Vec::new();

        // 分析组件之间的间距
        for (i, comp1) in components.iter().enumerate() {
            for comp2 in components.iter().skip(i + 1) {
                let gap = self.calculate_gap(&comp1.bounding_box, &comp2.bounding_box);
                if gap > 0 && gap < 100 {
                    gaps.push(gap as u32);
                }
            }

            // 推断内边距（基于组件内容和边框的距离）
            let padding = self.estimate_padding(comp1);
            paddings.push(padding);

            // 推断外边距
            let margin = self.estimate_margin(comp1);
            margins.push(margin);
        }

        // 找出最常见的间距值
        let common_gaps = self.find_common_values(&gaps, 5);
        let common_paddings = self.find_common_values(&paddings, 5);
        let common_margins = self.find_common_values(&margins, 5);

        // 推断基础单位
        let base_unit = common_gaps
            .first()
            .copied()
            .or(common_paddings.first().copied())
            .or(common_margins.first().copied())
            .unwrap_or(8) as u8;

        // 生成间距比例
        let scale = vec![
            base_unit,
            base_unit * 2,
            base_unit * 3,
            base_unit * 4,
            base_unit * 6,
            base_unit * 8,
        ];

        Ok(SpacingSystem {
            base_unit,
            scale,
            common_gaps,
            common_paddings,
            common_margins,
        })
    }

    /// 计算两个组件之间的间距
    fn calculate_gap(&self, bbox1: &BoundingBox, bbox2: &BoundingBox) -> i32 {
        let horizontal_gap = if bbox1.x + bbox1.width <= bbox2.x {
            bbox2.x as i32 - (bbox1.x + bbox1.width) as i32
        } else if bbox2.x + bbox2.width <= bbox1.x {
            bbox1.x as i32 - (bbox2.x + bbox2.width) as i32
        } else {
            -1
        };

        let vertical_gap = if bbox1.y + bbox1.height <= bbox2.y {
            bbox2.y as i32 - (bbox1.y + bbox1.height) as i32
        } else if bbox2.y + bbox2.height <= bbox1.y {
            bbox1.y as i32 - (bbox2.y + bbox2.height) as i32
        } else {
            -1
        };

        if horizontal_gap >= 0 && vertical_gap >= 0 {
            horizontal_gap.min(vertical_gap)
        } else if horizontal_gap >= 0 {
            horizontal_gap
        } else if vertical_gap >= 0 {
            vertical_gap
        } else {
            -1
        }
    }

    /// 估计内边距
    fn estimate_padding(&self, component: &VisualComponent) -> u32 {
        // 基于组件类型推断典型内边距
        match component.component_type {
            ComponentType::Button => 8,
            ComponentType::Input => 12,
            ComponentType::Card => 16,
            ComponentType::Modal => 24,
            _ => 8,
        }
    }

    /// 估计外边距
    fn estimate_margin(&self, component: &VisualComponent) -> u32 {
        // 基于组件类型推断典型外边距
        match component.component_type {
            ComponentType::Header => 0,
            ComponentType::Card => 16,
            ComponentType::Button => 8,
            _ => 8,
        }
    }

    /// 找出最常见的值
    fn find_common_values(&self, values: &[u32], count: usize) -> Vec<u32> {
        use std::collections::HashMap;

        let mut counts: HashMap<u32, usize> = HashMap::new();

        for &value in values {
            // 量化值（四舍五入到最近的4的倍数）
            let quantized = ((value + 2) / 4) * 4;
            *counts.entry(quantized).or_insert(0) += 1;
        }

        let mut sorted: Vec<_> = counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        sorted.iter().take(count).map(|(&value, _)| value).collect()
    }

    /// 生成CSS样式
    pub fn generate_component_styles(&self, component: &VisualComponent) -> String {
        let mut css = String::new();

        // 基础样式
        css.push_str(&format!(
            ".{} {{\n",
            format!("{:?}", component.component_type).to_lowercase()
        ));

        // 尺寸
        css.push_str(&format!("  width: {}px;\n", component.bounding_box.width));
        css.push_str(&format!("  height: {}px;\n", component.bounding_box.height));

        // 背景色
        if let Some(ref color) = component.visual_style.background_color {
            css.push_str(&format!("  background-color: {};\n", color.to_hex()));
        }

        // 文本色
        if let Some(ref color) = component.visual_style.text_color {
            css.push_str(&format!("  color: {};\n", color.to_hex()));
        }

        // 边框
        if component.visual_style.border_width > 0 {
            css.push_str(&format!(
                "  border-width: {}px;\n",
                component.visual_style.border_width
            ));
            css.push_str("  border-style: solid;\n");
            if let Some(ref color) = component.visual_style.border_color {
                css.push_str(&format!("  border-color: {};\n", color.to_hex()));
            }
        }

        // 圆角
        if component.visual_style.border_radius > 0 {
            css.push_str(&format!(
                "  border-radius: {}px;\n",
                component.visual_style.border_radius
            ));
        }

        // 阴影
        if let Some(ref shadow) = component.visual_style.shadow {
            css.push_str(&format!(
                "  box-shadow: {}px {}px {}px {}px {};\n",
                shadow.offset_x,
                shadow.offset_y,
                shadow.blur_radius,
                shadow.spread_radius,
                shadow.color.to_rgba()
            ));
        }

        // 透明度
        if component.visual_style.opacity < 1.0 {
            css.push_str(&format!("  opacity: {};\n", component.visual_style.opacity));
        }

        // 组件特定样式
        match component.component_type {
            ComponentType::Button => {
                css.push_str("  cursor: pointer;\n");
                css.push_str("  display: inline-flex;\n");
                css.push_str("  align-items: center;\n");
                css.push_str("  justify-content: center;\n");
            }
            ComponentType::Input => {
                css.push_str("  display: block;\n");
                css.push_str("  padding: 8px 12px;\n");
            }
            ComponentType::Card => {
                css.push_str("  overflow: hidden;\n");
            }
            ComponentType::Modal => {
                css.push_str("  position: fixed;\n");
                css.push_str("  z-index: 1000;\n");
            }
            _ => {}
        }

        css.push_str("}\n");
        css
    }

    /// 生成完整的CSS
    pub fn generate_full_css(&self, analysis: &VisualAnalysis) -> String {
        let mut css = String::new();

        // CSS变量
        css.push_str(&self.generate_css_variables(analysis));
        css.push('\n');

        // 基础样式
        css.push_str("/* Base Styles */\n");
        css.push_str("* {\n");
        css.push_str("  box-sizing: border-box;\n");
        css.push_str("}\n\n");

        css.push_str("body {\n");
        if let Some(ref color) = analysis.color_scheme.background {
            css.push_str(&format!("  background-color: {};\n", color.to_hex()));
        }
        if let Some(ref color) = analysis.color_scheme.text_primary {
            css.push_str(&format!("  color: {};\n", color.to_hex()));
        }
        if !analysis.typography.font_families.is_empty() {
            css.push_str(&format!(
                "  font-family: {};\n",
                analysis.typography.font_families.join(", ")
            ));
        }
        css.push_str("}\n\n");

        // 组件样式
        css.push_str("/* Component Styles */\n");
        for component in &analysis.components {
            css.push_str(&self.generate_component_styles(component));
            css.push('\n');
        }

        // 布局样式
        css.push_str("/* Layout Styles */\n");
        css.push_str(&LayoutAnalyzer::new(&self.config).generate_layout_css(&analysis.layout));

        css
    }

    /// 生成CSS变量
    fn generate_css_variables(&self, analysis: &VisualAnalysis) -> String {
        let mut css = String::from(":root {\n");

        // 颜色变量
        if let Some(ref c) = analysis.color_scheme.primary {
            css.push_str(&format!("  --color-primary: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.secondary {
            css.push_str(&format!("  --color-secondary: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.background {
            css.push_str(&format!("  --color-background: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.surface {
            css.push_str(&format!("  --color-surface: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.text_primary {
            css.push_str(&format!("  --color-text-primary: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.text_secondary {
            css.push_str(&format!("  --color-text-secondary: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.accent {
            css.push_str(&format!("  --color-accent: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.error {
            css.push_str(&format!("  --color-error: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.warning {
            css.push_str(&format!("  --color-warning: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.success {
            css.push_str(&format!("  --color-success: {};\n", c.to_hex()));
        }

        // 间距变量
        css.push_str(&format!(
            "  --spacing-unit: {}px;\n",
            analysis.spacing.base_unit
        ));
        for (i, &value) in analysis.spacing.scale.iter().enumerate() {
            css.push_str(&format!("  --spacing-{}: {}px;\n", i + 1, value));
        }

        // 排版变量
        for (i, &size) in analysis.typography.font_sizes.iter().enumerate() {
            css.push_str(&format!("  --font-size-{}: {}px;\n", i + 1, size));
        }

        // 圆角变量
        css.push_str("  --radius-sm: 4px;\n");
        css.push_str("  --radius-md: 8px;\n");
        css.push_str("  --radius-lg: 12px;\n");
        css.push_str("  --radius-full: 9999px;\n");

        css.push_str("}\n");
        css
    }
}
