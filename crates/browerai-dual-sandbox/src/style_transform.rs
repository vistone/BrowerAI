//! 样式变换算法 - 基于学习的样式生成新体验

use crate::common::{Color, ColorScheme, FontFamily, StyleSystem};
use serde::{Serialize, Deserialize};

/// 样式变换器
pub struct StyleTransformer;

/// 变换配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformConfig {
    /// 变换类型
    pub transform_type: TransformType,
    /// 强度 (0.0 - 1.0)
    pub intensity: f32,
}

/// 变换类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformType {
    /// 保持原样
    Original,
    /// 深色主题
    DarkTheme,
    /// 高对比度
    HighContrast,
    /// 暖色调
    WarmTone,
    /// 冷色调
    CoolTone,
    /// 极简
    Minimal,
    /// 活力
    Vibrant,
}

impl StyleTransformer {
    /// 创建新的样式变换器
    pub fn new() -> Self {
        Self
    }

    /// 变换样式
    pub fn transform(&self, original: &StyleSystem, config: &TransformConfig) -> StyleSystem {
        match config.transform_type {
            TransformType::Original => original.clone(),
            TransformType::DarkTheme => self.to_dark_theme(original),
            TransformType::HighContrast => self.to_high_contrast(original),
            TransformType::WarmTone => self.to_warm_tone(original),
            TransformType::CoolTone => self.to_cool_tone(original),
            TransformType::Minimal => self.to_minimal(original),
            TransformType::Vibrant => self.to_vibrant(original),
        }
    }

    /// 转换为深色主题
    fn to_dark_theme(&self, original: &StyleSystem) -> StyleSystem {
        let mut new = original.clone();

        // 如果原始颜色为空，使用默认的深色主题配色
        if original.colors.background_colors.is_empty() && original.colors.text_colors.is_empty() {
            // 使用标准深色主题配色
            new.colors.background_colors = vec![
                Color { raw: "#1a1a2e".to_string(), hex: "#1a1a2e".to_string(), rgb: (26, 26, 46), alpha: 1.0, usage_count: 1, usage_context: vec!["background".to_string()] },
                Color { raw: "#16213e".to_string(), hex: "#16213e".to_string(), rgb: (22, 33, 62), alpha: 1.0, usage_count: 1, usage_context: vec!["background".to_string()] },
                Color { raw: "#0f3460".to_string(), hex: "#0f3460".to_string(), rgb: (15, 52, 96), alpha: 1.0, usage_count: 1, usage_context: vec!["background".to_string()] },
            ];
            new.colors.text_colors = vec![
                Color { raw: "#eaeaea".to_string(), hex: "#eaeaea".to_string(), rgb: (234, 234, 234), alpha: 1.0, usage_count: 1, usage_context: vec!["text".to_string()] },
                Color { raw: "#a0a0a0".to_string(), hex: "#a0a0a0".to_string(), rgb: (160, 160, 160), alpha: 1.0, usage_count: 1, usage_context: vec!["text".to_string()] },
            ];
            new.colors.primary_colors = vec![
                Color { raw: "#e94560".to_string(), hex: "#e94560".to_string(), rgb: (233, 69, 96), alpha: 1.0, usage_count: 1, usage_context: vec!["primary".to_string()] },
            ];
        } else {
            // 反转背景色和文字色
            if !original.colors.text_colors.is_empty() {
                new.colors.background_colors = original.colors.text_colors.iter()
                    .map(|c| self.darken_color(c))
                    .take(5) // 限制数量
                    .collect();
            }

            if !original.colors.background_colors.is_empty() {
                new.colors.text_colors = original.colors.background_colors.iter()
                    .map(|c| self.lighten_color(c))
                    .take(5)
                    .collect();
            }

            // 调整主色
            if !original.colors.primary_colors.is_empty() {
                new.colors.primary_colors = original.colors.primary_colors.iter()
                    .map(|c| self.adjust_brightness(c, 1.2))
                    .take(5)
                    .collect();
            }
        }

        new
    }

    /// 转换为高对比度
    fn to_high_contrast(&self, original: &StyleSystem) -> StyleSystem {
        let mut new = original.clone();

        // 背景设为纯白或纯黑
        new.colors.background_colors = vec![
            Color {
                raw: "#ffffff".to_string(),
                hex: "#ffffff".to_string(),
                rgb: (255, 255, 255),
                alpha: 1.0,
                usage_count: 1,
                usage_context: vec!["background".to_string()],
            },
            Color {
                raw: "#000000".to_string(),
                hex: "#000000".to_string(),
                rgb: (0, 0, 0),
                alpha: 1.0,
                usage_count: 1,
                usage_context: vec!["background".to_string()],
            },
        ];

        // 文字设为对比色
        new.colors.text_colors = vec![
            Color {
                raw: "#000000".to_string(),
                hex: "#000000".to_string(),
                rgb: (0, 0, 0),
                alpha: 1.0,
                usage_count: 1,
                usage_context: vec!["text".to_string()],
            },
            Color {
                raw: "#ffffff".to_string(),
                hex: "#ffffff".to_string(),
                rgb: (255, 255, 255),
                alpha: 1.0,
                usage_count: 1,
                usage_context: vec!["text".to_string()],
            },
        ];

        // 主色保持但增强对比
        new.colors.primary_colors = original.colors.primary_colors.iter()
            .map(|c| self.enhance_contrast(c))
            .collect();

        new
    }

    /// 转换为暖色调
    fn to_warm_tone(&self, original: &StyleSystem) -> StyleSystem {
        let mut new = original.clone();

        // 暖色主色
        new.colors.primary_colors = vec![
            Color { raw: "#e74c3c".to_string(), hex: "#e74c3c".to_string(), rgb: (231, 76, 60), alpha: 1.0, usage_count: 1, usage_context: vec!["primary".to_string()] },
            Color { raw: "#f39c12".to_string(), hex: "#f39c12".to_string(), rgb: (243, 156, 18), alpha: 1.0, usage_count: 1, usage_context: vec!["secondary".to_string()] },
            Color { raw: "#d35400".to_string(), hex: "#d35400".to_string(), rgb: (211, 84, 0), alpha: 1.0, usage_count: 1, usage_context: vec!["accent".to_string()] },
        ];

        // 暖色背景
        new.colors.background_colors = vec![
            Color { raw: "#fff5f0".to_string(), hex: "#fff5f0".to_string(), rgb: (255, 245, 240), alpha: 1.0, usage_count: 1, usage_context: vec!["background".to_string()] },
            Color { raw: "#ffecd2".to_string(), hex: "#ffecd2".to_string(), rgb: (255, 236, 210), alpha: 1.0, usage_count: 1, usage_context: vec!["background".to_string()] },
        ];

        // 暖色文字
        new.colors.text_colors = vec![
            Color { raw: "#5d4037".to_string(), hex: "#5d4037".to_string(), rgb: (93, 64, 55), alpha: 1.0, usage_count: 1, usage_context: vec!["text".to_string()] },
            Color { raw: "#8d6e63".to_string(), hex: "#8d6e63".to_string(), rgb: (141, 110, 99), alpha: 1.0, usage_count: 1, usage_context: vec!["text".to_string()] },
        ];

        new
    }

    /// 转换为冷色调
    fn to_cool_tone(&self, original: &StyleSystem) -> StyleSystem {
        let mut new = original.clone();

        // 冷色主色
        new.colors.primary_colors = vec![
            Color { raw: "#3498db".to_string(), hex: "#3498db".to_string(), rgb: (52, 152, 219), alpha: 1.0, usage_count: 1, usage_context: vec!["primary".to_string()] },
            Color { raw: "#2ecc71".to_string(), hex: "#2ecc71".to_string(), rgb: (46, 204, 113), alpha: 1.0, usage_count: 1, usage_context: vec!["secondary".to_string()] },
            Color { raw: "#1abc9c".to_string(), hex: "#1abc9c".to_string(), rgb: (26, 188, 156), alpha: 1.0, usage_count: 1, usage_context: vec!["accent".to_string()] },
        ];

        // 冷色背景
        new.colors.background_colors = vec![
            Color { raw: "#f0f8ff".to_string(), hex: "#f0f8ff".to_string(), rgb: (240, 248, 255), alpha: 1.0, usage_count: 1, usage_context: vec!["background".to_string()] },
            Color { raw: "#e8f6f3".to_string(), hex: "#e8f6f3".to_string(), rgb: (232, 246, 243), alpha: 1.0, usage_count: 1, usage_context: vec!["background".to_string()] },
        ];

        // 冷色文字
        new.colors.text_colors = vec![
            Color { raw: "#2c3e50".to_string(), hex: "#2c3e50".to_string(), rgb: (44, 62, 80), alpha: 1.0, usage_count: 1, usage_context: vec!["text".to_string()] },
            Color { raw: "#34495e".to_string(), hex: "#34495e".to_string(), rgb: (52, 73, 94), alpha: 1.0, usage_count: 1, usage_context: vec!["text".to_string()] },
        ];

        new
    }

    /// 转换为极简风格
    fn to_minimal(&self, original: &StyleSystem) -> StyleSystem {
        let mut new = original.clone();

        // 黑白灰配色
        new.colors = ColorScheme {
            primary_colors: vec![
                Color {
                    raw: "#000000".to_string(),
                    hex: "#000000".to_string(),
                    rgb: (0, 0, 0),
                    alpha: 1.0,
                    usage_count: 1,
                    usage_context: vec!["primary".to_string()],
                },
            ],
            secondary_colors: vec![
                Color {
                    raw: "#666666".to_string(),
                    hex: "#666666".to_string(),
                    rgb: (102, 102, 102),
                    alpha: 1.0,
                    usage_count: 1,
                    usage_context: vec!["secondary".to_string()],
                },
            ],
            background_colors: vec![
                Color {
                    raw: "#ffffff".to_string(),
                    hex: "#ffffff".to_string(),
                    rgb: (255, 255, 255),
                    alpha: 1.0,
                    usage_count: 1,
                    usage_context: vec!["background".to_string()],
                },
            ],
            text_colors: vec![
                Color {
                    raw: "#333333".to_string(),
                    hex: "#333333".to_string(),
                    rgb: (51, 51, 51),
                    alpha: 1.0,
                    usage_count: 1,
                    usage_context: vec!["text".to_string()],
                },
            ],
            accent_colors: vec![],
        };

        // 简化字体
        new.typography.font_families = vec![
            FontFamily {
                name: "system-ui".to_string(),
                fallbacks: vec!["-apple-system".to_string(), "sans-serif".to_string()],
                usage_count: 1,
                source: "system".to_string(),
            },
        ];

        new
    }

    /// 转换为活力风格
    fn to_vibrant(&self, original: &StyleSystem) -> StyleSystem {
        let mut new = original.clone();

        // 鲜艳配色
        new.colors.primary_colors = vec![
            Color {
                raw: "#ff6b6b".to_string(),
                hex: "#ff6b6b".to_string(),
                rgb: (255, 107, 107),
                alpha: 1.0,
                usage_count: 1,
                usage_context: vec!["primary".to_string()],
            },
            Color {
                raw: "#4ecdc4".to_string(),
                hex: "#4ecdc4".to_string(),
                rgb: (78, 205, 196),
                alpha: 1.0,
                usage_count: 1,
                usage_context: vec!["secondary".to_string()],
            },
            Color {
                raw: "#ffe66d".to_string(),
                hex: "#ffe66d".to_string(),
                rgb: (255, 230, 109),
                alpha: 1.0,
                usage_count: 1,
                usage_context: vec!["accent".to_string()],
            },
        ];

        new
    }

    /// RGB 转 Hex
    fn rgb_to_hex(&self, rgb: (u8, u8, u8)) -> String {
        format!("#{:02x}{:02x}{:02x}", rgb.0, rgb.1, rgb.2)
    }

    /// 加深颜色
    fn darken_color(&self, color: &Color) -> Color {
        let factor = 0.7;
        let new_rgb = (
            (color.rgb.0 as f32 * factor) as u8,
            (color.rgb.1 as f32 * factor) as u8,
            (color.rgb.2 as f32 * factor) as u8,
        );
        let new_hex = self.rgb_to_hex(new_rgb);
        Color {
            raw: new_hex.clone(),
            hex: new_hex,
            rgb: new_rgb,
            alpha: color.alpha,
            usage_count: color.usage_count,
            usage_context: color.usage_context.clone(),
        }
    }

    /// 提亮颜色
    fn lighten_color(&self, color: &Color) -> Color {
        let factor = 1.3;
        let new_rgb = (
            ((color.rgb.0 as f32 * factor).min(255.0)) as u8,
            ((color.rgb.1 as f32 * factor).min(255.0)) as u8,
            ((color.rgb.2 as f32 * factor).min(255.0)) as u8,
        );
        let new_hex = self.rgb_to_hex(new_rgb);
        Color {
            raw: new_hex.clone(),
            hex: new_hex,
            rgb: new_rgb,
            alpha: color.alpha,
            usage_count: color.usage_count,
            usage_context: color.usage_context.clone(),
        }
    }

    /// 调整亮度
    fn adjust_brightness(&self, color: &Color, factor: f32) -> Color {
        let new_rgb = (
            ((color.rgb.0 as f32 * factor).min(255.0)) as u8,
            ((color.rgb.1 as f32 * factor).min(255.0)) as u8,
            ((color.rgb.2 as f32 * factor).min(255.0)) as u8,
        );
        let new_hex = self.rgb_to_hex(new_rgb);
        Color {
            raw: new_hex.clone(),
            hex: new_hex,
            rgb: new_rgb,
            alpha: color.alpha,
            usage_count: color.usage_count,
            usage_context: color.usage_context.clone(),
        }
    }

    /// 增强对比度
    fn enhance_contrast(&self, color: &Color) -> Color {
        // 简单增强：让颜色更饱和
        let avg = (color.rgb.0 as f32 + color.rgb.1 as f32 + color.rgb.2 as f32) / 3.0;
        let new_rgb = (
            if color.rgb.0 as f32 > avg { ((color.rgb.0 as f32 + 30.0).min(255.0)) as u8 } else { (color.rgb.0 as f32 - 30.0).max(0.0) as u8 },
            if color.rgb.1 as f32 > avg { ((color.rgb.1 as f32 + 30.0).min(255.0)) as u8 } else { (color.rgb.1 as f32 - 30.0).max(0.0) as u8 },
            if color.rgb.2 as f32 > avg { ((color.rgb.2 as f32 + 30.0).min(255.0)) as u8 } else { (color.rgb.2 as f32 - 30.0).max(0.0) as u8 },
        );
        let new_hex = self.rgb_to_hex(new_rgb);
        Color {
            raw: new_hex.clone(),
            hex: new_hex,
            rgb: new_rgb,
            alpha: color.alpha,
            usage_count: color.usage_count,
            usage_context: color.usage_context.clone(),
        }
    }
}

/// 生成 CSS 代码
pub fn generate_css(styles: &StyleSystem) -> String {
    let mut css = String::new();

    // 生成 CSS 变量
    css.push_str(":root {\n");
    
    // 颜色变量
    for (i, color) in styles.colors.primary_colors.iter().enumerate() {
        css.push_str(&format!("  --color-primary-{}: {};\n", i, color.hex));
    }
    
    for (i, color) in styles.colors.background_colors.iter().enumerate() {
        css.push_str(&format!("  --color-bg-{}: {};\n", i, color.hex));
    }
    
    for (i, color) in styles.colors.text_colors.iter().enumerate() {
        css.push_str(&format!("  --color-text-{}: {};\n", i, color.hex));
    }
    
    // 字体变量
    if let Some(font) = styles.typography.font_families.first() {
        css.push_str(&format!("  --font-family: {};\n", font.name));
    }
    
    css.push_str("}\n\n");

    // 基础样式
    css.push_str("body {\n");
    if let Some(bg) = styles.colors.background_colors.first() {
        css.push_str(&format!("  background-color: {};\n", bg.hex));
    }
    if let Some(text) = styles.colors.text_colors.first() {
        css.push_str(&format!("  color: {};\n", text.hex));
    }
    if let Some(font) = styles.typography.font_families.first() {
        css.push_str(&format!("  font-family: {};\n", font.name));
    }
    css.push_str("}\n\n");

    // 标题样式
    css.push_str("h1, h2, h3, h4, h5, h6 {\n");
    if let Some(primary) = styles.colors.primary_colors.first() {
        css.push_str(&format!("  color: {};\n", primary.hex));
    }
    css.push_str("}\n\n");

    // 链接样式
    css.push_str("a {\n");
    if let Some(primary) = styles.colors.primary_colors.first() {
        css.push_str(&format!("  color: {};\n", primary.hex));
    }
    css.push_str("  text-decoration: none;\n");
    css.push_str("}\n\n");

    // 按钮样式
    css.push_str("button, .button {\n");
    if let Some(primary) = styles.colors.primary_colors.first() {
        css.push_str(&format!("  background-color: {};\n", primary.hex));
    }
    css.push_str("  color: white;\n");
    css.push_str("  border: none;\n");
    css.push_str("  padding: 0.5rem 1rem;\n");
    css.push_str("  border-radius: 4px;\n");
    css.push_str("  cursor: pointer;\n");
    css.push_str("}\n\n");

    css
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_dark_theme() {
        let transformer = StyleTransformer::new();
        
        // 创建有内容的原始样式
        let mut original = StyleSystem::default();
        original.colors.text_colors.push(crate::common::Color {
            raw: "#333333".to_string(),
            hex: "#333333".to_string(),
            rgb: (51, 51, 51),
            alpha: 1.0,
            usage_count: 1,
            usage_context: vec!["text".to_string()],
        });
        
        let config = TransformConfig {
            transform_type: TransformType::DarkTheme,
            intensity: 1.0,
        };
        
        let transformed = transformer.transform(&original, &config);
        
        // 深色主题应该有背景色 (从文字色转换而来)
        assert!(!transformed.colors.background_colors.is_empty());
    }

    #[test]
    fn test_transform_warm_tone() {
        let transformer = StyleTransformer::new();
        
        let original = StyleSystem::default();
        let config = TransformConfig {
            transform_type: TransformType::WarmTone,
            intensity: 1.0,
        };
        
        let transformed = transformer.transform(&original, &config);
        
        // 暖色调应该有主色
        assert!(!transformed.colors.primary_colors.is_empty());
        assert_eq!(transformed.colors.primary_colors[0].hex, "#e74c3c");
    }

    #[test]
    fn test_generate_css() {
        let styles = StyleSystem::default();
        let css = generate_css(&styles);
        
        assert!(css.contains(":root"));
        assert!(css.contains("body"));
    }
}
