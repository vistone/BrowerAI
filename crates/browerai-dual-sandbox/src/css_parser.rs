//! CSS 解析器 - 使用正则提取样式信息

use crate::common::{Color, ColorScheme, CssRule, FontFamily, FontSize, TypographySystem};
use std::collections::HashMap;

/// CSS 解析器
pub struct CssParser;

/// 解析结果
#[derive(Debug, Default)]
pub struct ParsedCss {
    /// 规则列表
    pub rules: Vec<CssRule>,
    /// 颜色方案
    pub colors: ColorScheme,
    /// 字体系统
    pub typography: TypographySystem,
}

impl CssParser {
    /// 创建新的 CSS 解析器
    pub fn new() -> Self {
        Self
    }

    /// 解析 CSS 内容
    pub fn parse(&self, css: &str, source_file: &str) -> ParsedCss {
        let mut parsed = ParsedCss::default();
        
        // 使用正则提取规则
        let rule_re = regex::Regex::new(r"([^{}]+)\{([^}]*)\}").unwrap();
        
        for cap in rule_re.captures_iter(css) {
            let selector = cap[1].trim().to_string();
            let declarations_str = &cap[2];
            
            let mut declarations = HashMap::new();
            
            // 解析声明
            for decl in declarations_str.split(';') {
                if let Some(pos) = decl.find(':') {
                    let property = decl[..pos].trim().to_string();
                    let value = decl[pos + 1..].trim().to_string();
                    
                    // 提取样式信息
                    self.extract_style_info(&property, &value, &mut parsed);
                    
                    declarations.insert(property, value);
                }
            }
            
            if !selector.is_empty() && !declarations.is_empty() {
                parsed.rules.push(CssRule {
                    selector,
                    declarations,
                    media_query: None,
                    source_file: source_file.to_string(),
                    specificity: 0,
                });
            }
        }
        
        parsed
    }

    /// 提取样式信息
    fn extract_style_info(&self, property: &str, value: &str, parsed: &mut ParsedCss) {
        match property {
            "color" | "background-color" | "border-color" | "background" => {
                if let Some(color) = self.parse_color(value, property) {
                    self.categorize_color(color, property, &mut parsed.colors);
                }
            }
            "font-family" => {
                if let Some(font) = self.parse_font_family(value) {
                    parsed.typography.font_families.push(font);
                }
            }
            "font-size" => {
                if let Some(size) = self.parse_font_size(value) {
                    parsed.typography.font_sizes.push(size);
                }
            }
            "line-height" => {
                if let Ok(lh) = value.parse::<f32>() {
                    parsed.typography.line_heights.push(lh);
                }
            }
            "font-weight" => {
                if let Ok(weight) = value.parse::<u16>() {
                    parsed.typography.font_weights.push(weight);
                }
            }
            _ => {}
        }
    }

    /// 解析颜色
    fn parse_color(&self, value: &str, context: &str) -> Option<Color> {
        let value = value.trim();
        
        // 解析 hex
        if value.starts_with('#') {
            return self.parse_hex_color(value, context);
        }
        
        // 解析 rgb/rgba
        if value.starts_with("rgb") {
            return self.parse_rgb_color(value, context);
        }
        
        // 命名颜色
        if let Some(hex) = self.named_color_to_hex(value) {
            return self.parse_hex_color(&hex, context);
        }
        
        None
    }

    /// 解析 hex 颜色
    fn parse_hex_color(&self, hex: &str, context: &str) -> Option<Color> {
        let hex = hex.trim_start_matches('#');
        
        let (r, g, b) = match hex.len() {
            3 => {
                // #RGB
                let r = u8::from_str_radix(&hex[0..1].repeat(2), 16).ok()?;
                let g = u8::from_str_radix(&hex[1..2].repeat(2), 16).ok()?;
                let b = u8::from_str_radix(&hex[2..3].repeat(2), 16).ok()?;
                (r, g, b)
            }
            6 => {
                // #RRGGBB
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                (r, g, b)
            }
            _ => return None,
        };
        
        Some(Color {
            raw: format!("#{}", hex),
            hex: format!("#{:02x}{:02x}{:02x}", r, g, b),
            rgb: (r, g, b),
            alpha: 1.0,
            usage_count: 1,
            usage_context: vec![context.to_string()],
        })
    }

    /// 解析 rgb 颜色
    fn parse_rgb_color(&self, value: &str, context: &str) -> Option<Color> {
        // 简化实现 - 提取 rgb 值
        let re = regex::Regex::new(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)").unwrap();
        
        if let Some(cap) = re.captures(value) {
            let r = cap[1].parse::<u8>().ok()?;
            let g = cap[2].parse::<u8>().ok()?;
            let b = cap[3].parse::<u8>().ok()?;
            
            return Some(Color {
                raw: value.to_string(),
                hex: format!("#{:02x}{:02x}{:02x}", r, g, b),
                rgb: (r, g, b),
                alpha: 1.0,
                usage_count: 1,
                usage_context: vec![context.to_string()],
            });
        }
        
        None
    }

    /// 命名颜色转 hex
    fn named_color_to_hex(&self, name: &str) -> Option<String> {
        let colors: HashMap<&str, &str> = [
            ("black", "#000000"),
            ("white", "#ffffff"),
            ("red", "#ff0000"),
            ("green", "#008000"),
            ("blue", "#0000ff"),
            ("yellow", "#ffff00"),
            ("cyan", "#00ffff"),
            ("magenta", "#ff00ff"),
            ("silver", "#c0c0c0"),
            ("gray", "#808080"),
            ("grey", "#808080"),
            ("maroon", "#800000"),
            ("olive", "#808000"),
            ("lime", "#00ff00"),
            ("aqua", "#00ffff"),
            ("teal", "#008080"),
            ("navy", "#000080"),
            ("fuchsia", "#ff00ff"),
            ("purple", "#800080"),
        ].iter().cloned().collect();
        
        colors.get(name).map(|&s| s.to_string())
    }

    /// 分类颜色
    fn categorize_color(&self, color: Color, property: &str, scheme: &mut ColorScheme) {
        if property.contains("background") {
            scheme.background_colors.push(color);
        } else if property == "color" {
            scheme.text_colors.push(color);
        } else if property.contains("border") {
            scheme.accent_colors.push(color);
        } else {
            scheme.primary_colors.push(color);
        }
    }

    /// 解析字体家族
    fn parse_font_family(&self, value: &str) -> Option<FontFamily> {
        let families: Vec<String> = value
            .split(',')
            .map(|s| {
                s.trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string()
            })
            .filter(|s| !s.is_empty())
            .collect();
        
        if families.is_empty() {
            return None;
        }
        
        let name = families[0].clone();
        let fallbacks = families[1..].to_vec();
        
        let source = if name.to_lowercase().contains("google") {
            "google-fonts"
        } else if ["Arial", "Helvetica", "Times", "Courier", "Georgia", "Verdana"].contains(&name.as_str()) {
            "system"
        } else {
            "custom"
        }.to_string();
        
        Some(FontFamily {
            name,
            fallbacks,
            usage_count: 1,
            source,
        })
    }

    /// 解析字体大小
    fn parse_font_size(&self, value: &str) -> Option<FontSize> {
        let value = value.trim();
        
        let pixels = if value.ends_with("px") {
            value.trim_end_matches("px").parse::<f32>().ok()?
        } else if value.ends_with("rem") {
            value.trim_end_matches("rem").parse::<f32>().ok()? * 16.0
        } else if value.ends_with("em") {
            value.trim_end_matches("em").parse::<f32>().ok()? * 16.0
        } else if value.ends_with("pt") {
            value.trim_end_matches("pt").parse::<f32>().ok()? * 1.33
        } else {
            return None;
        };
        
        Some(FontSize {
            value: value.to_string(),
            pixels,
            context: "body".to_string(),
        })
    }
}

impl Default for CssParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_css() {
        let parser = CssParser::new();
        let css = r#"
            body {
                color: #333333;
                background-color: #ffffff;
                font-family: Arial, sans-serif;
                font-size: 16px;
            }
            h1 {
                color: #0066cc;
                font-size: 24px;
            }
        "#;
        
        let parsed = parser.parse(css, "test.css");
        
        assert!(!parsed.rules.is_empty());
        assert!(!parsed.colors.text_colors.is_empty());
        assert!(!parsed.typography.font_families.is_empty());
    }

    #[test]
    fn test_parse_hex_colors() {
        let parser = CssParser::new();
        
        let color1 = parser.parse_hex_color("#ff0000", "test");
        assert!(color1.is_some());
        let c = color1.unwrap();
        assert_eq!(c.rgb, (255, 0, 0));
        
        let color2 = parser.parse_hex_color("#f00", "test");
        assert!(color2.is_some());
    }

    #[test]
    fn test_named_colors() {
        let parser = CssParser::new();
        
        assert_eq!(parser.named_color_to_hex("black"), Some("#000000".to_string()));
        assert_eq!(parser.named_color_to_hex("white"), Some("#ffffff".to_string()));
        assert_eq!(parser.named_color_to_hex("red"), Some("#ff0000".to_string()));
        assert_eq!(parser.named_color_to_hex("unknown"), None);
    }

    #[test]
    fn test_parse_font_family() {
        let parser = CssParser::new();
        
        let font = parser.parse_font_family("Arial, sans-serif");
        assert!(font.is_some());
        let f = font.unwrap();
        assert_eq!(f.name, "Arial");
        assert_eq!(f.source, "system");
    }

    #[test]
    fn test_parse_font_size() {
        let parser = CssParser::new();
        
        let size1 = parser.parse_font_size("16px");
        assert!(size1.is_some());
        assert_eq!(size1.unwrap().pixels, 16.0);
        
        let size2 = parser.parse_font_size("1.5rem");
        assert!(size2.is_some());
        assert_eq!(size2.unwrap().pixels, 24.0);
    }
}
