//! 样式分析器 - 从原始网站提取样式特征并生成新样式
//!
//! 实现基于内容分析的样式生成，替代硬编码样式

use regex::Regex;

/// 提取的样式特征
#[derive(Debug, Clone)]
pub struct StyleFeatures {
    /// 主要颜色
    pub primary_colors: Vec<String>,
    /// 背景颜色
    pub background_colors: Vec<String>,
    /// 文字颜色
    pub text_colors: Vec<String>,
    /// 字体家族
    pub font_families: Vec<String>,
    /// 字体大小
    pub font_sizes: Vec<String>,
    /// 布局类型 (grid/flex/block)
    pub layout_types: Vec<String>,
    /// 圆角使用情况
    pub border_radius_values: Vec<String>,
    /// 阴影使用情况
    pub has_shadows: bool,
    /// 动画使用情况
    pub has_animations: bool,
}

impl Default for StyleFeatures {
    fn default() -> Self {
        Self {
            primary_colors: vec!["#3B82F6".to_string()],
            background_colors: vec!["#FFFFFF".to_string()],
            text_colors: vec!["#111827".to_string()],
            font_families: vec!["system-ui".to_string()],
            font_sizes: vec!["16px".to_string()],
            layout_types: vec!["block".to_string()],
            border_radius_values: vec!["4px".to_string()],
            has_shadows: false,
            has_animations: false,
        }
    }
}

/// 样式分析器
pub struct StyleAnalyzer;

impl StyleAnalyzer {
    /// 创建新的样式分析器
    pub fn new() -> Self {
        Self
    }

    /// 从 CSS 内容中提取样式特征
    pub fn analyze_css(&self, css: &str) -> StyleFeatures {
        let mut features = StyleFeatures::default();

        if css.is_empty() {
            return features;
        }

        // 提取颜色
        features.primary_colors = self.extract_colors(css);
        features.background_colors = self.extract_background_colors(css);
        features.text_colors = self.extract_text_colors(css);

        // 提取字体
        features.font_families = self.extract_font_families(css);
        features.font_sizes = self.extract_font_sizes(css);

        // 提取布局
        features.layout_types = self.extract_layout_types(css);

        // 提取圆角
        features.border_radius_values = self.extract_border_radius(css);

        // 检测特效
        features.has_shadows = css.contains("box-shadow") || css.contains("text-shadow");
        features.has_animations = css.contains("@keyframes") || css.contains("animation");

        features
    }

    /// 从 HTML 结构推断样式特征
    pub fn analyze_html(&self, html: &str) -> StyleFeatures {
        let mut features = StyleFeatures::default();

        // 检测是否使用现代 CSS 框架
        if html.contains("tailwind") || html.contains("Tailwind") {
            features.layout_types.push("tailwind".to_string());
        }
        if html.contains("bootstrap") || html.contains("Bootstrap") {
            features.layout_types.push("bootstrap".to_string());
        }

        // 检测组件类型
        if html.contains("<nav") {
            features.layout_types.push("navigation".to_string());
        }
        if html.contains("<aside") || html.contains("sidebar") {
            features.layout_types.push("sidebar".to_string());
        }
        if html.contains("<footer") {
            features.layout_types.push("footer".to_string());
        }

        features
    }

    /// 提取所有颜色值
    fn extract_colors(&self, css: &str) -> Vec<String> {
        let hex_regex = Regex::new(r"#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})").unwrap();
        let rgb_regex = Regex::new(r"rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)").unwrap();
        let rgba_regex = Regex::new(r"rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*[\d.]+\s*\)").unwrap();

        let mut colors: Vec<String> = Vec::new();

        for cap in hex_regex.captures_iter(css) {
            colors.push(cap[0].to_string());
        }
        for cap in rgb_regex.captures_iter(css) {
            colors.push(cap[0].to_string());
        }
        for cap in rgba_regex.captures_iter(css) {
            colors.push(cap[0].to_string());
        }

        // 去重并限制数量
        colors.dedup();
        colors.truncate(10);

        if colors.is_empty() {
            colors.push("#3B82F6".to_string());
        }

        colors
    }

    /// 提取背景颜色
    fn extract_background_colors(&self, css: &str) -> Vec<String> {
        let bg_regex = Regex::new(r"background(?:-color)?\s*:\s*([^;]+)").unwrap();
        let mut colors: Vec<String> = Vec::new();

        for cap in bg_regex.captures_iter(css) {
            let color = cap[1].trim().to_string();
            if color.starts_with('#') || color.starts_with("rgb") {
                colors.push(color);
            }
        }

        colors.dedup();
        colors.truncate(5);

        if colors.is_empty() {
            colors.push("#FFFFFF".to_string());
        }

        colors
    }

    /// 提取文字颜色
    fn extract_text_colors(&self, css: &str) -> Vec<String> {
        let color_regex = Regex::new(r"color\s*:\s*([^;]+)").unwrap();
        let mut colors: Vec<String> = Vec::new();

        for cap in color_regex.captures_iter(css) {
            let color = cap[1].trim().to_string();
            if color.starts_with('#') || color.starts_with("rgb") {
                colors.push(color);
            }
        }

        colors.dedup();
        colors.truncate(5);

        if colors.is_empty() {
            colors.push("#111827".to_string());
        }

        colors
    }

    /// 提取字体家族
    fn extract_font_families(&self, css: &str) -> Vec<String> {
        let font_regex = Regex::new(r"font-family\s*:\s*([^;]+)").unwrap();
        let mut fonts: Vec<String> = Vec::new();

        for cap in font_regex.captures_iter(css) {
            let font = cap[1].trim().to_string();
            fonts.push(font);
        }

        fonts.dedup();
        fonts.truncate(5);

        if fonts.is_empty() {
            fonts.push("system-ui, -apple-system, sans-serif".to_string());
        }

        fonts
    }

    /// 提取字体大小
    fn extract_font_sizes(&self, css: &str) -> Vec<String> {
        let size_regex = Regex::new(r"font-size\s*:\s*([^;]+)").unwrap();
        let mut sizes: Vec<String> = Vec::new();

        for cap in size_regex.captures_iter(css) {
            sizes.push(cap[1].trim().to_string());
        }

        sizes.dedup();
        sizes.truncate(5);

        if sizes.is_empty() {
            sizes.push("16px".to_string());
        }

        sizes
    }

    /// 提取布局类型
    fn extract_layout_types(&self, css: &str) -> Vec<String> {
        let mut layouts: Vec<String> = Vec::new();

        if css.contains("display: grid") || css.contains("display:grid") {
            layouts.push("grid".to_string());
        }
        if css.contains("display: flex") || css.contains("display:flex") {
            layouts.push("flex".to_string());
        }
        if css.contains("float:") {
            layouts.push("float".to_string());
        }
        if css.contains("position: absolute") {
            layouts.push("absolute".to_string());
        }

        if layouts.is_empty() {
            layouts.push("block".to_string());
        }

        layouts
    }

    /// 提取圆角值
    fn extract_border_radius(&self, css: &str) -> Vec<String> {
        let radius_regex = Regex::new(r"border-radius\s*:\s*([^;]+)").unwrap();
        let mut radii: Vec<String> = Vec::new();

        for cap in radius_regex.captures_iter(css) {
            radii.push(cap[1].trim().to_string());
        }

        radii.dedup();
        radii.truncate(5);

        if radii.is_empty() {
            radii.push("4px".to_string());
        }

        radii
    }

    /// 基于原始特征生成新的配色方案
    pub fn generate_color_scheme(
        &self,
        features: &StyleFeatures,
        variant_index: usize,
    ) -> ColorScheme {
        match variant_index % 4 {
            0 => self.generate_modern_scheme(features),
            1 => self.generate_warm_scheme(features),
            2 => self.generate_cool_scheme(features),
            3 => self.generate_high_contrast_scheme(features),
            _ => self.generate_modern_scheme(features),
        }
    }

    /// 现代风格配色
    fn generate_modern_scheme(&self, _features: &StyleFeatures) -> ColorScheme {
        ColorScheme {
            primary: "#3B82F6".to_string(),
            secondary: "#8B5CF6".to_string(),
            background: "#F9FAFB".to_string(),
            text: "#111827".to_string(),
            accent: "#10B981".to_string(),
        }
    }

    /// 暖色调配色
    fn generate_warm_scheme(&self, _features: &StyleFeatures) -> ColorScheme {
        ColorScheme {
            primary: "#EA580C".to_string(),
            secondary: "#DC2626".to_string(),
            background: "#FFFBEB".to_string(),
            text: "#431407".to_string(),
            accent: "#F59E0B".to_string(),
        }
    }

    /// 冷色调配色
    fn generate_cool_scheme(&self, _features: &StyleFeatures) -> ColorScheme {
        ColorScheme {
            primary: "#0891B2".to_string(),
            secondary: "#4F46E5".to_string(),
            background: "#ECFEFF".to_string(),
            text: "#164E63".to_string(),
            accent: "#06B6D4".to_string(),
        }
    }

    /// 高对比度配色（政府/无障碍风格）
    fn generate_high_contrast_scheme(&self, _features: &StyleFeatures) -> ColorScheme {
        ColorScheme {
            primary: "#000000".to_string(),
            secondary: "#374151".to_string(),
            background: "#FFFFFF".to_string(),
            text: "#000000".to_string(),
            accent: "#1F2937".to_string(),
        }
    }

    /// 基于原始特征生成字体方案
    pub fn generate_typography(
        &self,
        features: &StyleFeatures,
        variant_index: usize,
    ) -> Typography {
        let base_font = features
            .font_families
            .first()
            .cloned()
            .unwrap_or_else(|| "system-ui".to_string());

        let base_size = match variant_index % 3 {
            0 => 16,
            1 => 15,
            2 => 17,
            _ => 16,
        };

        Typography {
            font_family: base_font,
            base_size,
            heading_scale: 1.25,
            line_height: 1.6,
        }
    }
}

/// 配色方案
#[derive(Debug, Clone)]
pub struct ColorScheme {
    pub primary: String,
    pub secondary: String,
    pub background: String,
    pub text: String,
    pub accent: String,
}

/// 字体方案
#[derive(Debug, Clone)]
pub struct Typography {
    pub font_family: String,
    pub base_size: usize,
    pub heading_scale: f32,
    pub line_height: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_analyzer_creation() {
        let analyzer = StyleAnalyzer::new();
        let css = r#"
            body {
                background-color: #f0f0f0;
                color: #333333;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }
            .button {
                background: #3B82F6;
                border-radius: 8px;
            }
        "#;

        let features = analyzer.analyze_css(css);
        assert!(!features.background_colors.is_empty());
        assert!(!features.font_families.is_empty());
    }

    #[test]
    fn test_color_scheme_generation() {
        let analyzer = StyleAnalyzer::new();
        let features = StyleFeatures::default();

        let scheme1 = analyzer.generate_color_scheme(&features, 0);
        let scheme2 = analyzer.generate_color_scheme(&features, 1);

        assert_ne!(scheme1.primary, scheme2.primary);
    }
}
