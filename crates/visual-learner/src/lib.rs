//! 视觉学习系统
//! 通过截图分析和计算机视觉理解页面组件

use anyhow::Result;
use image::Rgba;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod analyzer;
pub mod component_detector;
pub mod layout_analyzer;
pub mod color_extractor;
pub mod style_inferencer;

pub use analyzer::VisualAnalyzer;
pub use component_detector::ComponentDetector;
pub use layout_analyzer::LayoutAnalyzer;
pub use color_extractor::ColorExtractor;
pub use style_inferencer::StyleInferencer;

/// 视觉分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualAnalysis {
    pub screenshot_path: String,
    pub viewport: ViewportInfo,
    pub components: Vec<VisualComponent>,
    pub layout: LayoutInfo,
    pub color_scheme: ColorScheme,
    pub typography: TypographySystem,
    pub spacing: SpacingSystem,
}

/// 视口信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewportInfo {
    pub width: u32,
    pub height: u32,
    pub device_scale_factor: f64,
}

/// 视觉组件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualComponent {
    pub id: String,
    pub component_type: ComponentType,
    pub bounding_box: BoundingBox,
    pub confidence: f64,
    pub visual_style: VisualStyle,
    pub semantic_label: Option<String>,
    pub children: Vec<String>,
    pub parent: Option<String>,
}

/// 组件类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComponentType {
    Button,
    Input,
    Select,
    Checkbox,
    Radio,
    Card,
    Modal,
    Dropdown,
    Navigation,
    Tab,
    List,
    ListItem,
    Header,
    Footer,
    Sidebar,
    Image,
    Text,
    Icon,
    Avatar,
    Badge,
    Divider,
    Container,
    Unknown,
}

/// 边界框
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl BoundingBox {
    pub fn center(&self) -> (u32, u32) {
        (self.x + self.width / 2, self.y + self.height / 2)
    }
    
    pub fn area(&self) -> u32 {
        self.width * self.height
    }
    
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }
    
    pub fn contains(&self, other: &BoundingBox) -> bool {
        self.x <= other.x
            && self.x + self.width >= other.x + other.width
            && self.y <= other.y
            && self.y + self.height >= other.y + other.height
    }
}

/// 视觉样式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualStyle {
    pub background_color: Option<Color>,
    pub text_color: Option<Color>,
    pub border_color: Option<Color>,
    pub border_width: u8,
    pub border_radius: u8,
    pub shadow: Option<Shadow>,
    pub opacity: f32,
}

/// 颜色
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub fn to_hex(&self) -> String {
        if self.a == 255 {
            format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
        } else {
            format!("#{:02x}{:02x}{:02x}{:02x}", self.r, self.g, self.b, self.a)
        }
    }
    
    pub fn to_rgb(&self) -> String {
        format!("rgb({}, {}, {})", self.r, self.g, self.b)
    }
    
    pub fn to_rgba(&self) -> String {
        format!("rgba({}, {}, {}, {})", self.r, self.g, self.b, self.a as f32 / 255.0)
    }
    
    pub fn luminance(&self) -> f32 {
        let r = self.r as f32 / 255.0;
        let g = self.g as f32 / 255.0;
        let b = self.b as f32 / 255.0;
        
        0.299 * r + 0.587 * g + 0.114 * b
    }
    
    pub fn is_dark(&self) -> bool {
        self.luminance() < 0.5
    }
}

impl From<Rgba<u8>> for Color {
    fn from(rgba: Rgba<u8>) -> Self {
        Self {
            r: rgba.0[0],
            g: rgba.0[1],
            b: rgba.0[2],
            a: rgba.0[3],
        }
    }
}

/// 阴影
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shadow {
    pub offset_x: i8,
    pub offset_y: i8,
    pub blur_radius: u8,
    pub spread_radius: i8,
    pub color: Color,
}

/// 布局信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutInfo {
    pub layout_type: LayoutType,
    pub sections: Vec<LayoutSection>,
    pub grid_columns: Option<u8>,
    pub grid_rows: Option<u8>,
    pub gap: u32,
}

/// 布局类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LayoutType {
    SingleColumn,
    TwoColumn,
    ThreeColumn,
    Grid,
    FlexRow,
    FlexColumn,
    Masonry,
    Complex,
}

/// 布局区域
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutSection {
    pub name: String,
    pub bounding_box: BoundingBox,
    pub section_type: SectionType,
    pub components: Vec<String>,
}

/// 区域类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SectionType {
    Header,
    Hero,
    Content,
    Sidebar,
    Footer,
    Navigation,
    CallToAction,
    Feature,
    Testimonial,
    Pricing,
    Unknown,
}

/// 配色方案
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ColorScheme {
    pub primary: Option<Color>,
    pub secondary: Option<Color>,
    pub background: Option<Color>,
    pub surface: Option<Color>,
    pub text_primary: Option<Color>,
    pub text_secondary: Option<Color>,
    pub accent: Option<Color>,
    pub error: Option<Color>,
    pub warning: Option<Color>,
    pub success: Option<Color>,
    pub all_colors: Vec<(Color, f32)>, // 颜色和占比
}

/// 排版系统
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypographySystem {
    pub font_families: Vec<String>,
    pub font_sizes: Vec<u8>,
    pub font_weights: Vec<u16>,
    pub line_heights: Vec<f32>,
    pub letter_spacings: Vec<f32>,
}

/// 间距系统
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpacingSystem {
    pub base_unit: u8,
    pub scale: Vec<u8>,
    pub common_gaps: Vec<u32>,
    pub common_paddings: Vec<u32>,
    pub common_margins: Vec<u32>,
}

/// 视觉学习配置
#[derive(Debug, Clone)]
pub struct VisualLearningConfig {
    pub screenshot_quality: u8, // 1-100
    pub full_page: bool,
    pub capture_hover_states: bool,
    pub capture_mobile: bool,
    pub component_confidence_threshold: f64,
    pub color_cluster_count: usize,
}

impl Default for VisualLearningConfig {
    fn default() -> Self {
        Self {
            screenshot_quality: 90,
            full_page: true,
            capture_hover_states: false,
            capture_mobile: false,
            component_confidence_threshold: 0.7,
            color_cluster_count: 8,
        }
    }
}

/// 视觉学习引擎
pub struct VisualLearningEngine {
    config: VisualLearningConfig,
    analyzer: VisualAnalyzer,
    component_detector: ComponentDetector,
    layout_analyzer: LayoutAnalyzer,
    color_extractor: ColorExtractor,
    style_inferencer: StyleInferencer,
}

impl VisualLearningEngine {
    pub fn new(config: VisualLearningConfig) -> Self {
        Self {
            config: config.clone(),
            analyzer: VisualAnalyzer::new(&config),
            component_detector: ComponentDetector::new(&config),
            layout_analyzer: LayoutAnalyzer::new(&config),
            color_extractor: ColorExtractor::new(&config),
            style_inferencer: StyleInferencer::new(&config),
        }
    }

    /// 获取学习引擎配置
    pub fn config(&self) -> &VisualLearningConfig {
        &self.config
    }

    /// 获取视觉分析器
    pub fn analyzer(&self) -> &VisualAnalyzer {
        &self.analyzer
    }

    /// 分析页面截图
    pub async fn analyze_screenshot(&self, image_path: &str) -> Result<VisualAnalysis> {
        // 加载图片
        let image = image::open(image_path)?;
        
        // 检测组件
        let components = self.component_detector.detect_components(&image)?;
        
        // 分析布局
        let layout = self.layout_analyzer.analyze_layout(&image, &components)?;
        
        // 提取颜色
        let color_scheme = self.color_extractor.extract_colors(&image)?;
        
        // 推断样式
        let (typography, spacing) = self.style_inferencer.infer_styles(&image, &components)?;
        
        Ok(VisualAnalysis {
            screenshot_path: image_path.to_string(),
            viewport: ViewportInfo {
                width: image.width(),
                height: image.height(),
                device_scale_factor: 1.0,
            },
            components,
            layout,
            color_scheme,
            typography,
            spacing,
        })
    }

    /// 从URL分析页面
    pub async fn analyze_url(&self, url: &str) -> Result<VisualAnalysis> {
        // 使用 Playwright 截图
        let screenshot_path = self.capture_screenshot(url).await?;
        
        // 分析截图
        self.analyze_screenshot(&screenshot_path).await
    }

    /// 捕获页面截图
    async fn capture_screenshot(&self, url: &str) -> Result<String> {
        use playwright::Playwright;
        
        let playwright = Playwright::initialize().await?;
        let browser = playwright.chromium().launcher().headless(true).launch().await?;
        let context = browser.context_builder()
            .viewport(Some(playwright::api::Viewport {
                width: 1280,
                height: 720,
            }))
            .build()
            .await?;
        
        let page = context.new_page().await?;
        page.goto_builder(url).goto().await?;
        
        // 等待页面加载完成
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        
        let screenshot_path = format!("screenshot_{}.png", 
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs());
        
        browser.close().await?;
        
        Ok(screenshot_path)
    }

    /// 生成设计系统
    pub fn generate_design_system(&self, analyses: &[VisualAnalysis]) -> DesignSystem {
        let mut design_system = DesignSystem::default();
        
        // 合并颜色方案
        let mut color_frequencies: HashMap<String, (Color, usize)> = HashMap::new();
        for analysis in analyses {
            for (color, freq) in &analysis.color_scheme.all_colors {
                let key = color.to_hex();
                let entry = color_frequencies.entry(key).or_insert((color.clone(), 0));
                entry.1 += *freq as usize;
            }
        }
        
        // 选择最常见的颜色
        let mut sorted_colors: Vec<_> = color_frequencies.values().collect();
        sorted_colors.sort_by(|a, b| b.1.cmp(&a.1));
        
        if !sorted_colors.is_empty() {
            design_system.primary_color = Some(sorted_colors[0].0.clone());
        }
        if sorted_colors.len() >= 2 {
            design_system.secondary_color = Some(sorted_colors[1].0.clone());
        }
        
        // 分析间距系统
        let mut all_gaps = Vec::new();
        for analysis in analyses {
            all_gaps.extend(&analysis.spacing.common_gaps);
        }
        
        // 找出最常见的间距值
        let mut gap_counts: HashMap<u32, usize> = HashMap::new();
        for gap in all_gaps {
            *gap_counts.entry(gap).or_insert(0) += 1;
        }
        
        let mut sorted_gaps: Vec<_> = gap_counts.iter().collect();
        sorted_gaps.sort_by(|a, b| b.1.cmp(a.1));
        
        if let Some((base, _)) = sorted_gaps.first() {
            design_system.spacing_base = Some(**base as u8);
            design_system.spacing_scale = sorted_gaps.iter()
                .take(5)
                .map(|(gap, _)| **gap as u8)
                .collect();
        }
        
        design_system
    }
}

/// 设计系统
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DesignSystem {
    pub name: String,
    pub primary_color: Option<Color>,
    pub secondary_color: Option<Color>,
    pub background_color: Option<Color>,
    pub surface_color: Option<Color>,
    pub text_primary_color: Option<Color>,
    pub text_secondary_color: Option<Color>,
    pub accent_color: Option<Color>,
    pub error_color: Option<Color>,
    pub warning_color: Option<Color>,
    pub success_color: Option<Color>,
    pub spacing_base: Option<u8>,
    pub spacing_scale: Vec<u8>,
    pub border_radius_base: Option<u8>,
    pub border_radius_scale: Vec<u8>,
    pub shadow_sm: Option<Shadow>,
    pub shadow_md: Option<Shadow>,
    pub shadow_lg: Option<Shadow>,
    pub font_family_primary: Option<String>,
    pub font_family_secondary: Option<String>,
    pub font_sizes: Vec<u8>,
    pub font_weights: Vec<u16>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_to_hex() {
        let color = Color { r: 255, g: 128, b: 64, a: 255 };
        assert_eq!(color.to_hex(), "#ff8040");
        
        let color_with_alpha = Color { r: 255, g: 128, b: 64, a: 128 };
        assert_eq!(color_with_alpha.to_hex(), "#ff804080");
    }

    #[test]
    fn test_color_to_rgb() {
        let color = Color { r: 255, g: 128, b: 64, a: 255 };
        assert_eq!(color.to_rgb(), "rgb(255, 128, 64)");
    }

    #[test]
    fn test_color_to_rgba() {
        let color = Color { r: 255, g: 128, b: 64, a: 128 };
        assert_eq!(color.to_rgba(), "rgba(255, 128, 64, 0.5019608)");
    }

    #[test]
    fn test_color_luminance() {
        let white = Color { r: 255, g: 255, b: 255, a: 255 };
        let black = Color { r: 0, g: 0, b: 0, a: 255 };
        
        assert!(white.luminance() > 0.9);
        assert!(black.luminance() < 0.1);
    }

    #[test]
    fn test_color_is_dark() {
        let dark = Color { r: 30, g: 30, b: 30, a: 255 };
        let light = Color { r: 200, g: 200, b: 200, a: 255 };
        
        assert!(dark.is_dark());
        assert!(!light.is_dark());
    }

    #[test]
    fn test_bounding_box_center() {
        let bbox = BoundingBox { x: 100, y: 100, width: 200, height: 100 };
        assert_eq!(bbox.center(), (200, 150));
    }

    #[test]
    fn test_bounding_box_area() {
        let bbox = BoundingBox { x: 0, y: 0, width: 100, height: 50 };
        assert_eq!(bbox.area(), 5000);
    }

    #[test]
    fn test_bounding_box_intersects() {
        let bbox1 = BoundingBox { x: 0, y: 0, width: 100, height: 100 };
        let bbox2 = BoundingBox { x: 50, y: 50, width: 100, height: 100 };
        let bbox3 = BoundingBox { x: 200, y: 200, width: 100, height: 100 };
        
        assert!(bbox1.intersects(&bbox2));
        assert!(!bbox1.intersects(&bbox3));
    }

    #[test]
    fn test_bounding_box_contains() {
        let parent = BoundingBox { x: 0, y: 0, width: 200, height: 200 };
        let child = BoundingBox { x: 50, y: 50, width: 100, height: 100 };
        let outside = BoundingBox { x: 300, y: 300, width: 50, height: 50 };
        
        assert!(parent.contains(&child));
        assert!(!parent.contains(&outside));
    }

    #[test]
    fn test_component_type_variants() {
        let types = vec![
            ComponentType::Button,
            ComponentType::Input,
            ComponentType::Card,
            ComponentType::Modal,
        ];
        assert_eq!(types.len(), 4);
    }

    #[test]
    fn test_layout_type_variants() {
        let types = vec![
            LayoutType::SingleColumn,
            LayoutType::Grid,
            LayoutType::FlexRow,
            LayoutType::Masonry,
        ];
        assert_eq!(types.len(), 4);
    }

    #[test]
    fn test_section_type_variants() {
        let types = vec![
            SectionType::Header,
            SectionType::Content,
            SectionType::Footer,
        ];
        assert_eq!(types.len(), 3);
    }

    #[test]
    fn test_visual_learning_config_default() {
        let config = VisualLearningConfig::default();
        assert_eq!(config.screenshot_quality, 90);
        assert!(config.full_page);
        assert!(!config.capture_hover_states);
        assert_eq!(config.component_confidence_threshold, 0.7);
    }

    #[test]
    fn test_design_system_default() {
        let ds = DesignSystem::default();
        assert!(ds.primary_color.is_none());
        assert!(ds.spacing_scale.is_empty());
    }

    #[test]
    fn test_visual_component_creation() {
        let component = VisualComponent {
            id: "btn-1".to_string(),
            component_type: ComponentType::Button,
            bounding_box: BoundingBox { x: 10, y: 20, width: 100, height: 40 },
            confidence: 0.95,
            visual_style: VisualStyle {
                background_color: Some(Color { r: 0, g: 120, b: 255, a: 255 }),
                text_color: None,
                border_color: None,
                border_width: 0,
                border_radius: 4,
                shadow: None,
                opacity: 1.0,
            },
            semantic_label: Some("Submit button".to_string()),
            children: vec![],
            parent: None,
        };
        assert_eq!(component.id, "btn-1");
        assert_eq!(component.confidence, 0.95);
    }
}
