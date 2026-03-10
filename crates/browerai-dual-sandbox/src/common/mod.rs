//! 双沙盒共享类型

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 网站资源集合
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebsiteResources {
    /// HTML 内容
    pub html: String,
    /// CSS 文件内容 (URL -> Content)
    pub css_files: HashMap<String, String>,
    /// JS 文件内容 (URL -> Content)
    pub js_files: HashMap<String, String>,
    /// 图片资源
    pub images: Vec<String>,
    /// 字体资源
    pub fonts: Vec<String>,
}

/// 样式系统 - 从网站提取的完整样式信息
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StyleSystem {
    /// 颜色方案
    pub colors: ColorScheme,
    /// 字体系统
    pub typography: TypographySystem,
    /// 间距系统
    pub spacing: SpacingSystem,
    /// 布局系统
    pub layouts: LayoutSystem,
    /// 原始 CSS 规则
    pub raw_css_rules: Vec<CssRule>,
}

/// 颜色方案
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColorScheme {
    /// 主色调
    pub primary_colors: Vec<Color>,
    /// 辅助色
    pub secondary_colors: Vec<Color>,
    /// 背景色
    pub background_colors: Vec<Color>,
    /// 文字色
    pub text_colors: Vec<Color>,
    /// 强调色
    pub accent_colors: Vec<Color>,
}

/// 颜色
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Color {
    /// 原始值 (hex, rgb, rgba, hsl)
    pub raw: String,
    /// 标准化 hex
    pub hex: String,
    /// RGB 值
    pub rgb: (u8, u8, u8),
    /// 透明度
    pub alpha: f32,
    /// 使用次数
    pub usage_count: usize,
    /// 使用场景 (background, text, border, etc.)
    pub usage_context: Vec<String>,
}

/// 字体系统
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypographySystem {
    /// 字体家族
    pub font_families: Vec<FontFamily>,
    /// 字体大小层级
    pub font_sizes: Vec<FontSize>,
    /// 行高
    pub line_heights: Vec<f32>,
    /// 字重
    pub font_weights: Vec<u16>,
}

/// 字体家族
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontFamily {
    /// 名称
    pub name: String,
    /// 备用字体
    pub fallbacks: Vec<String>,
    /// 使用次数
    pub usage_count: usize,
    /// 来源 (google fonts, system, custom)
    pub source: String,
}

/// 字体大小
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontSize {
    /// 值 (px, rem, em)
    pub value: String,
    /// 像素值
    pub pixels: f32,
    /// 使用场景 (h1, h2, body, etc.)
    pub context: String,
}

/// 间距系统
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpacingSystem {
    /// padding 值
    pub paddings: Vec<String>,
    /// margin 值
    pub margins: Vec<String>,
    /// gap 值
    pub gaps: Vec<String>,
    /// 基础单位
    pub base_unit: f32,
}

/// 布局系统
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayoutSystem {
    /// 布局模式
    pub patterns: Vec<LayoutPattern>,
    /// 网格使用
    pub grid_usage: Vec<GridLayout>,
    /// Flex 使用
    pub flex_usage: Vec<FlexLayout>,
    /// 响应式断点
    pub breakpoints: Vec<Breakpoint>,
}

/// 布局模式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutPattern {
    HeaderMainFooter,
    SidebarContent,
    CardGrid,
    ListView,
    SingleColumn,
    Magazine,
    Dashboard,
    LandingPage,
}

/// 网格布局
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridLayout {
    /// 列数
    pub columns: u32,
    /// 行数
    pub rows: u32,
    /// 间距
    pub gap: String,
    /// 选择器
    pub selector: String,
}

/// Flex 布局
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlexLayout {
    /// 方向
    pub direction: String,
    /// 对齐方式
    pub align_items: String,
    /// 分布方式
    pub justify_content: String,
    /// 选择器
    pub selector: String,
}

/// 响应式断点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Breakpoint {
    /// 名称 (mobile, tablet, desktop)
    pub name: String,
    /// 宽度 (px)
    pub width: u32,
    /// 媒体查询
    pub query: String,
}

/// CSS 规则
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CssRule {
    /// 选择器
    pub selector: String,
    /// 声明 (属性 -> 值)
    pub declarations: HashMap<String, String>,
    /// 媒体查询
    pub media_query: Option<String>,
    /// 来源文件
    pub source_file: String,
    /// 优先级
    pub specificity: u32,
}

/// 功能映射
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionMapping {
    /// 原始功能名称
    pub original_name: String,
    /// 新功能名称
    pub new_name: String,
    /// 功能类型 (click, submit, input, etc.)
    pub function_type: String,
    /// 原始选择器
    pub original_selector: String,
    /// 新选择器
    pub new_selector: String,
    /// 原始代码
    pub original_code: String,
    /// 新代码
    pub new_code: String,
    /// 是否保留
    pub preserved: bool,
}
