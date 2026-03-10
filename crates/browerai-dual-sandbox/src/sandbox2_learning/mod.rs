//! 沙盒2: AI 学习引擎
//!
//! 理解网站意图、学习样式、分析功能

use crate::common::*;
use crate::sandbox1_standard::{DomTree, JsResource, RenderedPage};
use anyhow::Result;
use std::collections::HashMap;

/// 学习沙盒
pub struct LearningSandbox;

/// 学习后的网站
#[derive(Debug, Clone)]
pub struct LearnedWebsite {
    /// 网站意图
    pub intent: WebsiteIntent,
    /// 提取的样式
    pub styles: StyleSystem,
    /// 提取的功能
    pub functions: FunctionExtraction,
    /// 提取的布局
    pub layouts: LayoutExtraction,
    /// 资源分析
    pub resources: ResourceAnalysis,
}

/// 网站意图
#[derive(Debug, Clone)]
pub struct WebsiteIntent {
    /// 主要类型
    pub primary_type: WebsiteType,
    /// 置信度
    pub confidence: f32,
    /// 次要类型
    pub secondary_types: Vec<WebsiteType>,
    /// 核心功能
    pub core_features: Vec<String>,
    /// 目标用户
    pub target_audience: String,
}

/// 网站类型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WebsiteType {
    Blog,
    Ecommerce,
    Documentation,
    Portfolio,
    Dashboard,
    LandingPage,
    SocialMedia,
    News,
    Forum,
    Wiki,
    Corporate,
    Personal,
    Unknown,
}

/// 功能提取
#[derive(Debug, Clone, Default)]
pub struct FunctionExtraction {
    /// 用户交互功能
    pub user_functions: Vec<UserFunction>,
    /// 数据流
    pub data_flows: Vec<DataFlow>,
    /// API 调用
    pub api_calls: Vec<ApiCall>,
    /// 功能映射
    pub mappings: Vec<FunctionMapping>,
}

/// 用户功能
#[derive(Debug, Clone)]
pub struct UserFunction {
    /// 功能名称
    pub name: String,
    /// 功能描述
    pub description: String,
    /// 触发方式 (click, submit, input, etc.)
    pub trigger: String,
    /// 关联元素
    pub element_selector: String,
    /// 处理函数
    pub handler: String,
    /// 重要性 (0-1)
    pub importance: f32,
}

/// 数据流
#[derive(Debug, Clone)]
pub struct DataFlow {
    /// 源
    pub source: String,
    /// 目标
    pub target: String,
    /// 数据类型
    pub data_type: String,
}

/// API 调用
#[derive(Debug, Clone)]
pub struct ApiCall {
    /// 端点
    pub endpoint: String,
    /// 方法
    pub method: String,
    /// 用途
    pub purpose: String,
}

/// 布局提取
#[derive(Debug, Clone, Default)]
pub struct LayoutExtraction {
    /// 布局模式
    pub patterns: Vec<LayoutPattern>,
    /// 组件层次
    pub component_tree: ComponentTree,
    /// 视觉层次
    pub visual_hierarchy: Vec<VisualLevel>,
}

/// 组件树
#[derive(Debug, Clone, Default)]
pub struct ComponentTree {
    /// 根组件
    pub root: Box<Component>,
    /// 所有组件
    pub all_components: Vec<Component>,
}

/// 组件
#[derive(Debug, Clone, Default)]
pub struct Component {
    /// 类型
    pub component_type: ComponentType,
    /// 名称
    pub name: String,
    /// 子组件
    pub children: Vec<Component>,
    /// 样式类
    pub style_classes: Vec<String>,
    /// 功能
    pub functionality: Vec<String>,
}

/// 组件类型
#[derive(Debug, Clone, Default)]
pub enum ComponentType {
    #[default]
    Content,
    Header,
    Navigation,
    Hero,
    Sidebar,
    Footer,
    Card,
    List,
    Form,
    Button,
    Input,
    Image,
    Video,
    Table,
    Modal,
    Tooltip,
}

/// 视觉层级
#[derive(Debug, Clone)]
pub struct VisualLevel {
    /// 层级
    pub level: u32,
    /// 元素
    pub elements: Vec<String>,
    /// 重要性
    pub importance: f32,
}

/// 资源分析
#[derive(Debug, Clone, Default)]
pub struct ResourceAnalysis {
    /// 图片分析
    pub images: Vec<ImageAnalysis>,
    /// 字体分析
    pub fonts: Vec<FontAnalysis>,
    /// 性能影响
    pub performance_impact: PerformanceImpact,
}

/// 图片分析
#[derive(Debug, Clone)]
pub struct ImageAnalysis {
    /// URL
    pub url: String,
    /// 用途 (logo, background, content, icon)
    pub usage: String,
    /// 尺寸
    pub dimensions: (u32, u32),
    /// 文件大小
    pub file_size: usize,
}

/// 字体分析
#[derive(Debug, Clone)]
pub struct FontAnalysis {
    /// 家族
    pub family: String,
    /// 用途 (heading, body, code)
    pub usage: String,
    /// 来源
    pub source: String,
}

/// 性能影响
#[derive(Debug, Clone, Default)]
pub struct PerformanceImpact {
    /// 总下载大小
    pub total_download_bytes: usize,
    /// CSS 大小
    pub css_bytes: usize,
    /// JS 大小
    pub js_bytes: usize,
    /// 图片大小
    pub image_bytes: usize,
    /// 估计加载时间
    pub estimated_load_time_ms: u32,
}

impl LearningSandbox {
    /// 创建学习沙盒
    pub fn new() -> Self {
        Self
    }

    /// 学习网站
    pub async fn learn(&self, rendered: &RenderedPage) -> Result<LearnedWebsite> {
        log::info!("🧠 开始学习网站...");

        // 1. 理解意图
        let intent = self.understand_intent(rendered).await?;
        log::info!("   ✓ 意图: {:?} (置信度: {:.0}%)", intent.primary_type, intent.confidence * 100.0);

        // 2. 提取样式
        let styles = self.extract_styles(rendered).await?;
        log::info!("   ✓ 提取 {} 种颜色, {} 种字体", 
            styles.colors.primary_colors.len(),
            styles.typography.font_families.len()
        );

        // 3. 提取功能
        let functions = self.extract_functions(rendered).await?;
        log::info!("   ✓ 发现 {} 个用户功能", functions.user_functions.len());

        // 4. 提取布局
        let layouts = self.extract_layouts(rendered).await?;
        log::info!("   ✓ 识别 {} 种布局模式", layouts.patterns.len());

        // 5. 分析资源
        let resources = self.analyze_resources(rendered).await?;
        log::info!("   ✓ 分析资源: {} 图片, {} 字体", 
            resources.images.len(),
            resources.fonts.len()
        );

        Ok(LearnedWebsite {
            intent,
            styles,
            functions,
            layouts,
            resources,
        })
    }

    /// 理解网站意图
    async fn understand_intent(&self, rendered: &RenderedPage) -> Result<WebsiteIntent> {
        let html = &rendered.html;
        
        // 基于特征检测网站类型
        let mut type_scores: HashMap<WebsiteType, f32> = HashMap::new();

        // 博客特征
        if html.contains("<article") || html.contains("class=\"post\"") || html.contains("class=\"blog\"") {
            let score = type_scores.entry(WebsiteType::Blog).or_insert(0.0);
            *score += 0.5;
        }

        // 电商特征
        if html.contains("class=\"product\"") || html.contains("class=\"cart\"") || html.contains("class=\"price\"") {
            let score = type_scores.entry(WebsiteType::Ecommerce).or_insert(0.0);
            *score += 0.8;
        }

        // 文档特征
        if html.contains("class=\"docs\"") || html.contains("class=\"documentation\"") || html.contains("<code") {
            let score = type_scores.entry(WebsiteType::Documentation).or_insert(0.0);
            *score += 0.6;
        }

        // 仪表盘特征
        if html.contains("class=\"dashboard\"") || html.contains("class=\"chart\"") || html.contains("class=\"widget\"") {
            let score = type_scores.entry(WebsiteType::Dashboard).or_insert(0.0);
            *score += 0.7;
        }

        // 落地页特征
        if html.contains("class=\"hero\"") || html.contains("class=\"cta\"") || html.contains("class=\"landing\"") {
            let score = type_scores.entry(WebsiteType::LandingPage).or_insert(0.0);
            *score += 0.6;
        }

        // 选择最高分的类型
        let (primary_type, confidence) = type_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(t, s)| (t.clone(), *s))
            .unwrap_or((WebsiteType::Unknown, 0.0));

        // 提取核心功能
        let core_features = self.extract_core_features(html);

        Ok(WebsiteIntent {
            primary_type,
            confidence: confidence.max(0.5),
            secondary_types: Vec::new(),
            core_features,
            target_audience: String::new(),
        })
    }

    /// 提取核心功能
    fn extract_core_features(&self, html: &str) -> Vec<String> {
        let mut features = Vec::new();

        if html.contains("<form") {
            features.push("form_submission".to_string());
        }
        if html.contains("<input") {
            features.push("user_input".to_string());
        }
        if html.contains("<button") || html.contains("onclick") {
            features.push("interactive_buttons".to_string());
        }
        if html.contains("<nav") || html.contains("class=\"nav\"") {
            features.push("navigation".to_string());
        }
        if html.contains("<search") || html.contains("type=\"search\"") {
            features.push("search".to_string());
        }

        features
    }

    /// 提取样式
    async fn extract_styles(&self, rendered: &RenderedPage) -> Result<StyleSystem> {
        let mut styles = StyleSystem::default();

        // 从所有 CSS 资源提取
        for css_resource in &rendered.css_resources {
            self.extract_colors_from_css(&css_resource.content, &mut styles.colors);
            self.extract_typography_from_css(&css_resource.content, &mut styles.typography);
            self.extract_spacing_from_css(&css_resource.content, &mut styles.spacing);
        }

        // 从 DOM 计算样式
        self.compute_styles_from_dom(&rendered.dom_tree, &mut styles);

        Ok(styles)
    }

    /// 从 CSS 提取颜色（带去重和限制）
    fn extract_colors_from_css(&self, css: &str, colors: &mut ColorScheme) {
        use std::collections::HashSet;
        
        // 提取 hex 颜色
        let hex_re = regex::Regex::new(r"#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})").unwrap();
        
        // 使用HashSet去重
        let mut seen_primary: HashSet<String> = HashSet::new();
        let mut seen_background: HashSet<String> = HashSet::new();
        let mut seen_text: HashSet<String> = HashSet::new();
        
        // 限制每种类型最多提取的颜色数
        const MAX_COLORS_PER_TYPE: usize = 50;
        
        for cap in hex_re.captures_iter(css) {
            let hex = cap[0].to_string().to_lowercase();
            // 统一转换为6位hex
            let hex = if hex.len() == 4 {
                // #abc -> #aabbcc
                format!("#{}{}{}{}{}{}", 
                    &hex[1..2], &hex[1..2],
                    &hex[2..3], &hex[2..3],
                    &hex[3..4], &hex[3..4])
            } else {
                hex
            };
            
            // 分析颜色用途
            let context = self.analyze_color_context(css, cap.get(0).unwrap().start());
            let context_str = context.clone();
            
            let color = Color {
                raw: hex.clone(),
                hex: hex.clone(),
                rgb: self.hex_to_rgb(&hex),
                alpha: 1.0,
                usage_count: 1,
                usage_context: vec![context],
            };

            // 分类颜色（带去重和限制）
            if context_str.contains("background") {
                if !seen_background.contains(&hex) && colors.background_colors.len() < MAX_COLORS_PER_TYPE {
                    seen_background.insert(hex);
                    colors.background_colors.push(color);
                }
            } else if context_str.contains("color") && !context_str.contains("background") {
                if !seen_text.contains(&hex) && colors.text_colors.len() < MAX_COLORS_PER_TYPE {
                    seen_text.insert(hex);
                    colors.text_colors.push(color);
                }
            } else {
                if !seen_primary.contains(&hex) && colors.primary_colors.len() < MAX_COLORS_PER_TYPE {
                    seen_primary.insert(hex);
                    colors.primary_colors.push(color);
                }
            }
        }
    }

    /// 分析颜色上下文
    fn analyze_color_context(&self, css: &str, pos: usize) -> String {
        // 获取颜色前后的上下文
        let start = pos.saturating_sub(100);
        let context = &css[start..pos.min(css.len())];
        
        if context.contains("background") {
            "background".to_string()
        } else if context.contains("color") {
            "text".to_string()
        } else if context.contains("border") {
            "border".to_string()
        } else {
            "unknown".to_string()
        }
    }

    /// hex 转 rgb
    fn hex_to_rgb(&self, hex: &str) -> (u8, u8, u8) {
        let hex = hex.trim_start_matches('#');
        
        if hex.len() == 6 {
            // #RRGGBB
            let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
            let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
            let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
            (r, g, b)
        } else if hex.len() == 3 {
            // #RGB -> #RRGGBB
            let r = u8::from_str_radix(&hex[0..1].repeat(2), 16).unwrap_or(0);
            let g = u8::from_str_radix(&hex[1..2].repeat(2), 16).unwrap_or(0);
            let b = u8::from_str_radix(&hex[2..3].repeat(2), 16).unwrap_or(0);
            (r, g, b)
        } else {
            (0, 0, 0)
        }
    }

    /// 从 CSS 提取字体
    fn extract_typography_from_css(&self, css: &str, typography: &mut TypographySystem) {
        // 提取 font-family
        let font_re = regex::Regex::new(r"font-family\s*:\s*([^;]+)").unwrap();
        for cap in font_re.captures_iter(css) {
            let families: Vec<String> = cap[1]
                .split(',')
                .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
                .collect();

            if let Some(first) = families.first() {
                typography.font_families.push(FontFamily {
                    name: first.clone(),
                    fallbacks: families[1..].to_vec(),
                    usage_count: 1,
                    source: "css".to_string(),
                });
            }
        }

        // 提取 font-size
        let size_re = regex::Regex::new(r"font-size\s*:\s*([^;]+)").unwrap();
        for cap in size_re.captures_iter(css) {
            typography.font_sizes.push(FontSize {
                value: cap[1].trim().to_string(),
                pixels: self.parse_font_size(&cap[1]),
                context: "body".to_string(),
            });
        }
    }

    /// 解析字体大小
    fn parse_font_size(&self, size: &str) -> f32 {
        let size = size.trim();
        if size.ends_with("px") {
            size.trim_end_matches("px").parse().unwrap_or(16.0)
        } else if size.ends_with("rem") {
            size.trim_end_matches("rem").parse::<f32>().unwrap_or(1.0) * 16.0
        } else {
            16.0
        }
    }

    /// 从 CSS 提取间距
    fn extract_spacing_from_css(&self, css: &str, spacing: &mut SpacingSystem) {
        // 提取 padding
        let padding_re = regex::Regex::new(r"padding\s*:\s*([^;]+)").unwrap();
        for cap in padding_re.captures_iter(css) {
            spacing.paddings.push(cap[1].trim().to_string());
        }

        // 提取 margin
        let margin_re = regex::Regex::new(r"margin\s*:\s*([^;]+)").unwrap();
        for cap in margin_re.captures_iter(css) {
            spacing.margins.push(cap[1].trim().to_string());
        }
    }

    /// 从 DOM 计算样式
    fn compute_styles_from_dom(&self, _dom: &DomTree, _styles: &mut StyleSystem) {
        // 遍历 DOM 树，计算每个节点的实际样式
        // 简化实现
    }

    /// 提取功能
    async fn extract_functions(&self, rendered: &RenderedPage) -> Result<FunctionExtraction> {
        let mut functions = FunctionExtraction::default();

        // 从 JS 提取
        for js_resource in &rendered.js_resources {
            self.extract_functions_from_js(js_resource, &mut functions);
        }

        // 从 HTML 提取事件处理器
        self.extract_event_handlers(&rendered.html, &mut functions);

        Ok(functions)
    }

    /// 从 JS 提取功能
    fn extract_functions_from_js(&self, js: &JsResource, functions: &mut FunctionExtraction) {
        for func in &js.functions {
            functions.user_functions.push(UserFunction {
                name: func.name.clone(),
                description: String::new(),
                trigger: if func.is_event_handler { "event".to_string() } else { "call".to_string() },
                element_selector: func.attached_elements.join(", "),
                handler: func.body.clone(),
                importance: 0.5,
            });
        }
    }

    /// 提取事件处理器
    fn extract_event_handlers(&self, html: &str, functions: &mut FunctionExtraction) {
        // 提取 onclick="..."
        let onclick_re = regex::Regex::new(r#"onclick=["']([^"']+)["']"#).unwrap();
        for cap in onclick_re.captures_iter(html) {
            functions.user_functions.push(UserFunction {
                name: "onclick_handler".to_string(),
                description: "Click event handler".to_string(),
                trigger: "click".to_string(),
                element_selector: String::new(),
                handler: cap[1].to_string(),
                importance: 0.7,
            });
        }
    }

    /// 提取布局
    async fn extract_layouts(&self, rendered: &RenderedPage) -> Result<LayoutExtraction> {
        let mut layouts = LayoutExtraction::default();

        // 分析 DOM 结构识别布局模式
        self.analyze_layout_patterns(&rendered.dom_tree, &mut layouts);

        // 从 CSS 提取 grid/flex 布局
        for css_resource in &rendered.css_resources {
            self.extract_grid_flex_layouts(&css_resource.content, &mut layouts);
        }

        Ok(layouts)
    }

    /// 分析布局模式
    fn analyze_layout_patterns(&self, dom: &DomTree, layouts: &mut LayoutExtraction) {
        // 检测 header-main-footer 模式
        let has_header = dom.query_selector("header").len() > 0;
        let has_footer = dom.query_selector("footer").len() > 0;
        let has_main = dom.query_selector("main").len() > 0;

        if has_header && has_footer && has_main {
            layouts.patterns.push(LayoutPattern::HeaderMainFooter);
        }

        // 检测 sidebar-content 模式
        let has_aside = dom.query_selector("aside").len() > 0;
        if has_aside && has_main {
            layouts.patterns.push(LayoutPattern::SidebarContent);
        }
    }

    /// 提取 grid/flex 布局
    fn extract_grid_flex_layouts(&self, css: &str, layouts: &mut LayoutExtraction) {
        // 检测 grid
        if css.contains("display: grid") || css.contains("display:grid") {
            layouts.patterns.push(LayoutPattern::CardGrid);
        }

        // 检测 flex
        if css.contains("display: flex") || css.contains("display:flex") {
            // 可能是各种 flex 布局
        }
    }

    /// 分析资源
    async fn analyze_resources(&self, rendered: &RenderedPage) -> Result<ResourceAnalysis> {
        let mut resources = ResourceAnalysis::default();

        // 计算性能影响
        resources.performance_impact = PerformanceImpact {
            total_download_bytes: rendered.stats.bytes_downloaded,
            css_bytes: rendered.css_resources.iter().map(|c| c.content.len()).sum(),
            js_bytes: rendered.js_resources.iter().map(|j| j.content.len()).sum(),
            image_bytes: 0,
            estimated_load_time_ms: (rendered.stats.bytes_downloaded / 10000) as u32,
        };

        Ok(resources)
    }

    /// 生成新样式
    pub fn generate_new_styles(&self, original: &StyleSystem, variant_index: usize) -> StyleSystem {
        // 基于原始样式生成变体
        let mut new_styles = original.clone();

        // 应用变换 (颜色偏移、字体替换等)
        match variant_index % 4 {
            0 => {
                // 变体1: 保持原样
            }
            1 => {
                // 变体2: 深色主题
                new_styles.colors = self.transform_to_dark_theme(&original.colors);
            }
            2 => {
                // 变体3: 高对比
                new_styles.colors = self.transform_to_high_contrast(&original.colors);
            }
            3 => {
                // 变体4: 暖色调
                new_styles.colors = self.transform_to_warm_theme(&original.colors);
            }
            _ => {}
        }

        new_styles
    }

    /// 转换为深色主题
    fn transform_to_dark_theme(&self, colors: &ColorScheme) -> ColorScheme {
        // 简化实现
        colors.clone()
    }

    /// 转换为高对比
    fn transform_to_high_contrast(&self, colors: &ColorScheme) -> ColorScheme {
        // 简化实现
        colors.clone()
    }

    /// 转换为暖色调
    fn transform_to_warm_theme(&self, colors: &ColorScheme) -> ColorScheme {
        // 简化实现
        colors.clone()
    }
}
