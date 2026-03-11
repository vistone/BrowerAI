//! BrowerAI 真正的双沙盒架构
//!
//! 沙盒1: 标准渲染引擎 - 像 Chrome 一样完整解析网站
//! 沙盒2: AI 学习引擎 - 理解意图、学习样式、生成功能映射
//!
//! 核心理念: 保功能、换体验
//!
//! 新架构 - 真正的AI学习:
//! 1. 组件提取: 从HTML/CSS中识别UI组件（按钮、表单、导航等）
//! 2. JS理解: 将JS代码转换为功能意图
//! 3. 重组生成: 基于学习到的组件和意图，生成全新结构的网站

pub mod common;
pub mod component_extractor;
pub mod css_parser;
pub mod function_generator;
pub mod generator;
pub mod js_parser;
pub mod js_understander;
pub mod sandbox1_standard;
pub mod sandbox2_learning;
pub mod smart_generator;
pub mod style_transform;

// 重新导出主要类型
pub use common::{
    ColorScheme, FunctionMapping, LayoutPattern, StyleSystem, TypographySystem, WebsiteResources,
};
pub use component_extractor::{
    ButtonComponent, CardComponent, ComponentExtractor, ComponentLibrary, FormComponent,
    LayoutComponent, NavComponent,
};
pub use css_parser::{CssParser, ParsedCss};
pub use function_generator::{
    generate_js_file, FunctionGenerator, GeneratedFunctions, TargetFramework,
};
pub use generator::{
    ComponentNode, ComponentTree, GeneratedWebsite, GenerationConfig, WebsiteGenerator, WebsiteType,
};
pub use js_parser::{ApiCall, EventHandler, Function, JsParser, ParsedJs, Variable};
pub use js_understander::{
    BehaviorType, FunctionIntents, InteractionIntent, JsUnderstander, TriggerType,
};
pub use sandbox1_standard::{DomNode, JsFunction, RenderedPage, StandardSandbox};
pub use sandbox2_learning::{FunctionExtraction, LearnedWebsite, LearningSandbox, WebsiteIntent};
pub use smart_generator::{PageSection, PageStructure, SmartGeneratedWebsite, SmartGenerator};
pub use style_transform::{generate_css, StyleTransformer, TransformConfig, TransformType};

use anyhow::Result;

/// 双沙盒引擎 - 协调标准渲染和 AI 学习
pub struct DualSandboxEngine {
    /// 沙盒1: 标准渲染
    standard: StandardSandbox,
    /// 沙盒2: AI 学习
    learning: LearningSandbox,
}

impl DualSandboxEngine {
    /// 创建双沙盒引擎
    pub fn new() -> Result<Self> {
        Ok(Self {
            standard: StandardSandbox::new(),
            learning: LearningSandbox::new(),
        })
    }

    /// 处理网站 - 真正的AI学习流程
    ///
    /// 流程:
    /// 1. 沙盒1: 标准渲染 - 获取完整网站资源
    /// 2. 沙盒2: AI 学习 - 理解意图、提取样式、分析功能
    /// 3. 组件提取 - 识别UI组件（按钮、表单、导航等）
    /// 4. JS理解 - 将JS转换为功能意图
    /// 5. 重组生成 - 基于学习到的知识，生成全新结构的网站
    pub async fn process_website(&self, url: &str) -> Result<ProcessedWebsite> {
        log::info!("╔══════════════════════════════════════════════════════════════╗");
        log::info!("║  双沙盒处理开始 - 真正的AI学习                               ║");
        log::info!("╚══════════════════════════════════════════════════════════════╝");
        log::info!("🌐 目标网站: {}", url);

        // ===== 沙盒1: 标准渲染 =====
        log::info!("\n📦 [沙盒1] 标准渲染 - 获取完整网站...");
        let rendered = self.standard.render(url).await?;
        log::info!("   ✓ HTML: {} 字节", rendered.html.len());
        log::info!("   ✓ CSS 文件: {} 个", rendered.css_resources.len());
        log::info!("   ✓ JS 文件: {} 个", rendered.js_resources.len());
        log::info!("   ✓ DOM 节点: {} 个", rendered.dom_tree.node_count());

        // ===== 沙盒2: AI 学习 =====
        log::info!("\n🧠 [沙盒2] AI 学习 - 理解网站...");
        let learned = self.learning.learn(&rendered).await?;
        log::info!("   ✓ 网站意图: {:?}", learned.intent);
        log::info!(
            "   ✓ 颜色方案: {} 种主色",
            learned.styles.colors.primary_colors.len()
        );
        log::info!(
            "   ✓ 字体系统: {} 种字体",
            learned.styles.typography.font_families.len()
        );
        log::info!("   ✓ 功能点: {} 个", learned.functions.user_functions.len());
        log::info!("   ✓ 布局模式: {} 种", learned.layouts.patterns.len());

        // ===== 步骤3: 组件提取 =====
        log::info!("\n🔧 [组件提取] 识别UI组件...");
        let component_extractor = ComponentExtractor::new();
        let all_css = rendered
            .css_resources
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let components = component_extractor.extract(&rendered.html, &all_css);
        log::info!("   ✓ 按钮组件: {} 个", components.buttons.len());
        log::info!("   ✓ 表单组件: {} 个", components.forms.len());
        log::info!("   ✓ 导航组件: {} 个", components.navigations.len());
        log::info!("   ✓ 卡片组件: {} 个", components.cards.len());
        log::info!("   ✓ 布局组件: {} 个", components.layouts.len());

        // ===== 步骤4: JS理解 =====
        log::info!("\n📜 [JS理解] 解析功能意图...");
        let js_understander = JsUnderstander::new();
        let all_js = rendered
            .js_resources
            .iter()
            .map(|j| j.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let intents = js_understander.understand(&all_js);
        log::info!("   ✓ 交互意图: {} 个", intents.interactions.len());
        log::info!("   ✓ API意图: {} 个", intents.api_intents.len());
        log::info!("   ✓ 数据流: {} 个", intents.data_flows.len());
        log::info!("   ✓ 状态管理: {} 个", intents.state_management.len());

        // ===== 步骤5: 智能重组生成 =====
        log::info!("\n🎨 [智能重组生成] 分析结构并生成新网站...");
        let styles_clone = learned.styles.clone();

        // 使用智能生成器，真正分析HTML结构并生成
        let smart_generator =
            SmartGenerator::new(&rendered.html, learned.styles.clone(), intents.clone());
        let generated_site = smart_generator.generate(TransformType::Original);

        log::info!("   ✓ 分析页面结构完成");
        log::info!("   ✓ 页面标题: {}", generated_site.metadata.title);
        log::info!("   ✓ 生成HTML: {} 字节", generated_site.html.len());
        log::info!("   ✓ 生成CSS: {} 字节", generated_site.css.len());
        log::info!("   ✓ 生成JS: {} 字节", generated_site.js.len());

        // 创建变体（使用不同的主题）
        let variants = self
            .generate_smart_variants(&rendered.html, &styles_clone, &intents, 3)
            .await?;
        log::info!("   ✓ 生成 {} 个体验变体", variants.len());

        log::info!("\n✅ 双沙盒处理完成! 真正的AI学习 + 重组生成");

        // 重建learned（因为styles被move了）
        let learned_rebuilt = LearnedWebsite {
            intent: learned.intent,
            styles: styles_clone.clone(),
            functions: learned.functions,
            layouts: learned.layouts,
            resources: learned.resources,
        };

        Ok(ProcessedWebsite {
            original: rendered,
            learned: learned_rebuilt,
            variants,
            generated: Some(generated_site),
            components: None,
            intents: None,
        })
    }

    /// 推断网站类型
    #[allow(dead_code)]
    fn infer_website_type(
        &self,
        intent_type: &crate::sandbox2_learning::WebsiteType,
    ) -> crate::generator::WebsiteType {
        use crate::generator::WebsiteType as GenType;
        use crate::sandbox2_learning::WebsiteType as LearnedType;

        match intent_type {
            LearnedType::LandingPage => GenType::LandingPage,
            LearnedType::Dashboard => GenType::Dashboard,
            LearnedType::Blog => GenType::Blog,
            LearnedType::Ecommerce => GenType::Ecommerce,
            LearnedType::Documentation => GenType::Documentation,
            _ => GenType::Generic,
        }
    }

    /// 生成智能体验变体
    async fn generate_smart_variants(
        &self,
        html: &str,
        styles: &StyleSystem,
        intents: &FunctionIntents,
        count: usize,
    ) -> Result<Vec<WebsiteVariant>> {
        let mut variants = Vec::new();

        for i in 0..count {
            let transform_type = match i % 6 {
                0 => TransformType::Original,
                1 => TransformType::DarkTheme,
                2 => TransformType::HighContrast,
                3 => TransformType::WarmTone,
                4 => TransformType::CoolTone,
                5 => TransformType::Minimal,
                _ => TransformType::Original,
            };

            let variant_name = match transform_type {
                TransformType::Original => "Original",
                TransformType::DarkTheme => "DarkTheme",
                TransformType::HighContrast => "HighContrast",
                TransformType::WarmTone => "WarmTone",
                TransformType::CoolTone => "CoolTone",
                TransformType::Minimal => "Minimal",
                TransformType::Vibrant => "Vibrant",
            };

            // 使用智能生成器生成变体
            let smart_generator = SmartGenerator::new(html, styles.clone(), intents.clone());
            let generated = smart_generator.generate(transform_type);

            variants.push(WebsiteVariant {
                name: format!("{}_{}", variant_name, i + 1),
                styles: styles.clone(),
                function_mappings: Vec::new(),
                html: generated.html,
                css: generated.css,
                js: generated.js,
            });
        }

        Ok(variants)
    }
}

/// 处理后的网站
pub struct ProcessedWebsite {
    /// 原始渲染结果
    pub original: RenderedPage,
    /// 学习结果
    pub learned: LearnedWebsite,
    /// 生成的变体
    pub variants: Vec<WebsiteVariant>,
    /// AI生成的网站（全新结构）
    pub generated: Option<GeneratedWebsite>,
    /// 提取的组件库
    pub components: Option<ComponentLibrary>,
    /// 理解的功能意图
    pub intents: Option<FunctionIntents>,
}

/// 网站变体
pub struct WebsiteVariant {
    /// 变体名称
    pub name: String,
    /// 新样式
    pub styles: StyleSystem,
    /// 功能映射
    pub function_mappings: Vec<FunctionMapping>,
    /// 生成的 HTML
    pub html: String,
    /// 生成的 CSS
    pub css: String,
    /// 生成的 JS
    pub js: String,
}
