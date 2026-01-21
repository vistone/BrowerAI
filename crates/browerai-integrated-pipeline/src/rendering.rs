//! 渲染模块 - 个性化渲染
//!
//! 提供完整的网站渲染功能，包括：
//! - 模板渲染
//! - 组件生成
//! - 样式注入
//! - 布局算法

use anyhow::{Context, Result};
use browerai_intelligent_rendering::{ComplianceLevel, TargetStyle};
use browerai_learning::high_fidelity_generator::HighFidelityGenerator;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub target_style: RenderTargetStyle,
    pub output_format: OutputFormat,
    pub optimize_for: OptimizationTarget,
    pub include_source_maps: bool,
    pub minify_output: bool,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub enum RenderTargetStyle {
    Enterprise {
        brand_color: String,
        typography: String,
        border_radius: f64,
        shadow_style: ShadowStyle,
    },
    Government {
        compliance_level: ComplianceLevel,
        accessibility_standard: AccessibilityStandard,
        security_level: SecurityLevel,
    },
    Custom {
        css_variables: HashMap<String, String>,
        component_library: String,
        design_token_path: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ShadowStyle {
    None,
    Subtle,
    Medium,
    Prominent,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AccessibilityStandard {
    WCAG2A,
    WCAG2AA,
    WCAG2AAA,
    Section508,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SecurityLevel {
    Standard,
    Enhanced,
    Maximum,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutputFormat {
    Html,
    SinglePage,
    MultiPage,
    ComponentLibrary,
    Pwa,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationTarget {
    Performance,
    Accessibility,
    SEO,
    BundleSize,
    Balanced,
}

#[derive(Debug, Clone)]
pub struct RenderResult {
    pub files: Vec<RenderedFile>,
    pub assets: Vec<AssetInfo>,
    pub metrics: RenderMetrics,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RenderedFile {
    pub path: String,
    pub content: String,
    pub file_type: FileType,
    pub size_bytes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FileType {
    Html,
    Css,
    JavaScript,
    Json,
    Image,
    Font,
    Other,
}

#[derive(Debug, Clone)]
pub struct AssetInfo {
    pub url: String,
    pub local_path: Option<String>,
    pub content_type: String,
    pub size_bytes: usize,
    pub inlined: bool,
}

#[derive(Debug, Clone)]
pub struct RenderMetrics {
    pub render_time_ms: u64,
    pub bundle_size_bytes: usize,
    pub first_contentful_paint_ms: Option<u64>,
    pub accessibility_score: Option<f64>,
    pub seo_score: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct LayoutSpec {
    pub grid_columns: u32,
    pub grid_gap: u32,
    pub container_max_width: u32,
    pub breakpoints: Vec<Breakpoint>,
    pub responsive_order: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Breakpoint {
    pub name: String,
    pub min_width: u32,
    pub max_width: Option<u32>,
    pub columns: u32,
    pub gap: u32,
}

pub struct RenderingModule {
    config: RenderConfig,
    generator: HighFidelityGenerator,
    templates: Arc<Mutex<TemplateStore>>,
    layout_engine: LayoutEngine,
}

struct TemplateStore {
    html_templates: HashMap<String, String>,
    css_templates: HashMap<String, String>,
    js_templates: HashMap<String, String>,
}

impl TemplateStore {
    fn new() -> Self {
        Self {
            html_templates: Self::default_html_templates(),
            css_templates: Self::default_css_templates(),
            js_templates: Self::default_js_templates(),
        }
    }

    fn default_html_templates() -> HashMap<String, String> {
        let mut templates = HashMap::new();
        templates.insert(
            "base".to_string(),
            include_str!("templates/base.html").to_string(),
        );
        templates.insert(
            "component".to_string(),
            include_str!("templates/component.html").to_string(),
        );
        templates.insert(
            "page".to_string(),
            include_str!("templates/page.html").to_string(),
        );
        templates
    }

    fn default_css_templates() -> HashMap<String, String> {
        let mut templates = HashMap::new();
        templates.insert(
            "variables".to_string(),
            include_str!("templates/variables.css").to_string(),
        );
        templates.insert(
            "reset".to_string(),
            include_str!("templates/reset.css").to_string(),
        );
        templates.insert(
            "utility".to_string(),
            include_str!("templates/utility.css").to_string(),
        );
        templates
    }

    fn default_js_templates() -> HashMap<String, String> {
        let mut templates = HashMap::new();
        templates.insert(
            "init".to_string(),
            include_str!("templates/init.js").to_string(),
        );
        templates.insert(
            "component".to_string(),
            include_str!("templates/component.js").to_string(),
        );
        templates
    }
}

impl RenderingModule {
    pub fn new() -> Self {
        Self::with_config(RenderConfig::default())
    }

    pub fn with_config(config: RenderConfig) -> Self {
        Self {
            config: config.clone(),
            generator: HighFidelityGenerator::new(),
            templates: Arc::new(Mutex::new(TemplateStore::new())),
            layout_engine: LayoutEngine::new(config.target_style.clone()),
        }
    }

    pub fn set_template(&mut self, template_type: &str, name: &str, content: String) {
        let mut store = self.templates.lock().unwrap();
        match template_type {
            "html" => store.html_templates.insert(name.to_string(), content),
            "css" => store.css_templates.insert(name.to_string(), content),
            "js" => store.js_templates.insert(name.to_string(), content),
            _ => None,
        };
    }

    pub async fn render(
        &self,
        learned_model: &browerai_learning::high_fidelity_generator::WebsiteAnalysisComplete,
    ) -> Result<RenderResult> {
        let start_time = std::time::Instant::now();

        let template_store = self.templates.lock().unwrap();

        let html = self.generator.generate_complete_page(learned_model).await?;

        let css = self.generate_styles(learned_model, &template_store).await?;

        let js = self
            .generate_scripts(learned_model, &template_store)
            .await?;

        let files = vec![
            RenderedFile {
                path: "index.html".to_string(),
                content: html,
                file_type: FileType::Html,
                size_bytes: 0,
            },
            RenderedFile {
                path: "styles/main.css".to_string(),
                content: css,
                file_type: FileType::Css,
                size_bytes: 0,
            },
            RenderedFile {
                path: "scripts/main.js".to_string(),
                content: js,
                file_type: FileType::JavaScript,
                size_bytes: 0,
            },
        ];

        let render_time = start_time.elapsed().as_millis() as u64;

        let metrics = RenderMetrics {
            render_time_ms: render_time,
            bundle_size_bytes: files.iter().map(|f| f.content.len()).sum(),
            first_contentful_paint_ms: Some(render_time + 100),
            accessibility_score: Some(self.calculate_accessibility_score(&files)),
            seo_score: Some(self.calculate_seo_score(&files)),
        };

        Ok(RenderResult {
            files,
            assets: Vec::new(),
            metrics,
            warnings: Vec::new(),
        })
    }

    async fn generate_styles(
        &self,
        model: &browerai_learning::high_fidelity_generator::WebsiteAnalysisComplete,
        template_store: &TemplateStore,
    ) -> Result<String> {
        let mut css = String::new();

        for (_, template) in &template_store.css_templates {
            css.push_str(&template);
            css.push('\n');
        }

        css.push_str(&format!(":root {{\n"));
        css.push_str(&format!("  --primary-color: {};\n", self.get_brand_color()));
        css.push_str("}\n");

        Ok(css)
    }

    async fn generate_scripts(
        &self,
        model: &browerai_learning::high_fidelity_generator::WebsiteAnalysisComplete,
        template_store: &TemplateStore,
    ) -> Result<String> {
        let mut js = String::new();

        for (_, template) in &template_store.js_templates {
            js.push_str(&template);
            js.push('\n');
        }

        Ok(js)
    }

    fn get_brand_color(&self) -> String {
        match &self.config.target_style {
            RenderTargetStyle::Enterprise { brand_color, .. } => brand_color.clone(),
            RenderTargetStyle::Government { .. } => "#0052CC".to_string(),
            RenderTargetStyle::Custom { css_variables, .. } => css_variables
                .get("brand-color")
                .cloned()
                .unwrap_or_else(|| "#0052CC".to_string()),
        }
    }

    fn calculate_accessibility_score(&self, files: &[RenderedFile]) -> f64 {
        let mut score = 100.0;

        for file in files {
            if file.file_type == FileType::Html {
                if !file.content.contains("<html") {
                    score -= 10.0;
                }
                if !file.content.contains("<title>") {
                    score -= 5.0;
                }
                if !file.content.contains("alt=\"") {
                    score -= 5.0;
                }
                if !file.content.contains("aria-") {
                    score -= 5.0;
                }
                if !file.content.contains("<main") {
                    score -= 5.0;
                }
                if !file.content.contains("<nav") {
                    score -= 3.0;
                }
            }
        }

        score.max(0.0)
    }

    fn calculate_seo_score(&self, files: &[RenderedFile]) -> f64 {
        let mut score = 100.0;

        for file in files {
            if file.file_type == FileType::Html {
                if !file.content.contains("<meta name=\"description\"") {
                    score -= 10.0;
                }
                if !file.content.contains("<h1") {
                    score -= 10.0;
                }
                if !file.content.contains("<link rel=\"canonical\"") {
                    score -= 5.0;
                }
                if !file.content.contains("<meta name=\"robots\"") {
                    score -= 3.0;
                }
                if !file.content.contains("og:") {
                    score -= 5.0;
                }
                if !file.content.contains("twitter:") {
                    score -= 2.0;
                }
            }
        }

        score.max(0.0)
    }
}

struct LayoutEngine {
    config: RenderTargetStyle,
    breakpoints: Vec<Breakpoint>,
}

impl LayoutEngine {
    fn new(config: RenderTargetStyle) -> Self {
        let breakpoints = vec![
            Breakpoint {
                name: "mobile".to_string(),
                min_width: 0,
                max_width: Some(767),
                columns: 4,
                gap: 16,
            },
            Breakpoint {
                name: "tablet".to_string(),
                min_width: 768,
                max_width: Some(1023),
                columns: 8,
                gap: 20,
            },
            Breakpoint {
                name: "desktop".to_string(),
                min_width: 1024,
                max_width: None,
                columns: 12,
                gap: 24,
            },
        ];

        Self {
            config,
            breakpoints,
        }
    }

    fn generate_grid_layout(&self, columns: u32, gap: u32) -> String {
        format!(
            r#"
.container {{
  display: grid;
  grid-template-columns: repeat({}, 1fr);
  gap: {}px;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 16px;
}}
"#,
            columns, gap
        )
    }

    fn generate_responsive_rules(&self) -> String {
        let mut rules = String::new();

        for bp in &self.breakpoints {
            let query = match (bp.min_width, bp.max_width) {
                (0, Some(max)) => format!("(max-width: {}px)", max),
                (min, None) => format!("(min-width: {}px)", min),
                (min, Some(max)) => format!("(min-width: {}px) and (max-width: {}px)", min, max),
            };

            rules.push_str(&format!(
                r#"@media {} {{
  .container {{
    grid-template-columns: repeat({}, 1fr);
    gap: {}px;
  }}
}}"#,
                query, bp.columns, bp.gap
            ));
        }

        rules
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            target_style: RenderTargetStyle::Enterprise {
                brand_color: "#0052CC".to_string(),
                typography: "Inter, system-ui, sans-serif".to_string(),
                border_radius: 8.0,
                shadow_style: ShadowStyle::Subtle,
            },
            output_format: OutputFormat::SinglePage,
            optimize_for: OptimizationTarget::Balanced,
            include_source_maps: false,
            minify_output: false,
            output_dir: PathBuf::from("./output"),
        }
    }
}

impl Default for RenderingModule {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rendering_module_creation() {
        let module = RenderingModule::new();
        assert!(
            module.config.output_dir.exists()
                || module.config.output_dir.to_string_lossy().is_empty()
        );
    }

    #[test]
    fn test_layout_engine_grid() {
        let engine = LayoutEngine::new(RenderTargetStyle::Enterprise {
            brand_color: "#000".to_string(),
            typography: "Arial".to_string(),
            border_radius: 4.0,
            shadow_style: ShadowStyle::None,
        });

        let grid_css = engine.generate_grid_layout(12, 24);
        assert!(grid_css.contains("grid-template-columns"));
        assert!(grid_css.contains("repeat(12, 1fr)"));
    }

    #[test]
    fn test_responsive_rules() {
        let engine = LayoutEngine::new(RenderTargetStyle::default());

        let rules = engine.generate_responsive_rules();
        assert!(rules.contains("@media"));
        assert!(rules.contains("(min-width:"));
    }

    #[test]
    fn test_accessibility_score() {
        let module = RenderingModule::new();

        let files = vec![
            RenderedFile {
                path: "index.html".to_string(),
                content: r#"<html><head><title>Test</title></head><body><img src="test.jpg" alt="Test"></body></html>"#.to_string(),
                file_type: FileType::Html,
                size_bytes: 100,
            },
        ];

        let score = module.calculate_accessibility_score(&files);
        assert!(score > 50.0);
    }

    #[test]
    fn test_seo_score() {
        let module = RenderingModule::new();

        let files = vec![RenderedFile {
            path: "index.html".to_string(),
            content: r#"
<html>
<head>
<title>Test Page</title>
<meta name="description" content="A test page">
<meta name="robots" content="index, follow">
<link rel="canonical" href="https://example.com">
</head>
<body><h1>Welcome</h1></body>
</html>
"#
            .to_string(),
            file_type: FileType::Html,
            size_bytes: 200,
        }];

        let score = module.calculate_seo_score(&files);
        assert!(score > 70.0);
    }
}
