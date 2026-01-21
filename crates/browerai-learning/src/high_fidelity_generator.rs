// 高保真网站生成器 - 完整的网站重建系统
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::browser_tech_detector::TechnologyDetectionResult;
use crate::external_resource_analyzer::ExternalResourceGraph;
use crate::wasm_analyzer::WasmModuleInfo;
use crate::websocket_analyzer::WebSocketInfo;

/// 网站完整分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebsiteAnalysisComplete {
    pub url: String,
    pub title: String,
    pub html_structure: HtmlStructure,
    pub styles: StyleAnalysis,
    pub scripts: ScriptAnalysis,
    pub resources: ExternalResourceGraph,
    pub tech_stack: TechnologyDetectionResult,
    pub wasm_modules: Vec<WasmModuleInfo>,
    pub websockets: Vec<WebSocketInfo>,
    pub api_endpoints: Vec<ApiEndpoint>,
    pub authentication: Option<AuthenticationFlow>,
    pub performance: PerformanceMetrics,
}

/// HTML 结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlStructure {
    pub elements: Vec<ElementInfo>,
    pub layout_hierarchy: LayoutHierarchy,
    pub semantic_structure: SemanticStructure,
}

/// 元素信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementInfo {
    pub tag: String,
    pub id: Option<String>,
    pub classes: Vec<String>,
    pub attributes: HashMap<String, String>,
    pub text_content: Option<String>,
    pub children_count: usize,
    pub xpath: String,
    pub css_selector: String,
}

/// 布局层级
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutHierarchy {
    pub header: Option<ElementInfo>,
    pub navigation: Vec<ElementInfo>,
    pub main_content: Option<ElementInfo>,
    pub sidebar: Vec<ElementInfo>,
    pub footer: Option<ElementInfo>,
}

/// 语义结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticStructure {
    pub forms: Vec<FormInfo>,
    pub tables: Vec<TableInfo>,
    pub lists: Vec<ListInfo>,
    pub media: Vec<MediaInfo>,
}

/// 表单信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormInfo {
    pub action: Option<String>,
    pub method: String,
    pub fields: Vec<FormField>,
    pub validation_rules: HashMap<String, Vec<String>>,
}

/// 表单字段
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormField {
    pub name: String,
    pub field_type: String,
    pub required: bool,
    pub placeholder: Option<String>,
    pub default_value: Option<String>,
    pub validation: Vec<String>,
}

/// 表格信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableInfo {
    pub headers: Vec<String>,
    pub rows_count: usize,
    pub is_sortable: bool,
    pub is_filterable: bool,
}

/// 列表信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListInfo {
    pub list_type: String, // ul, ol
    pub items_count: usize,
    pub nested_level: u32,
}

/// 媒体信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaInfo {
    pub media_type: String, // img, video, audio
    pub src: String,
    pub alt: Option<String>,
    pub dimensions: Option<(u32, u32)>,
}

/// 样式分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleAnalysis {
    pub inline_styles: Vec<InlineStyle>,
    pub stylesheets: Vec<Stylesheet>,
    pub computed_styles: HashMap<String, ComputedStyle>,
    pub css_variables: HashMap<String, String>,
    pub media_queries: Vec<MediaQuery>,
}

/// 内联样式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineStyle {
    pub selector: String,
    pub properties: HashMap<String, String>,
}

/// 样式表
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stylesheet {
    pub url: Option<String>,
    pub rules: Vec<CssRule>,
}

/// CSS 规则
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CssRule {
    pub selector: String,
    pub properties: HashMap<String, String>,
    pub specificity: u32,
}

/// 计算样式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputedStyle {
    pub element: String,
    pub final_styles: HashMap<String, String>,
}

/// 媒体查询
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaQuery {
    pub condition: String,
    pub rules: Vec<CssRule>,
}

/// 脚本分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptAnalysis {
    pub inline_scripts: Vec<String>,
    pub external_scripts: Vec<ExternalScript>,
    pub event_listeners: Vec<EventListener>,
    pub global_variables: Vec<String>,
    pub functions: Vec<FunctionInfo>,
    pub api_calls: Vec<ApiCall>,
}

/// 外部脚本
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalScript {
    pub url: String,
    pub async_loading: bool,
    pub defer_loading: bool,
    pub module_type: bool,
}

/// 事件监听器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventListener {
    pub element: String,
    pub event_type: String,
    pub handler: String,
}

/// 函数信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionInfo {
    pub name: String,
    pub parameters: Vec<String>,
    pub is_async: bool,
    pub calls: Vec<String>,
}

/// API 调用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiCall {
    pub method: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body: Option<String>,
}

/// API 端点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEndpoint {
    pub path: String,
    pub method: String,
    pub request_format: Option<serde_json::Value>,
    pub response_format: Option<serde_json::Value>,
    pub authentication_required: bool,
    pub rate_limit: Option<RateLimit>,
}

/// 速率限制
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_minute: u32,
    pub burst_size: u32,
}

/// 认证流程
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationFlow {
    pub method: String,
    pub login_url: String,
    pub token_endpoint: Option<String>,
    pub required_fields: Vec<String>,
    pub session_management: SessionManagement,
}

/// 会话管理
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManagement {
    pub storage_type: String, // cookie, localStorage, sessionStorage
    pub session_key: String,
    pub expiry_time: Option<u64>,
}

/// 性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub load_time_ms: u64,
    pub dom_content_loaded_ms: u64,
    pub first_paint_ms: u64,
    pub first_contentful_paint_ms: u64,
    pub largest_contentful_paint_ms: u64,
    pub total_resources: usize,
    pub total_size_bytes: usize,
    pub javascript_execution_time_ms: u64,
}

/// 高保真生成器
#[allow(dead_code)]
pub struct HighFidelityGenerator {
    preserve_structure: bool,
    optimize_performance: bool,
    include_comments: bool,
}

impl Default for HighFidelityGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl HighFidelityGenerator {
    pub fn new() -> Self {
        Self {
            preserve_structure: true,
            optimize_performance: true,
            include_comments: true,
        }
    }

    pub fn with_options(
        preserve_structure: bool,
        optimize_performance: bool,
        include_comments: bool,
    ) -> Self {
        Self {
            preserve_structure,
            optimize_performance,
            include_comments,
        }
    }

    /// 生成完整网站
    pub fn generate_website(&self, analysis: &WebsiteAnalysisComplete) -> Result<GeneratedWebsite> {
        log::info!("开始生成高保真网站: {}", analysis.url);

        let html = self.generate_html(analysis)?;
        let css = self.generate_css(&analysis.styles)?;
        let js = self.generate_javascript(&analysis.scripts, &analysis.api_endpoints)?;
        let assets = self.prepare_assets(&analysis.resources)?;

        let website = GeneratedWebsite {
            html,
            css,
            javascript: js,
            assets,
            config: self.generate_config(analysis)?,
        };

        log::info!("网站生成完成");
        Ok(website)
    }

    /// 生成 HTML
    fn generate_html(&self, analysis: &WebsiteAnalysisComplete) -> Result<String> {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html lang=\"zh-CN\">\n");
        html.push_str("<head>\n");
        html.push_str("  <meta charset=\"UTF-8\">\n");
        html.push_str(
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        html.push_str(&format!("  <title>{}</title>\n", analysis.title));

        if self.include_comments {
            html.push_str(&format!("  <!-- 原始网站: {} -->\n", analysis.url));
            html.push_str(&format!(
                "  <!-- 技术栈: {} -->\n",
                analysis
                    .tech_stack
                    .detected_technologies
                    .values()
                    .map(|info| info.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        html.push_str("  <link rel=\"stylesheet\" href=\"styles.css\">\n");
        html.push_str("</head>\n");
        html.push_str("<body>\n");

        // 生成布局结构
        html.push_str(&self.generate_layout(&analysis.html_structure)?);

        html.push_str("  <script src=\"main.js\"></script>\n");
        html.push_str("</body>\n");
        html.push_str("</html>\n");

        Ok(html)
    }

    /// 生成布局
    fn generate_layout(&self, structure: &HtmlStructure) -> Result<String> {
        let mut layout = String::new();

        // Header
        if let Some(_header) = &structure.layout_hierarchy.header {
            layout.push_str("  <header>\n");
            layout.push_str("    <h1>网站标题</h1>\n");
            layout.push_str("    <nav>\n");
            layout.push_str("      <ul>\n");
            layout.push_str("        <li><a href=\"#\">首页</a></li>\n");
            layout.push_str("        <li><a href=\"#\">关于</a></li>\n");
            layout.push_str("      </ul>\n");
            layout.push_str("    </nav>\n");
            layout.push_str("  </header>\n\n");
        }

        // Main content
        layout.push_str("  <main>\n");

        // Forms
        for form in &structure.semantic_structure.forms {
            layout.push_str(&self.generate_form(form)?);
        }

        // Tables
        for table in &structure.semantic_structure.tables {
            layout.push_str(&self.generate_table(table)?);
        }

        layout.push_str("  </main>\n\n");

        // Footer
        if let Some(_footer) = &structure.layout_hierarchy.footer {
            layout.push_str("  <footer>\n");
            layout.push_str("    <p>&copy; 2024 保留所有权利</p>\n");
            layout.push_str("  </footer>\n");
        }

        Ok(layout)
    }

    /// 生成表单
    fn generate_form(&self, form: &FormInfo) -> Result<String> {
        let mut html = String::new();

        html.push_str("    <form");
        if let Some(action) = &form.action {
            html.push_str(&format!(" action=\"{}\"", action));
        }
        html.push_str(&format!(" method=\"{}\">\n", form.method));

        for field in &form.fields {
            html.push_str("      <div class=\"form-group\">\n");
            html.push_str(&format!(
                "        <label for=\"{}\">{}</label>\n",
                field.name, field.name
            ));
            html.push_str(&format!(
                "        <input type=\"{}\" id=\"{}\" name=\"{}\"",
                field.field_type, field.name, field.name
            ));

            if field.required {
                html.push_str(" required");
            }

            if let Some(placeholder) = &field.placeholder {
                html.push_str(&format!(" placeholder=\"{}\"", placeholder));
            }

            html.push_str(">\n");
            html.push_str("      </div>\n");
        }

        html.push_str("      <button type=\"submit\">提交</button>\n");
        html.push_str("    </form>\n\n");

        Ok(html)
    }

    /// 生成表格
    fn generate_table(&self, table: &TableInfo) -> Result<String> {
        let mut html = String::new();

        html.push_str("    <table>\n");
        html.push_str("      <thead>\n        <tr>\n");

        for header in &table.headers {
            html.push_str(&format!("          <th>{}</th>\n", header));
        }

        html.push_str("        </tr>\n      </thead>\n");
        html.push_str("      <tbody>\n");
        html.push_str("        <!-- 数据行 -->\n");
        html.push_str("      </tbody>\n");
        html.push_str("    </table>\n\n");

        Ok(html)
    }

    /// 生成 CSS
    fn generate_css(&self, styles: &StyleAnalysis) -> Result<String> {
        let mut css = String::new();

        if self.include_comments {
            css.push_str("/* 自动生成的样式表 */\n\n");
        }

        // Reset styles
        css.push_str("* {\n");
        css.push_str("  margin: 0;\n");
        css.push_str("  padding: 0;\n");
        css.push_str("  box-sizing: border-box;\n");
        css.push_str("}\n\n");

        // CSS Variables
        if !styles.css_variables.is_empty() {
            css.push_str(":root {\n");
            for (key, value) in &styles.css_variables {
                css.push_str(&format!("  {}: {};\n", key, value));
            }
            css.push_str("}\n\n");
        }

        // Body styles
        css.push_str("body {\n");
        css.push_str("  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;\n");
        css.push_str("  line-height: 1.6;\n");
        css.push_str("  color: #333;\n");
        css.push_str("}\n\n");

        // Layout styles
        css.push_str("header {\n");
        css.push_str("  background: #333;\n");
        css.push_str("  color: #fff;\n");
        css.push_str("  padding: 1rem;\n");
        css.push_str("}\n\n");

        css.push_str("main {\n");
        css.push_str("  max-width: 1200px;\n");
        css.push_str("  margin: 0 auto;\n");
        css.push_str("  padding: 2rem;\n");
        css.push_str("}\n\n");

        // Form styles
        css.push_str(".form-group {\n");
        css.push_str("  margin-bottom: 1rem;\n");
        css.push_str("}\n\n");

        css.push_str("input {\n");
        css.push_str("  width: 100%;\n");
        css.push_str("  padding: 0.5rem;\n");
        css.push_str("  border: 1px solid #ddd;\n");
        css.push_str("  border-radius: 4px;\n");
        css.push_str("}\n\n");

        // Media queries
        for query in &styles.media_queries {
            css.push_str(&format!("@media {} {{\n", query.condition));
            for rule in &query.rules {
                css.push_str(&format!("  {} {{\n", rule.selector));
                for (prop, value) in &rule.properties {
                    css.push_str(&format!("    {}: {};\n", prop, value));
                }
                css.push_str("  }\n");
            }
            css.push_str("}\n\n");
        }

        Ok(css)
    }

    /// 生成 JavaScript
    fn generate_javascript(
        &self,
        scripts: &ScriptAnalysis,
        api_endpoints: &[ApiEndpoint],
    ) -> Result<String> {
        let mut js = String::new();

        if self.include_comments {
            js.push_str("// 自动生成的脚本\n\n");
        }

        // API 客户端
        if !api_endpoints.is_empty() {
            js.push_str(&self.generate_api_client(api_endpoints)?);
        }

        // 事件监听器
        js.push_str("\n// 事件监听器\n");
        js.push_str("document.addEventListener('DOMContentLoaded', function() {\n");

        for listener in &scripts.event_listeners {
            js.push_str(&format!(
                "  // {} on {}\n",
                listener.event_type, listener.element
            ));
        }

        js.push_str("  console.log('页面加载完成');\n");
        js.push_str("});\n\n");

        // 表单处理
        js.push_str("// 表单处理\n");
        js.push_str("const forms = document.querySelectorAll('form');\n");
        js.push_str("forms.forEach(form => {\n");
        js.push_str("  form.addEventListener('submit', async (e) => {\n");
        js.push_str("    e.preventDefault();\n");
        js.push_str("    const formData = new FormData(form);\n");
        js.push_str("    const data = Object.fromEntries(formData);\n");
        js.push_str("    console.log('提交数据:', data);\n");
        js.push_str("    // Send data to server\n");
        js.push_str("    try {\n");
        js.push_str("      const response = await fetch('/api/submit', {\n");
        js.push_str("        method: 'POST',\n");
        js.push_str("        headers: { 'Content-Type': 'application/json' },\n");
        js.push_str("        body: JSON.stringify(data)\n");
        js.push_str("      });\n");
        js.push_str("      const result = await response.json();\n");
        js.push_str("      console.log('提交成功:', result);\n");
        js.push_str("      return result;\n");
        js.push_str("    } catch (error) {\n");
        js.push_str("      console.error('提交失败:', error);\n");
        js.push_str("      throw error;\n");
        js.push_str("    }\n");
        js.push_str("  });\n");
        js.push_str("});\n");

        Ok(js)
    }

    /// 生成 API 客户端
    fn generate_api_client(&self, endpoints: &[ApiEndpoint]) -> Result<String> {
        let mut js = String::new();

        js.push_str("// API 客户端\n");
        js.push_str("class APIClient {\n");
        js.push_str("  constructor(baseURL) {\n");
        js.push_str("    this.baseURL = baseURL;\n");
        js.push_str("    this.headers = {\n");
        js.push_str("      'Content-Type': 'application/json'\n");
        js.push_str("    };\n");
        js.push_str("  }\n\n");

        js.push_str("  async request(method, path, data = null) {\n");
        js.push_str("    const options = {\n");
        js.push_str("      method,\n");
        js.push_str("      headers: this.headers\n");
        js.push_str("    };\n\n");
        js.push_str("    if (data) {\n");
        js.push_str("      options.body = JSON.stringify(data);\n");
        js.push_str("    }\n\n");
        js.push_str("    const response = await fetch(this.baseURL + path, options);\n");
        js.push_str("    return response.json();\n");
        js.push_str("  }\n");

        // 为每个端点生成方法
        for endpoint in endpoints {
            let method_name = endpoint
                .path
                .trim_start_matches('/')
                .replace(['/', '-'], "_");

            js.push_str(&format!("\n  async {}(data) {{\n", method_name));
            js.push_str(&format!(
                "    return this.request('{}', '{}', data);\n",
                endpoint.method, endpoint.path
            ));
            js.push_str("  }\n");
        }

        js.push_str("}\n\n");
        js.push_str("const api = new APIClient('/api');\n");

        Ok(js)
    }

    /// 准备资源
    fn prepare_assets(
        &self,
        resources: &ExternalResourceGraph,
    ) -> Result<HashMap<String, Vec<u8>>> {
        let assets = HashMap::new();

        log::info!("准备 {} 个外部资源", resources.resources.len());

        for resource in resources.resources.values() {
            let url = &resource.url;
            if url.starts_with("http") {
                log::debug!(
                    "Skipping resource download for {} (async not available in sync context)",
                    url
                );
                continue;

                // 异步版本需要在外层调用时使用
                // let client = reqwest::blocking::Client::new();
                // match client.get(url).send() {
                //     Ok(response) => {
                //         if response.status().is_success() {
                //             if let Ok(bytes) = response.bytes() {
                //                 assets.insert(url.clone(), bytes.to_vec());
                //             }
                //         }
                //     }
                //     Err(e) => {
                //         log::warn!("下载资源失败 {}: {}", url, e);
                //     }
                // }
            }
        }

        Ok(assets)
    }

    /// 生成配置
    fn generate_config(&self, analysis: &WebsiteAnalysisComplete) -> Result<WebsiteConfig> {
        Ok(WebsiteConfig {
            original_url: analysis.url.clone(),
            title: analysis.title.clone(),
            tech_stack: analysis.tech_stack.clone(),
            requires_authentication: analysis.authentication.is_some(),
            api_base_url: analysis
                .api_endpoints
                .first()
                .map(|e| e.path.split('/').take(2).collect::<Vec<_>>().join("/"))
                .unwrap_or_default(),
        })
    }
}

/// 生成的网站
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedWebsite {
    pub html: String,
    pub css: String,
    pub javascript: String,
    pub assets: HashMap<String, Vec<u8>>,
    pub config: WebsiteConfig,
}

/// 网站配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebsiteConfig {
    pub original_url: String,
    pub title: String,
    pub tech_stack: TechnologyDetectionResult,
    pub requires_authentication: bool,
    pub api_base_url: String,
}

impl GeneratedWebsite {
    /// 保存到文件系统
    pub fn save_to_directory(&self, output_dir: &str) -> Result<()> {
        use std::fs;
        use std::path::Path;

        let path = Path::new(output_dir);
        fs::create_dir_all(path)?;

        fs::write(path.join("index.html"), &self.html)?;
        fs::write(path.join("styles.css"), &self.css)?;
        fs::write(path.join("main.js"), &self.javascript)?;
        fs::write(
            path.join("config.json"),
            serde_json::to_string_pretty(&self.config)?,
        )?;

        // 保存资源文件
        let assets_dir = path.join("assets");
        fs::create_dir_all(&assets_dir)?;
        for (name, data) in &self.assets {
            fs::write(assets_dir.join(name), data)?;
        }

        log::info!("网站已保存到: {}", output_dir);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        let generator = HighFidelityGenerator::new();
        assert!(generator.preserve_structure);
        assert!(generator.optimize_performance);
        assert!(generator.include_comments);
    }

    #[test]
    fn test_form_generation() {
        let generator = HighFidelityGenerator::new();
        let form = FormInfo {
            action: Some("/submit".to_string()),
            method: "POST".to_string(),
            fields: vec![FormField {
                name: "username".to_string(),
                field_type: "text".to_string(),
                required: true,
                placeholder: Some("请输入用户名".to_string()),
                default_value: None,
                validation: vec![],
            }],
            validation_rules: HashMap::new(),
        };

        let html = generator.generate_form(&form).unwrap();
        assert!(html.contains("username"));
        assert!(html.contains("required"));
        assert!(html.contains("请输入用户名"));
    }

    #[test]
    fn test_table_generation() {
        let generator = HighFidelityGenerator::new();
        let table = TableInfo {
            headers: vec!["姓名".to_string(), "年龄".to_string()],
            rows_count: 10,
            is_sortable: true,
            is_filterable: false,
        };

        let html = generator.generate_table(&table).unwrap();
        assert!(html.contains("<table>"));
        assert!(html.contains("姓名"));
        assert!(html.contains("年龄"));
    }
}
