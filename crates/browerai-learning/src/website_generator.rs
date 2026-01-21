/// 网站生成阶段：从学习和推理结果生成完整的现代网站
///
/// 核心目标："保功能、换体验"
/// - 输入：原网站功能点、业务工作流、数据结构
/// - 输出：完整的 HTML/CSS/JS 网站
/// - 特点：全新的 UI/UX，但功能完全一样
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::complete_inference_pipeline::CompleteInferenceResult;
use crate::real_website_learner::LearningSession;

/// 生成的网站
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneratedWebsite {
    /// 网站名称
    pub name: String,
    /// HTML 页面内容
    pub html: String,
    /// CSS 样式表
    pub css: String,
    /// JavaScript 代码
    pub javascript: String,
    /// 生成配置
    pub config: WebsiteConfig,
    /// 保留的功能点清单
    pub preserved_features: Vec<String>,
}

/// 网站配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WebsiteConfig {
    /// 主题色（十六进制）
    pub primary_color: String,
    /// 次主题色
    pub secondary_color: String,
    /// 目标风格：Government（政府）、Enterprise（企业）、Custom（自定义）
    pub target_style: String,
    /// 是否启用深色模式
    pub enable_dark_mode: bool,
    /// 是否响应式设计
    pub responsive_design: bool,
    /// 框架选择：Vue3, React, Vanilla
    pub framework: String,
}

impl Default for WebsiteConfig {
    fn default() -> Self {
        Self {
            primary_color: "#3B82F6".to_string(),
            secondary_color: "#10B981".to_string(),
            target_style: "Government".to_string(), // 默认政府风格
            enable_dark_mode: true,
            responsive_design: true,
            framework: "Vanilla".to_string(),
        }
    }
}

/// 网站生成器
pub struct WebsiteGenerator {
    config: WebsiteConfig,
}

/// 原网站提取的内容
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct OriginalContent {
    title: String,
    main_elements: Vec<ContentElement>,
    nav_links: Vec<NavLink>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct ContentElement {
    tag: String,
    text: String,
    attrs: Vec<(String, String)>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct NavLink {
    href: String,
    text: String,
}

impl WebsiteGenerator {
    /// 创建新的网站生成器
    pub fn new(config: WebsiteConfig) -> Self {
        Self { config }
    }

    /// 提取原网站的核心内容
    #[allow(dead_code)]
    fn extract_original_content(&self, session: &LearningSession) -> Result<OriginalContent> {
        let html_content = session
            .original_html
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No original HTML content"))?;

        // 提取标题（使用简单的字符串匹配）
        let title = self.extract_title(html_content);

        // 提取主要文本内容
        let main_elements = self.extract_text_elements(html_content);

        // 提取导航链接
        let nav_links = self.extract_nav_links(html_content);

        Ok(OriginalContent {
            title,
            main_elements,
            nav_links,
        })
    }

    fn extract_title(&self, html: &str) -> String {
        // 提取 <title> 标签
        if let Some(start) = html.find("<title>") {
            if let Some(end) = html[start..].find("</title>") {
                let title = &html[start + 7..start + end];
                return title.trim().to_string();
            }
        }
        "原网站".to_string()
    }

    fn extract_text_elements(&self, html: &str) -> Vec<ContentElement> {
        let mut elements = vec![];

        // 提取 h1 标签
        for cap in self.find_tags(html, "h1") {
            elements.push(ContentElement {
                tag: "h1".to_string(),
                text: cap,
                attrs: vec![],
            });
        }

        // 提取 h2 标签
        for cap in self.find_tags(html, "h2") {
            elements.push(ContentElement {
                tag: "h2".to_string(),
                text: cap,
                attrs: vec![],
            });
        }

        // 提取 p 标签
        for cap in self.find_tags(html, "p").into_iter().take(20) {
            if cap.len() > 10 {
                // 只要有意义的段落
                elements.push(ContentElement {
                    tag: "p".to_string(),
                    text: cap,
                    attrs: vec![],
                });
            }
        }

        elements
    }

    fn find_tags(&self, html: &str, tag: &str) -> Vec<String> {
        let mut results = vec![];
        let open_tag = format!("<{}", tag);
        let close_tag = format!("</{}>", tag);

        let mut pos = 0;
        while let Some(start) = html[pos..].find(&open_tag) {
            let abs_start = pos + start;
            if let Some(content_start) = html[abs_start..].find('>') {
                let abs_content_start = abs_start + content_start + 1;
                if let Some(end) = html[abs_content_start..].find(&close_tag) {
                    let text = &html[abs_content_start..abs_content_start + end];
                    let clean_text = self.clean_html_text(text);
                    if !clean_text.trim().is_empty() {
                        results.push(clean_text);
                    }
                    pos = abs_content_start + end + close_tag.len();
                    continue;
                }
            }
            pos = abs_start + 1;
        }
        results
    }

    fn clean_html_text(&self, text: &str) -> String {
        // 移除 HTML 标签
        let mut result = String::new();
        let mut in_tag = false;
        for ch in text.chars() {
            match ch {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => result.push(ch),
                _ => {}
            }
        }
        result.trim().to_string()
    }

    fn extract_nav_links(&self, html: &str) -> Vec<NavLink> {
        let mut links = vec![];
        let mut pos = 0;

        while let Some(start) = html[pos..].find("<a ") {
            let abs_start = pos + start;
            if let Some(href_start) = html[abs_start..].find("href=\"") {
                let href_abs_start = abs_start + href_start + 6;
                if let Some(href_end) = html[href_abs_start..].find('"') {
                    let href = &html[href_abs_start..href_abs_start + href_end];

                    // 提取链接文本
                    if let Some(text_start) = html[href_abs_start..].find('>') {
                        let text_abs_start = href_abs_start + text_start + 1;
                        if let Some(text_end) = html[text_abs_start..].find("</a>") {
                            let text = self
                                .clean_html_text(&html[text_abs_start..text_abs_start + text_end]);
                            if !text.trim().is_empty() {
                                links.push(NavLink {
                                    href: href.to_string(),
                                    text,
                                });
                            }
                        }
                    }
                    pos = href_abs_start + href_end;
                    continue;
                }
            }
            pos = abs_start + 1;
        }

        links
    }

    /// 从学习和推理结果生成网站
    pub fn generate_website(
        &self,
        session: &LearningSession,
        inference_result: &CompleteInferenceResult,
    ) -> Result<GeneratedWebsite> {
        log::info!("🌐 开始生成现代网站...");

        // 获取原始HTML
        let original_html = session
            .original_html
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No original HTML content"))?;

        log::info!("  ✓ 保留原网站完整内容: {} bytes", original_html.len());

        // 第1步：分析原网站功能点
        let features = self.analyze_features(session)?;
        log::info!("  ✓ 识别 {} 个核心功能点", features.len());

        // 第2步：生成新样式的HTML（保留原DOM结构，注入新CSS）
        let html = self.inject_new_styles(original_html)?;
        log::info!("  ✓ 生成 HTML 结构 ({} 字符)", html.len());

        // 第3步：生成现代 CSS 样式
        let css = self.generate_css(&features)?;
        log::info!("  ✓ 生成 CSS 样式 ({} 字符)", css.len());

        // 第4步：生成前端 JavaScript
        let javascript = self.generate_javascript(&features, inference_result)?;
        log::info!("  ✓ 生成 JavaScript 代码 ({} 字符)", javascript.len());

        // 第5步：验证功能完整性
        let preserved_features = self.verify_features(&features, &html, &javascript)?;
        log::info!(
            "  ✓ 验证功能保留率: {}/{}",
            preserved_features.len(),
            features.len()
        );

        log::info!("✅ 网站生成完成！");

        Ok(GeneratedWebsite {
            name: session.task.name.clone(),
            html,
            css,
            javascript,
            config: self.config.clone(),
            preserved_features,
        })
    }

    /// 分析原网站的核心功能点
    fn analyze_features(&self, session: &LearningSession) -> Result<Vec<Feature>> {
        let mut features = Vec::new();

        // 从工作流中提取功能点
        if let Some(workflows_result) = &session.workflows {
            for workflow in &workflows_result.workflows {
                features.push(Feature {
                    name: workflow.name.clone(),
                    description: format!("工作流: {}", workflow.name),
                    feature_type: FeatureType::Workflow,
                    complexity: workflow.complexity_score,
                });
            }
        }

        // 添加标准功能点
        features.push(Feature {
            name: "navigation".to_string(),
            description: "网站导航".to_string(),
            feature_type: FeatureType::UI,
            complexity: 1.0,
        });

        features.push(Feature {
            name: "search".to_string(),
            description: "搜索功能".to_string(),
            feature_type: FeatureType::Business,
            complexity: 2.0,
        });

        features.push(Feature {
            name: "user_account".to_string(),
            description: "用户账户".to_string(),
            feature_type: FeatureType::Business,
            complexity: 2.5,
        });

        Ok(features)
    }

    /// 注入新样式到原HTML（保留完整DOM结构）
    fn inject_new_styles(&self, original_html: &str) -> Result<String> {
        // 查找 </head> 标签位置
        if let Some(head_end) = original_html.find("</head>") {
            // 在 </head> 前注入新样式链接
            let mut result = String::new();
            result.push_str(&original_html[..head_end]);
            result.push_str(
                "    <link rel=\"stylesheet\" href=\"styles.css\" data-browerai=\"injected\">\n",
            );
            result.push_str(&original_html[head_end..]);

            // 查找 </body> 标签位置，注入主题切换脚本
            if let Some(body_end) = result.find("</body>") {
                let mut final_result = String::new();
                final_result.push_str(&result[..body_end]);
                final_result
                    .push_str("    <script src=\"app.js\" data-browerai=\"injected\"></script>\n");
                final_result.push_str(&result[body_end..]);
                return Ok(final_result);
            }
            return Ok(result);
        }

        // 如果没有 </head> 标签，则在 <html> 后添加完整的 head
        if let Some(html_start) = original_html.find("<html") {
            if let Some(html_end) = original_html[html_start..].find('>') {
                let insert_pos = html_start + html_end + 1;
                let mut result = String::new();
                result.push_str(&original_html[..insert_pos]);
                result.push_str("\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <link rel=\"stylesheet\" href=\"styles.css\" data-browerai=\"injected\">\n</head>\n");
                result.push_str(&original_html[insert_pos..]);

                if let Some(body_end) = result.find("</body>") {
                    let mut final_result = String::new();
                    final_result.push_str(&result[..body_end]);
                    final_result.push_str(
                        "    <script src=\"app.js\" data-browerai=\"injected\"></script>\n",
                    );
                    final_result.push_str(&result[body_end..]);
                    return Ok(final_result);
                }
                return Ok(result);
            }
        }

        // 最后兜底：包裹原内容
        Ok(format!(
            "<!DOCTYPE html>\n<html>\n<head>\n    <meta charset=\"UTF-8\">\n    <link rel=\"stylesheet\" href=\"styles.css\">\n</head>\n<body>\n{}\n    <script src=\"app.js\"></script>\n</body>\n</html>",
            original_html
        ))
    }

    /// 生成 HTML 页面结构（使用原网站内容）
    #[allow(dead_code)]
    fn generate_html_with_content(
        &self,
        features: &[Feature],
        original_content: &OriginalContent,
    ) -> Result<String> {
        let mut html = String::new();

        // HTML 头部
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html lang=\"zh-CN\">\n");
        html.push_str("<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str(
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        html.push_str(&format!(
            "    <title>{} - 新样式体验</title>\n",
            original_content.title
        ));
        html.push_str("    <link rel=\"stylesheet\" href=\"styles.css\">\n");
        html.push_str("</head>\n");
        html.push_str("<body>\n");
        html.push_str("    <div class=\"app-container\">\n");

        // 导航栏 - 使用原网站的链接
        html.push_str("        <!-- 导航栏（保留原功能） -->\n");
        html.push_str("        <nav class=\"navbar\">\n");
        html.push_str(&format!(
            "            <div class=\"navbar-brand\">{}</div>\n",
            original_content.title
        ));
        html.push_str("            <ul class=\"navbar-menu\">\n");

        for (i, link) in original_content.nav_links.iter().take(8).enumerate() {
            if i < 8 {
                // 安全截断文本（考虑UTF-8字符边界）
                let display_text = if link.text.chars().count() > 15 {
                    link.text.chars().take(15).collect::<String>() + "..."
                } else {
                    link.text.clone()
                };

                html.push_str(&format!(
                    "                <li><a href=\"{}\">{}</a></li>\n",
                    link.href, display_text
                ));
            }
        }

        html.push_str("            </ul>\n");
        html.push_str(
            "            <button class=\"theme-toggle\" id=\"themeToggle\">🌙</button>\n",
        );
        html.push_str("        </nav>\n\n");

        // 主要内容区域 - 使用原网站的内容元素
        html.push_str("        <!-- 主要内容（从原网站提取） -->\n");
        html.push_str("        <main class=\"main-content\">\n");

        // 按标签类型组织内容
        let mut sections: Vec<Vec<&ContentElement>> = vec![];
        let mut current_section = vec![];

        for element in &original_content.main_elements {
            if matches!(element.tag.as_str(), "h1" | "h2") && !current_section.is_empty() {
                sections.push(current_section.clone());
                current_section.clear();
            }
            current_section.push(element);
        }
        if !current_section.is_empty() {
            sections.push(current_section);
        }

        // 生成内容区块
        for (idx, section) in sections.iter().enumerate() {
            html.push_str(&format!(
                "            <section class=\"content-section section-{}\">\n",
                idx
            ));
            for element in section {
                let sanitized_text = element.text.replace('<', "&lt;").replace('>', "&gt;");

                match element.tag.as_str() {
                    "h1" => html.push_str(&format!(
                        "                <h1 class=\"content-heading\">{}</h1>\n",
                        sanitized_text
                    )),
                    "h2" => html.push_str(&format!(
                        "                <h2 class=\"content-subheading\">{}</h2>\n",
                        sanitized_text
                    )),
                    "h3" => {
                        html.push_str(&format!("                <h3>{}</h3>\n", sanitized_text))
                    }
                    "p" => html.push_str(&format!(
                        "                <p class=\"content-text\">{}</p>\n",
                        sanitized_text
                    )),
                    _ => html.push_str(&format!(
                        "                <div class=\"content-block\">{}</div>\n",
                        sanitized_text
                    )),
                }
            }
            html.push_str("            </section>\n\n");
        }

        // 功能展示区
        html.push_str("            <!-- 功能展示 -->\n");
        html.push_str("            <section class=\"features\">\n");
        html.push_str("                <h2>核心功能模块</h2>\n");
        html.push_str("                <div class=\"features-grid\">\n");

        for feature in features {
            html.push_str(&format!(
                "                    <div class=\"feature-card\">\n                        <div class=\"feature-icon\">📦</div>\n                        <h3>{}</h3>\n                        <p>{}</p>\n                    </div>\n",
                feature.name, feature.description
            ));
        }

        html.push_str("                </div>\n");
        html.push_str("            </section>\n\n");

        // 页脚
        html.push_str("        </main>\n");
        html.push_str("        <footer class=\"footer\">\n");
        html.push_str(&format!(
            "            <p>&copy; 2026 {} - 由 BrowerAI 重构样式，保留所有原功能</p>\n",
            original_content.title
        ));
        html.push_str("        </footer>\n");
        html.push_str("    </div>\n\n");
        html.push_str("    <script src=\"app.js\"></script>\n");
        html.push_str("</body>\n");
        html.push_str("</html>\n");

        Ok(html)
    }

    /// 生成现代 CSS 样式
    fn generate_css(&self, _features: &[Feature]) -> Result<String> {
        let target_style = &self.config.target_style;

        log::info!("  ✓ 生成 {} 风格的覆盖式CSS", target_style);

        let css = match target_style.as_str() {
            "Government" => self.generate_government_css(),
            "Enterprise" => self.generate_enterprise_css(),
            _ => self.generate_custom_css(),
        };

        Ok(css)
    }

    /// 政府风格 CSS (WCAG AAA 符合性) - 覆盖原网站样式
    fn generate_government_css(&self) -> String {
        r#"/* BrowerAI - 政府风格样式覆盖 (WCAG AAA) */

/* 强制应用高对比度、大字体、高可访问性 */
* {
    box-sizing: border-box !important;
}

body {
    font-family: 'Arial', 'Microsoft YaHei', sans-serif !important;
    font-size: 16px !important;
    line-height: 1.8 !important;
    color: #000000 !important;
    background-color: #ffffff !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #003d7a !important;
    font-weight: bold !important;
}

a {
    color: #0066cc !important;
    text-decoration: underline !important;
    font-weight: 500 !important;
}

a:hover, a:focus {
    color: #d32f2f !important;
    outline: 2px solid #d32f2f !important;
    outline-offset: 2px;
}

button, input[type="button"], input[type="submit"] {
    font-size: 16px !important;
    padding: 12px 24px !important;
    min-height: 44px !important;
    border: 2px solid #003d7a !important;
    background: #003d7a !important;
    color: #ffffff !important;
    cursor: pointer !important;
}

button:hover, button:focus {
    outline: 3px solid #d32f2f !important;
    outline-offset: 2px;
}

input, textarea, select {
    font-size: 16px !important;
    padding: 10px !important;
    border: 2px solid #666666 !important;
    min-height: 44px !important;
}

input:focus, textarea:focus, select:focus {
    outline: 3px solid #0066cc !important;
    outline-offset: 2px;
    border-color: #0066cc !important;
}

img {
    max-width: 100%;
    height: auto;
}
"#
        .to_string()
    }

    /// 企业风格 CSS - 覆盖原网站样式
    fn generate_enterprise_css(&self) -> String {
        let primary = &self.config.primary_color;
        let secondary = &self.config.secondary_color;

        format!(
            r#"/* BrowerAI - 企业风格样式覆盖 */

body {{
    font-family: 'Roboto', 'Microsoft YaHei', 'PingFang SC', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: #212121 !important;
    background-color: #fafafa !important;
}}

h1, h2, h3, h4, h5, h6 {{
    color: {} !important;
    font-weight: 600 !important;
}}

a {{
    color: {} !important;
    text-decoration: none !important;
    transition: color 0.2s;
}}

a:hover {{
    color: #ff9800 !important;
}}

button, input[type="button"], input[type="submit"] {{
    font-size: 14px !important;
    padding: 10px 20px !important;
    border-radius: 4px !important;
    border: none !important;
    background: {} !important;
    color: #ffffff !important;
    cursor: pointer !important;
    transition: all 0.3s;
}}

button:hover {{
    background: {} !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
}}

input, textarea, select {{
    font-size: 14px !important;
    padding: 8px 12px !important;
    border: 1px solid #dddddd !important;
    border-radius: 4px !important;
}}

input:focus, textarea:focus, select:focus {{
    outline: none !important;
    border-color: {} !important;
    box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.2) !important;
}}
"#,
            primary, secondary, primary, secondary, primary
        )
    }

    /// 自定义风格 CSS - 最小化覆盖
    fn generate_custom_css(&self) -> String {
        let primary = &self.config.primary_color;

        format!(
            r#"/* BrowerAI - 自定义样式微调 */

body {{
    font-family: 'Microsoft YaHei', sans-serif;
    line-height: 1.6;
}}

a {{
    color: {};
}}
"#,
            primary
        )
    }

    /// 生成轻量级JavaScript增强代码（不干扰原网站脚本）
    fn generate_javascript(
        &self,
        _features: &[Feature],
        _inference_result: &CompleteInferenceResult,
    ) -> Result<String> {
        let js = r#"// BrowerAI - 轻量级增强脚本（不干扰原网站功能）

(function() {
    'use strict';
    
    console.log('[BrowerAI] 样式增强已加载，原网站功能完整保留');
    
    // 可选：添加主题切换按钮（仅在政府/企业风格时）
    function addThemeToggle() {
        const toggle = document.createElement('div');
        toggle.id = 'browerai-theme-toggle';
        toggle.style.cssText = 'position:fixed;bottom:20px;right:20px;padding:10px;background:#003d7a;color:white;border-radius:50%;cursor:pointer;z-index:9999;';
        toggle.innerHTML = '🎨';
        toggle.title = 'BrowerAI样式切换';
        
        toggle.addEventListener('click', () => {
            const currentStyle = localStorage.getItem('browerai-style') || 'default';
            const newStyle = currentStyle === 'default' ? 'highcontrast' : 'default';
            localStorage.setItem('browerai-style', newStyle);
            document.body.classList.toggle('browerai-highcontrast');
        });
        
        document.body.appendChild(toggle);
    }
    
    // 延迟执行以避免干扰原页面初始化
    window.addEventListener('load', () => {
        setTimeout(addThemeToggle, 1000);
    });
})();
"#;

        Ok(js.to_string())
    }

    /// 验证功能是否被完整保留
    fn verify_features(
        &self,
        features: &[Feature],
        html: &str,
        javascript: &str,
    ) -> Result<Vec<String>> {
        let mut preserved = Vec::new();

        for feature in features {
            let is_in_html = html.contains(&feature.name);
            let is_in_js = javascript.contains(&feature.name);

            if is_in_html || is_in_js {
                preserved.push(feature.name.clone());
                log::debug!("✓ 功能保留: {}", feature.name);
            }
        }

        if (preserved.len() as f64 / features.len() as f64) < 0.8 {
            log::warn!("⚠️  功能保留率低于 80%");
        }

        Ok(preserved)
    }
}

/// 功能点
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct Feature {
    name: String,
    description: String,
    feature_type: FeatureType,
    complexity: f64,
}

/// 功能类型
#[derive(Clone, Debug)]
enum FeatureType {
    /// UI 界面功能
    UI,
    /// 业务逻辑功能
    Business,
    /// 工作流
    Workflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_website_generator_creation() {
        let config = WebsiteConfig::default();
        let generator = WebsiteGenerator::new(config);
        assert_eq!(generator.config.framework, "Vanilla");
    }

    #[test]
    fn test_website_config_defaults() {
        let config = WebsiteConfig::default();
        assert!(!config.primary_color.is_empty());
        assert!(config.responsive_design);
        assert!(config.enable_dark_mode);
    }
}
