use anyhow::Result;
use reqwest::blocking::Client;
use std::time::Duration;

use crate::ai::AiRuntime;
use crate::parser::{HtmlParser, CssParser, JsParser};
use crate::renderer::RenderEngine;

/// 真实网站访问和学习系统
pub struct WebsiteLearner {
    runtime: AiRuntime,
    client: Client,
}

impl WebsiteLearner {
    /// 创建新的网站学习器
    pub fn new(runtime: AiRuntime) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("BrowerAI/0.1.0 (AI Learning Browser)")
            .build()?;

        Ok(Self { runtime, client })
    }

    /// 访问并学习一个网站
    pub fn visit_and_learn(&self, url: &str) -> Result<VisitReport> {
        log::info!("🌐 开始访问网站: {}", url);
        
        let start = std::time::Instant::now();
        
        // 1. 获取 HTML
        log::info!("  📥 正在获取 HTML...");
        let response = self.client.get(url).send()?;
        let html = response.text()?;
        let fetch_duration = start.elapsed();
        
        log::info!("  ✅ 获取成功，大小: {} bytes，耗时: {:.2}s", 
            html.len(), 
            fetch_duration.as_secs_f64()
        );

        // 2. 解析 HTML
        log::info!("  🔍 正在解析 HTML...");
        let parser = HtmlParser::with_ai_runtime(self.runtime.clone());
        let parse_start = std::time::Instant::now();
        
        let dom = match parser.parse(&html) {
            Ok(dom) => {
                let parse_duration = parse_start.elapsed();
                log::info!("  ✅ HTML 解析成功，耗时: {:.2}ms", parse_duration.as_secs_f64() * 1000.0);
                
                // 记录到反馈管道（保存实际HTML内容）
                self.runtime.feedback().record_html_parsing(
                    true,
                    0.5, // 默认复杂度
                    true,
                    None,
                    Some(html.to_string()),
                    html.len(),
                );
                
                Some(dom)
            }
            Err(e) => {
                log::error!("  ❌ HTML 解析失败: {}", e);
                self.runtime.feedback().record_html_parsing(
                    false,
                    0.0,
                    true,
                    Some(e.to_string()),
                    Some(html.to_string()),
                    html.len(),
                );
                None
            }
        };

        // 3. 提取文本内容
        let text_content = if let Some(ref dom) = dom {
            let text = parser.extract_text(dom);
            log::info!("  📝 提取文本内容: {} 字符", text.len());
            Some(text)
        } else {
            None
        };

        // 4. 查找并解析 CSS（简化版）
        log::info!("  🎨 正在查找 CSS...");
        let css_parser = CssParser::with_ai_runtime(self.runtime.clone());
        let css_count = self.extract_and_parse_css(&html, &css_parser);

        // 5. 查找并解析 JavaScript（简化版）
        log::info!("  ⚙️  正在查找 JavaScript...");
        let js_parser = JsParser::with_ai_runtime(self.runtime.clone());
        let js_count = self.extract_and_parse_js(&html, &js_parser);

        // 6. 渲染（如果解析成功）
        let render_node_count = if let Some(ref dom) = dom {
            log::info!("  🖼️  正在渲染...");
            let mut render_engine = RenderEngine::new();
            match render_engine.render(dom, &[]) {
                Ok(tree) => {
                    log::info!("  ✅ 渲染完成，节点数: {}", tree.nodes.len());
                    Some(tree.nodes.len())
                }
                Err(e) => {
                    log::error!("  ❌ 渲染失败: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let total_duration = start.elapsed();

        let report = VisitReport {
            url: url.to_string(),
            success: dom.is_some(),
            html_size: html.len(),
            text_length: text_content.as_ref().map(|t| t.len()),
            css_count,
            js_count,
            render_node_count,
            fetch_duration_ms: fetch_duration.as_secs_f64() * 1000.0,
            total_duration_ms: total_duration.as_secs_f64() * 1000.0,
        };

        log::info!("✅ 访问完成！");
        log::info!("  总耗时: {:.2}ms", report.total_duration_ms);
        log::info!("  反馈事件数: {}", self.runtime.feedback().len());

        Ok(report)
    }

    /// 提取并解析 CSS
    fn extract_and_parse_css(&self, html: &str, parser: &CssParser) -> usize {
        let mut count = 0;
        
        // 简单的 CSS 提取（查找 <style> 标签）
        for style_block in html.split("<style>").skip(1) {
            if let Some(css) = style_block.split("</style>").next() {
                match parser.parse(css) {
                    Ok(rules) => {
                        count += rules.len();
                        self.runtime.feedback().record_css_parsing(
                            true,
                            rules.len(),
                            true,
                            None,
                            Some(css.to_string()),
                        );
                    }
                    Err(e) => {
                        self.runtime.feedback().record_css_parsing(
                            false,
                            0,
                            true,
                            Some(e.to_string()),
                            Some(css.to_string()),
                        );
                    }
                }
            }
        }
        
        count
    }

    /// 提取并解析 JavaScript
    fn extract_and_parse_js(&self, html: &str, parser: &JsParser) -> usize {
        let mut count = 0;
        
        // 简单的 JS 提取（查找 <script> 标签）
        for script_block in html.split("<script>").skip(1) {
            if let Some(js) = script_block.split("</script>").next() {
                if !js.trim().is_empty() {
                    match parser.parse(js) {
                        Ok(ast) => {
                            count += ast.statement_count;
                            self.runtime.feedback().record_js_parsing(
                                true,
                                ast.statement_count,
                                vec![],
                                true,
                                None,
                                Some(js.to_string()),
                            );
                        }
                        Err(e) => {
                            self.runtime.feedback().record_js_parsing(
                                false,
                                0,
                                vec![],
                                true,
                                Some(e.to_string()),
                                Some(js.to_string()),
                            );
                        }
                    }
                }
            }
        }
        
        count
    }

    /// 批量访问多个网站
    pub fn batch_visit(&self, urls: &[&str]) -> Vec<VisitReport> {
        let mut reports = Vec::new();
        
        for (i, url) in urls.iter().enumerate() {
            log::info!("\n📍 [{}/{}] 访问: {}", i + 1, urls.len(), url);
            
            match self.visit_and_learn(url) {
                Ok(report) => reports.push(report),
                Err(e) => log::error!("❌ 访问失败: {}", e),
            }
            
            // 避免请求过快
            if i < urls.len() - 1 {
                std::thread::sleep(Duration::from_secs(1));
            }
        }
        
        reports
    }

    /// 导出学习到的反馈数据
    pub fn export_feedback(&self, path: &str) -> Result<()> {
        let json = self.runtime.feedback().export_training_samples()?;
        std::fs::write(path, json)?;
        log::info!("💾 反馈数据已导出到: {}", path);
        Ok(())
    }
}

/// 网站访问报告
#[derive(Debug, Clone)]
pub struct VisitReport {
    pub url: String,
    pub success: bool,
    pub html_size: usize,
    pub text_length: Option<usize>,
    pub css_count: usize,
    pub js_count: usize,
    pub render_node_count: Option<usize>,
    pub fetch_duration_ms: f64,
    pub total_duration_ms: f64,
}

impl VisitReport {
    /// 生成可读的报告
    pub fn format(&self) -> String {
        format!(
            "网站: {}\n\
             成功: {}\n\
             HTML 大小: {} bytes\n\
             文本长度: {} 字符\n\
             CSS 规则: {}\n\
             JS 语句: {}\n\
             渲染节点: {}\n\
             获取耗时: {:.2}ms\n\
             总耗时: {:.2}ms",
            self.url,
            if self.success { "✅" } else { "❌" },
            self.html_size,
            self.text_length.unwrap_or(0),
            self.css_count,
            self.js_count,
            self.render_node_count.unwrap_or(0),
            self.fetch_duration_ms,
            self.total_duration_ms
        )
    }
}
