use anyhow::Result;
use reqwest::blocking::Client;
use std::time::Duration;

use crate::ai::AiRuntime;
use crate::parser::{HtmlParser, CssParser, JsParser};
use crate::renderer::RenderEngine;

/// Real website visiting and learning system
pub struct WebsiteLearner {
    runtime: AiRuntime,
    client: Client,
}

impl WebsiteLearner {
    /// Create a new website learner
    pub fn new(runtime: AiRuntime) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("BrowerAI/0.1.0 (AI Learning Browser)")
            .build()?;

        Ok(Self { runtime, client })
    }

    /// Visit and learn from a website
    pub fn visit_and_learn(&self, url: &str) -> Result<VisitReport> {
        log::info!("🌐 Starting website visit: {}", url);
        
        let start = std::time::Instant::now();
        
        // 1. Fetch HTML
        log::info!("  📥 Fetching HTML...");
        let response = self.client.get(url).send()?;
        let html = response.text()?;
        let fetch_duration = start.elapsed();
        
        log::info!("  ✅ Fetch succeeded, size: {} bytes, duration: {:.2}s", 
            html.len(), 
            fetch_duration.as_secs_f64()
        );

        // 2. Parse HTML
        log::info!("  🔍 Parsing HTML...");
        let parser = HtmlParser::with_ai_runtime(self.runtime.clone());
        let parse_start = std::time::Instant::now();
        
        let dom = match parser.parse(&html) {
            Ok(dom) => {
                let parse_duration = parse_start.elapsed();
                log::info!("  ✅ HTML parsing succeeded, duration: {:.2}ms", parse_duration.as_secs_f64() * 1000.0);
                
                // Record to feedback pipeline (save actual HTML content)
                self.runtime.feedback().record_html_parsing(
                    true,
                    0.5, // Default complexity
                    true,
                    None,
                    Some(html.to_string()),
                    html.len(),
                );
                
                Some(dom)
            }
            Err(e) => {
                log::error!("  ❌ HTML parsing failed: {}", e);
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

        // 3. Extract text content
        let text_content = if let Some(ref dom) = dom {
            let text = parser.extract_text(dom);
            log::info!("  📝 Extracted text content: {} characters", text.len());
            Some(text)
        } else {
            None
        };

        // 4. Find and parse CSS (simplified)
        log::info!("  🎨 Searching for CSS...");
        let css_parser = CssParser::with_ai_runtime(self.runtime.clone());
        let css_count = self.extract_and_parse_css(&html, &css_parser);

        // 5. Find and parse JavaScript (simplified)
        log::info!("  ⚙️  Searching for JavaScript...");
        let js_parser = JsParser::with_ai_runtime(self.runtime.clone());
        let js_count = self.extract_and_parse_js(&html, &js_parser);

        // 6. Render (if parsing succeeded)
        let render_node_count = if let Some(ref dom) = dom {
            log::info!("  🖼️  Rendering...");
            let mut render_engine = RenderEngine::new();
            match render_engine.render(dom, &[]) {
                Ok(tree) => {
                    log::info!("  ✅ Rendering completed, node count: {}", tree.nodes.len());
                    Some(tree.nodes.len())
                }
                Err(e) => {
                    log::error!("  ❌ Rendering failed: {}", e);
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

        log::info!("✅ Visit completed!");
        log::info!("  Total duration: {:.2}ms", report.total_duration_ms);
        log::info!("  Feedback events: {}", self.runtime.feedback().len());

        Ok(report)
    }

    /// Extract and parse CSS
    fn extract_and_parse_css(&self, html: &str, parser: &CssParser) -> usize {
        let mut count = 0;
        
        // Simple CSS extraction (find <style> tags)
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

    /// Extract and parse JavaScript
    fn extract_and_parse_js(&self, html: &str, parser: &JsParser) -> usize {
        let mut count = 0;
        
        // Simple JS extraction (find <script> tags)
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

    /// Batch visit multiple websites
    pub fn batch_visit(&self, urls: &[&str]) -> Vec<VisitReport> {
        let mut reports = Vec::new();
        
        for (i, url) in urls.iter().enumerate() {
            log::info!("\n📍 [{}/{}] Visiting: {}", i + 1, urls.len(), url);
            
            match self.visit_and_learn(url) {
                Ok(report) => reports.push(report),
                Err(e) => log::error!("❌ Visit failed: {}", e),
            }
            
            // Avoid too frequent requests
            if i < urls.len() - 1 {
                std::thread::sleep(Duration::from_secs(1));
            }
        }
        
        reports
    }

    /// Export learned feedback data
    pub fn export_feedback(&self, path: &str) -> Result<()> {
        let json = self.runtime.feedback().export_training_samples()?;
        std::fs::write(path, json)?;
        log::info!("💾 Feedback data exported to: {}", path);
        Ok(())
    }
}

/// Website visit report
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
    /// Generate a readable report
    pub fn format(&self) -> String {
        format!(
            "Website: {}\n\
             Success: {}\n\
             HTML size: {} bytes\n\
             Text length: {} characters\n\
             CSS rules: {}\n\
             JS statements: {}\n\
             Render nodes: {}\n\
             Fetch duration: {:.2}ms\n\
             Total duration: {:.2}ms",
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
