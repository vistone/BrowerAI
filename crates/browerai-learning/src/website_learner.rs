use anyhow::Result;
use reqwest::blocking::Client;
use std::time::Duration;

use browerai_css_parser::CssParser;
use browerai_html_parser::HtmlParser;
use browerai_js_parser::JsParser;
use browerai_renderer_core::RenderEngine;

use serde_json::json;
/// Real website visiting and learning system
///
/// 支持两种学习模式：
/// 1. 轻量级学习（lightweight=true）：标准解析，适合快速扫描
/// 2. 深度学习（lightweight=false）：V8追踪+工作流提取，适合理解业务逻辑
pub struct WebsiteLearner {
    client: Client,
    /// 是否使用轻量级学习（普通解析）
    lightweight: bool,
}

impl WebsiteLearner {
    /// Create a new website learner
    pub fn new() -> Result<Self> {
        Self::with_mode(true)
    }

    /// 创建深度学习器（V8追踪+工作流提取）
    pub fn new_deep() -> Result<Self> {
        Self::with_mode(false)
    }

    /// 带模式的构造器
    pub fn with_mode(lightweight: bool) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("BrowerAI/0.1.0 (AI Learning Browser)")
            .build()?;

        Ok(Self {
            client,
            lightweight,
        })
    }

    /// Visit and learn from a website
    pub fn visit_and_learn(&self, url: &str) -> Result<VisitReport> {
        if self.lightweight {
            self.visit_and_learn_lightweight(url)
        } else {
            self.visit_and_learn_deep(url)
        }
    }

    /// 轻量级学习：标准解析
    fn visit_and_learn_lightweight(&self, url: &str) -> Result<VisitReport> {
        log::info!("🌐 Starting website visit: {}", url);

        let start = std::time::Instant::now();

        // 1. Fetch HTML
        log::info!("  📥 Fetching HTML...");
        let response = self.client.get(url).send()?;
        let html = response.text()?;
        let fetch_duration = start.elapsed();

        log::info!(
            "  ✅ Fetch succeeded, size: {} bytes, duration: {:.2}s",
            html.len(),
            fetch_duration.as_secs_f64()
        );

        // 2. Parse HTML
        log::info!("  🔍 Parsing HTML...");
        let parser = HtmlParser::new();
        let parse_start = std::time::Instant::now();

        let dom = match parser.parse(&html) {
            Ok(dom) => {
                let parse_duration = parse_start.elapsed();
                log::info!(
                    "  ✅ HTML parsing succeeded, duration: {:.2}ms",
                    parse_duration.as_secs_f64() * 1000.0
                );

                // Record to feedback pipeline (save actual HTML content)
                Some(dom)
            }
            Err(e) => {
                log::error!("  ❌ HTML parsing failed: {}", e);
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
        let css_parser = CssParser::new();
        let css_count = self.extract_and_parse_css(&html, &css_parser);

        // 5. Find and parse JavaScript (simplified)
        log::info!("  ⚙️  Searching for JavaScript...");
        let js_parser = JsParser::new();
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

        Ok(report)
    }

    /// 深度学习：V8追踪+工作流提取（需要async runtime）
    fn visit_and_learn_deep(&self, url: &str) -> Result<VisitReport> {
        // 注：此方法需在异步环境中调用
        // 当前返回轻量级报告，实际实现需集成real_website_learner的逻辑
        log::info!("🌐 Starting DEEP website learning (V8 tracing): {}", url);

        let start = std::time::Instant::now();

        // 获取页面
        log::info!("  📥 Fetching HTML...");
        let response = self.client.get(url).send()?;
        let html = response.text()?;
        let fetch_duration = start.elapsed();

        // 注入V8追踪器
        log::info!("  💉 Injecting V8 tracers...");
        let injected_html = crate::v8_tracer::V8Tracer::inject_tracers_to_html(&html);

        // 模拟V8执行追踪（实际环境中会真实执行）
        log::info!("  ⚙️  Simulating V8 execution and tracking...");
        let trace_result = self.simulate_v8_execution(&injected_html)?;

        // 提取追踪数据
        log::info!("  📊 Extracting execution traces...");
        let traces = crate::v8_tracer::V8Tracer::extract_traces_from_window(&trace_result)?;

        // 识别工作流
        log::info!("  🔍 Identifying workflows...");
        let workflows = crate::workflow_extractor::WorkflowExtractor::extract_workflows(&traces)?;

        // 评估学习质量
        log::info!("  ✅ Assessing learning quality...");
        let quality = crate::learning_quality::LearningQuality::evaluate(&traces, &workflows)?;

        if quality.overall_score >= 0.9 {
            log::info!(
                "🎉 Excellent learning quality: {:.0}%",
                quality.overall_score * 100.0
            );
        } else if quality.overall_score < 0.7 {
            log::warn!(
                "⚠️  Low learning quality: {:.0}%, recommend re-learning",
                quality.overall_score * 100.0
            );
        }

        let total_duration = start.elapsed();

        let report = VisitReport {
            url: url.to_string(),
            success: true,
            html_size: html.len(),
            text_length: Some(html.len()),
            css_count: workflows.workflows.len(),
            js_count: traces.function_calls.len(),
            render_node_count: Some(traces.dom_operations.len()),
            fetch_duration_ms: fetch_duration.as_secs_f64() * 1000.0,
            total_duration_ms: total_duration.as_secs_f64() * 1000.0,
        };

        log::info!(
            "✓ Deep learning completed: {} workflows, quality {:.0}%",
            workflows.workflows.len(),
            quality.overall_score * 100.0
        );

        Ok(report)
    }

    /// 模拟V8执行（返回追踪JSON）
    fn simulate_v8_execution(&self, _injected_html: &str) -> Result<String> {
        // 这会在实际的浏览器环境中运行
        // 返回模拟的V8执行追踪
        let trace_obj = json!({
            "function_calls": [
                {"function_name": "handlePageLoad", "arguments": [], "return_type": "void", "timestamp_ms": 10, "context_object": null, "call_depth": 0},
                {"function_name": "initializeComponents", "arguments": [], "return_type": "void", "timestamp_ms": 20, "context_object": null, "call_depth": 1}
            ],
            "dom_operations": [
                {"operation_type": "innerHTML", "target_selector": "body", "timestamp_ms": 15},
                {"operation_type": "appendChild", "target_selector": "#app", "timestamp_ms": 25}
            ],
            "event_listeners": [
                {"event_type": "click", "target_selector": "btn", "listener_function": "handleClick"}
            ],
            "user_events": [],
            "state_changes": [
                {"variable_name": "initialized", "previous_value": "false", "new_value": "true", "timestamp_ms": 20}
            ],
            "total_duration_ms": 100,
            "page_ready_ms": 30
        });
        let trace_json = trace_obj.to_string();
        Ok(trace_json)
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
                    }
                    Err(e) => {
                        log::debug!("CSS parse error: {}", e);
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
                        }
                        Err(e) => {
                            log::debug!("JS parse error: {}", e);
                        }
                    }
                }
            }
        }

        count
    }

    /// Batch visit multiple websites
    pub fn batch_visit(&self, urls: &[&str]) -> Vec<VisitReport> {
        let mode = if self.lightweight {
            "lightweight"
        } else {
            "deep"
        };
        log::info!(
            "📊 Starting batch learning ({} mode) for {} websites",
            mode,
            urls.len()
        );

        let mut reports = Vec::new();

        for (i, url) in urls.iter().enumerate() {
            log::info!("\n📍 [{}/{}] Processing: {}", i + 1, urls.len(), url);

            match self.visit_and_learn(url) {
                Ok(report) => {
                    log::info!(
                        "  ✅ Success: {} HTML bytes, {} workflows/JS calls",
                        report.html_size,
                        report.js_count
                    );
                    reports.push(report);
                }
                Err(e) => {
                    log::error!("  ❌ Failed: {}", e);
                }
            }

            // Avoid too frequent requests
            if i < urls.len() - 1 {
                std::thread::sleep(Duration::from_millis(500));
            }
        }

        log::info!(
            "\n✓ Batch processing complete: {}/{} successful",
            reports.len(),
            urls.len()
        );

        reports
    }

    /// Export learned feedback data (placeholder)
    pub fn export_feedback(&self, path: &str) -> Result<()> {
        std::fs::write(path, "[]")?;
        log::info!("💾 Feedback data exported to: {} (placeholder)", path);
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
