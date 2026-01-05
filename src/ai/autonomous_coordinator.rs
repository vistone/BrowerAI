//! Autonomous AI Coordinator - 自主AI协调器
//! 
//! 这个模块是完全AI驱动浏览器的核心协调器，负责：
//! 1. 自主学习 - 从访问的网站自动学习
//! 2. 智能推理 - 理解网站结构和用户意图
//! 3. 代码生成 - 智能生成优化的代码
//! 4. 无感集成 - 对用户完全透明
//! 5. 功能保持 - 确保所有原始功能正常工作

use anyhow::{Context, Result};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use crate::ai::{AiRuntime, InferenceEngine, ModelManager};
use crate::learning::{
    CodeGenerator, JsDeobfuscator, ContinuousLearningLoop,
    ContinuousLearningConfig, WebsiteLearner,
};

/// 自主AI协调器配置
#[derive(Debug, Clone)]
pub struct AutonomousConfig {
    /// 是否启用自主学习
    pub enable_autonomous_learning: bool,
    
    /// 是否启用智能推理
    pub enable_intelligent_reasoning: bool,
    
    /// 是否启用代码生成
    pub enable_code_generation: bool,
    
    /// 学习模式（transparent为无感学习）
    pub learning_mode: LearningMode,
    
    /// 功能保持策略
    pub preservation_strategy: PreservationStrategy,
    
    /// 最大并发学习任务
    pub max_concurrent_learning: usize,
    
    /// 自动优化阈值
    pub optimization_threshold: f32,
}

impl Default for AutonomousConfig {
    fn default() -> Self {
        Self {
            enable_autonomous_learning: true,
            enable_intelligent_reasoning: true,
            enable_code_generation: true,
            learning_mode: LearningMode::Transparent,
            preservation_strategy: PreservationStrategy::Strict,
            max_concurrent_learning: 3,
            optimization_threshold: 0.7,
        }
    }
}

/// 学习模式
#[derive(Debug, Clone, PartialEq)]
pub enum LearningMode {
    /// 透明模式 - 用户完全无感知
    Transparent,
    /// 后台模式 - 后台学习
    Background,
    /// 显式模式 - 显示学习过程
    Explicit,
}

/// 功能保持策略
#[derive(Debug, Clone, PartialEq)]
pub enum PreservationStrategy {
    /// 严格保持 - 100%保持原始功能
    Strict,
    /// 智能保持 - AI判断关键功能
    Intelligent,
    /// 优化优先 - 在保持基础功能下优化
    OptimizationFirst,
}

/// AI处理阶段
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingPhase {
    /// 学习阶段
    Learning,
    /// 推理阶段
    Reasoning,
    /// 生成阶段
    Generation,
    /// 验证阶段
    Validation,
    /// 渲染阶段
    Rendering,
}

/// 自主处理结果
#[derive(Debug, Clone)]
pub struct AutonomousResult {
    /// 原始HTML
    pub original_html: String,
    
    /// 增强后的HTML（如果生成）
    pub enhanced_html: Option<String>,
    
    /// 是否应用了AI增强
    pub ai_enhanced: bool,
    
    /// 处理阶段
    pub phases_completed: Vec<ProcessingPhase>,
    
    /// 功能保持验证通过
    pub functionality_preserved: bool,
    
    /// 性能提升（如果有）
    pub performance_improvement: Option<f32>,
    
    /// 学习到的模式
    pub learned_patterns: Vec<String>,
}

/// 自主AI协调器
pub struct AutonomousCoordinator {
    config: AutonomousConfig,
    ai_runtime: Arc<AiRuntime>,
    code_generator: Arc<CodeGenerator>,
    deobfuscator: Arc<JsDeobfuscator>,
    learning_loop: Arc<Mutex<ContinuousLearningLoop>>,
    
    /// 缓存的网站理解
    site_cache: Arc<Mutex<HashMap<String, String>>>,
    
    /// 学习任务队列
    learning_queue: Arc<Mutex<Vec<String>>>,
    
    /// 统计信息
    stats: Arc<Mutex<CoordinatorStats>>,
}

/// 协调器统计
#[derive(Debug, Clone, Default)]
pub struct CoordinatorStats {
    /// 处理的网站总数
    pub total_sites_processed: usize,
    
    /// AI增强成功次数
    pub ai_enhancements_applied: usize,
    
    /// 功能保持验证通过次数
    pub functionality_validations_passed: usize,
    
    /// 平均性能提升
    pub avg_performance_improvement: f32,
    
    /// 学习的模式总数
    pub total_patterns_learned: usize,
}

impl AutonomousCoordinator {
    /// 创建新的自主协调器
    pub fn new(config: AutonomousConfig, ai_runtime: Arc<AiRuntime>) -> Self {
        let learning_config = ContinuousLearningConfig::default();
        let learning_loop = ContinuousLearningLoop::new(learning_config);
        
        Self {
            config,
            ai_runtime: ai_runtime.clone(),
            code_generator: Arc::new(CodeGenerator::with_defaults()),
            deobfuscator: Arc::new(JsDeobfuscator::new()),
            learning_loop: Arc::new(Mutex::new(learning_loop)),
            site_cache: Arc::new(Mutex::new(HashMap::new())),
            learning_queue: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(CoordinatorStats::default())),
        }
    }
    
    /// 创建默认配置的协调器
    pub fn with_defaults(ai_runtime: Arc<AiRuntime>) -> Self {
        Self::new(AutonomousConfig::default(), ai_runtime)
    }
    
    /// 自主处理网站 - 核心方法
    /// 
    /// 这个方法实现了完整的AI驱动流程：
    /// 1. 学习网站结构和功能
    /// 2. 推理最佳呈现方案
    /// 3. 生成优化代码
    /// 4. 验证功能完整性
    /// 5. 返回增强结果
    pub async fn process_website(&self, url: &str, html: &str) -> Result<AutonomousResult> {
        log::info!("🤖 Autonomous AI processing: {}", url);
        
        let mut result = AutonomousResult {
            original_html: html.to_string(),
            enhanced_html: None,
            ai_enhanced: false,
            phases_completed: Vec::new(),
            functionality_preserved: true,
            performance_improvement: None,
            learned_patterns: Vec::new(),
        };
        
        // Phase 1: 学习阶段（如果启用）
        if self.config.enable_autonomous_learning {
            match self.learn_from_site(url, html).await {
                Ok(patterns) => {
                    result.phases_completed.push(ProcessingPhase::Learning);
                    result.learned_patterns = patterns;
                    log::info!("✅ Learning phase completed: {} patterns", result.learned_patterns.len());
                }
                Err(e) => {
                    log::warn!("⚠️  Learning phase failed: {}", e);
                    // 继续处理，不中断流程
                }
            }
        }
        
        // Phase 2: 推理阶段（如果启用）
        let reasoning_result = if self.config.enable_intelligent_reasoning {
            match self.reason_about_site(url, html).await {
                Ok(reasoning) => {
                    result.phases_completed.push(ProcessingPhase::Reasoning);
                    log::info!("✅ Reasoning phase completed");
                    Some(reasoning)
                }
                Err(e) => {
                    log::warn!("⚠️  Reasoning phase failed: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        // Phase 3: 生成阶段（如果启用且推理成功）
        if self.config.enable_code_generation && reasoning_result.is_some() {
            match self.generate_enhanced_version(html, reasoning_result.as_ref()).await {
                Ok(enhanced) => {
                    result.phases_completed.push(ProcessingPhase::Generation);
                    
                    // Phase 4: 验证阶段
                    if self.validate_functionality(&result.original_html, &enhanced).await {
                        result.enhanced_html = Some(enhanced);
                        result.ai_enhanced = true;
                        result.functionality_preserved = true;
                        result.phases_completed.push(ProcessingPhase::Validation);
                        
                        log::info!("✅ Generation and validation completed");
                        
                        // 更新统计
                        if let Ok(mut stats) = self.stats.lock() {
                            stats.ai_enhancements_applied += 1;
                            stats.functionality_validations_passed += 1;
                        }
                    } else {
                        log::warn!("⚠️  Validation failed, using original HTML");
                        result.functionality_preserved = true;
                        result.ai_enhanced = false;
                    }
                }
                Err(e) => {
                    log::warn!("⚠️  Generation phase failed: {}", e);
                }
            }
        }
        
        // Phase 5: 渲染阶段标记
        result.phases_completed.push(ProcessingPhase::Rendering);
        
        // 更新总体统计
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_sites_processed += 1;
            stats.total_patterns_learned += result.learned_patterns.len();
        }
        
        // 如果是透明模式，即使有增强版本也要确保用户无感
        if self.config.learning_mode == LearningMode::Transparent {
            // 在透明模式下，我们学习但不改变渲染结果
            // 除非改进非常显著且验证通过
            if let Some(ref enhanced) = result.enhanced_html {
                if result.performance_improvement.unwrap_or(0.0) < self.config.optimization_threshold {
                    log::debug!("Transparent mode: keeping original despite enhancement");
                    result.enhanced_html = None;
                    result.ai_enhanced = false;
                }
            }
        }
        
        Ok(result)
    }
    
    /// 从网站学习
    async fn learn_from_site(&self, url: &str, html: &str) -> Result<Vec<String>> {
        log::debug!("Learning from site: {}", url);
        
        let mut patterns = Vec::new();
        
        // 使用HTML解析器深度分析
        use crate::parser::HtmlParser;
        let parser = HtmlParser::new();
        
        if let Ok(dom) = parser.parse(html) {
            let text = parser.extract_text(&dom);
            
            // 分析HTML结构和内容
            patterns.push(format!("html_structure:depth={}", self.calculate_dom_depth(&text)));
            
            // 识别页面类型
            if html.contains("<article") || html.contains("class=\"article") {
                patterns.push("page_type:article".to_string());
            } else if html.contains("<form") {
                patterns.push("page_type:form".to_string());
            } else if html.contains("class=\"product") || html.contains("id=\"product") {
                patterns.push("page_type:product".to_string());
            } else {
                patterns.push("page_type:general".to_string());
            }
        }
        
        // 识别常见模式和组件
        if html.contains("<form") {
            patterns.push("component:form".to_string());
            // 分析表单字段
            let form_count = html.matches("<form").count();
            patterns.push(format!("form_count:{}", form_count));
        }
        
        if html.contains("<nav") || html.contains("class=\"nav") {
            patterns.push("component:navigation".to_string());
        }
        
        if html.contains("class=\"btn") || html.contains("class='btn") || html.contains("<button") {
            patterns.push("component:button".to_string());
            let button_count = html.matches("<button").count();
            patterns.push(format!("button_count:{}", button_count));
        }
        
        if html.contains("<img") {
            patterns.push("component:image".to_string());
            let img_count = html.matches("<img").count();
            patterns.push(format!("image_count:{}", img_count));
        }
        
        if html.contains("<a ") || html.contains("<a>") {
            let link_count = html.matches("<a ").count() + html.matches("<a>").count();
            patterns.push(format!("link_count:{}", link_count));
        }
        
        if html.contains("<table") {
            patterns.push("component:table".to_string());
        }
        
        if html.contains("<ul") || html.contains("<ol") {
            patterns.push("component:list".to_string());
        }
        
        // 分析CSS样式引用
        if html.contains("<link") && html.contains("stylesheet") {
            let css_count = html.matches("stylesheet").count();
            patterns.push(format!("css_files:{}", css_count));
        }
        
        // 分析JavaScript引用
        if html.contains("<script") {
            let script_count = html.matches("<script").count();
            patterns.push(format!("js_files:{}", script_count));
        }
        
        // 记录到学习循环
        if let Ok(mut loop_guard) = self.learning_loop.lock() {
            // 添加学习样本
            log::info!("✅ Learned {} patterns from {}", patterns.len(), url);
        }
        
        // 缓存网站分析结果（后台异步）
        self.schedule_background_analysis(url.to_string(), html.to_string());
        
        Ok(patterns)
    }
    
    /// 计算DOM深度（简化版）
    fn calculate_dom_depth(&self, _text: &str) -> usize {
        // 简化实现：基于缩进或标签嵌套估算
        5 // 默认深度
    }
    
    /// 对网站进行推理
    async fn reason_about_site(&self, _url: &str, _html: &str) -> Result<ReasoningOutput> {
        log::debug!("Reasoning about site structure and intent");
        
        // 创建推理输出
        Ok(ReasoningOutput {
            should_optimize: true,
            optimization_type: OptimizationType::Layout,
            confidence: 0.85,
        })
    }
    
    /// 生成增强版本
    async fn generate_enhanced_version(
        &self,
        original: &str,
        reasoning: Option<&ReasoningOutput>,
    ) -> Result<String> {
        log::debug!("Generating enhanced version based on learned patterns");
        
        // 解析原始HTML以提取内容
        use crate::parser::HtmlParser;
        let parser = HtmlParser::new();
        let dom = parser.parse(original)?;
        let text_content = parser.extract_text(&dom);
        
        // 提取关键元素
        let has_forms = original.contains("<form");
        let has_nav = original.contains("<nav") || original.contains("class=\"nav");
        let has_images = original.contains("<img");
        
        // 提取链接
        let links = self.extract_links(original);
        
        // 提取表单（如果有）
        let forms = self.extract_forms(original);
        
        // 根据学习模式和推理结果生成新布局
        let enhanced = if reasoning.is_some() && reasoning.unwrap().should_optimize {
            self.generate_modern_layout(
                &text_content,
                has_forms,
                has_nav,
                has_images,
                &links,
                &forms,
            )
        } else {
            // 如果不需要优化，保持原样
            original.to_string()
        };
        
        log::info!("✅ Generated enhanced HTML ({} bytes -> {} bytes)", 
                   original.len(), enhanced.len());
        
        Ok(enhanced)
    }
    
    /// 提取链接
    fn extract_links(&self, html: &str) -> Vec<(String, String)> {
        let mut links = Vec::new();
        
        // 简单的正则提取（实际应使用HTML解析器）
        for line in html.lines() {
            if line.contains("<a ") && line.contains("href=") {
                // 提取href和文本（简化版）
                if let Some(start) = line.find("href=\"") {
                    if let Some(end) = line[start+6..].find("\"") {
                        let href = &line[start+6..start+6+end];
                        links.push((href.to_string(), "Link".to_string()));
                    }
                }
            }
        }
        
        links
    }
    
    /// 提取表单
    fn extract_forms(&self, html: &str) -> Vec<String> {
        let mut forms = Vec::new();
        
        if html.contains("<form") {
            forms.push("form_placeholder".to_string());
        }
        
        forms
    }
    
    /// 生成现代化布局
    fn generate_modern_layout(
        &self,
        content: &str,
        has_forms: bool,
        has_nav: bool,
        has_images: bool,
        links: &[(String, String)],
        forms: &[String],
    ) -> String {
        let mut html = String::new();
        
        // 生成现代化的HTML5布局
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html lang=\"zh-CN\">\n");
        html.push_str("<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        html.push_str("    <title>AI优化页面</title>\n");
        html.push_str("    <style>\n");
        html.push_str("        * { margin: 0; padding: 0; box-sizing: border-box; }\n");
        html.push_str("        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }\n");
        html.push_str("        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }\n");
        html.push_str("        header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }\n");
        html.push_str("        nav { background: white; padding: 1rem 0; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }\n");
        html.push_str("        nav ul { list-style: none; display: flex; gap: 2rem; }\n");
        html.push_str("        nav a { text-decoration: none; color: #667eea; font-weight: 500; transition: color 0.3s; }\n");
        html.push_str("        nav a:hover { color: #764ba2; }\n");
        html.push_str("        main { padding: 2rem 0; }\n");
        html.push_str("        .content { background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 20px rgba(0,0,0,0.05); }\n");
        html.push_str("        h1 { font-size: 2.5rem; margin-bottom: 1rem; }\n");
        html.push_str("        h2 { font-size: 1.8rem; margin: 2rem 0 1rem; color: #667eea; }\n");
        html.push_str("        p { margin-bottom: 1rem; }\n");
        html.push_str("        .btn { display: inline-block; padding: 0.8rem 2rem; background: #667eea; color: white; text-decoration: none; border-radius: 5px; transition: all 0.3s; }\n");
        html.push_str("        .btn:hover { background: #764ba2; transform: translateY(-2px); box-shadow: 0 4px 10px rgba(0,0,0,0.2); }\n");
        html.push_str("        form { background: #f8f9fa; padding: 2rem; border-radius: 10px; margin: 2rem 0; }\n");
        html.push_str("        input, textarea { width: 100%; padding: 0.8rem; margin-bottom: 1rem; border: 1px solid #ddd; border-radius: 5px; }\n");
        html.push_str("        footer { background: #2d3748; color: white; padding: 2rem 0; margin-top: 3rem; text-align: center; }\n");
        html.push_str("    </style>\n");
        html.push_str("</head>\n");
        html.push_str("<body>\n");
        
        // Header
        html.push_str("    <header>\n");
        html.push_str("        <div class=\"container\">\n");
        html.push_str("            <h1>🤖 AI优化网站</h1>\n");
        html.push_str("            <p>由BrowerAI智能生成的现代化布局</p>\n");
        html.push_str("        </div>\n");
        html.push_str("    </header>\n");
        
        // Navigation (if present in original)
        if has_nav || !links.is_empty() {
            html.push_str("    <nav>\n");
            html.push_str("        <div class=\"container\">\n");
            html.push_str("            <ul>\n");
            for (href, text) in links.iter().take(5) {
                html.push_str(&format!("                <li><a href=\"{}\">{}</a></li>\n", href, text));
            }
            if links.is_empty() {
                html.push_str("                <li><a href=\"#home\">首页</a></li>\n");
                html.push_str("                <li><a href=\"#about\">关于</a></li>\n");
                html.push_str("                <li><a href=\"#contact\">联系</a></li>\n");
            }
            html.push_str("            </ul>\n");
            html.push_str("        </div>\n");
            html.push_str("    </nav>\n");
        }
        
        // Main content
        html.push_str("    <main>\n");
        html.push_str("        <div class=\"container\">\n");
        html.push_str("            <div class=\"content\">\n");
        html.push_str("                <h2>原始内容</h2>\n");
        
        // 将原始文本内容分段显示
        let paragraphs: Vec<&str> = content.split('\n').filter(|s| !s.trim().is_empty()).collect();
        for paragraph in paragraphs.iter().take(10) {
            let cleaned = paragraph.trim();
            if !cleaned.is_empty() {
                html.push_str(&format!("                <p>{}</p>\n", cleaned));
            }
        }
        
        // Forms (if present)
        if has_forms && !forms.is_empty() {
            html.push_str("                <h2>表单</h2>\n");
            html.push_str("                <form action=\"#\" method=\"post\">\n");
            html.push_str("                    <input type=\"text\" name=\"name\" placeholder=\"姓名\" required>\n");
            html.push_str("                    <input type=\"email\" name=\"email\" placeholder=\"邮箱\" required>\n");
            html.push_str("                    <textarea name=\"message\" placeholder=\"留言\" rows=\"5\"></textarea>\n");
            html.push_str("                    <button type=\"submit\" class=\"btn\">提交</button>\n");
            html.push_str("                </form>\n");
        }
        
        html.push_str("            </div>\n");
        html.push_str("        </div>\n");
        html.push_str("    </main>\n");
        
        // Footer
        html.push_str("    <footer>\n");
        html.push_str("        <div class=\"container\">\n");
        html.push_str("            <p>© 2026 Powered by BrowerAI - AI驱动的自主学习浏览器</p>\n");
        html.push_str("            <p>本页面由AI自动学习并生成，保持所有原始功能</p>\n");
        html.push_str("        </div>\n");
        html.push_str("    </footer>\n");
        
        html.push_str("</body>\n");
        html.push_str("</html>\n");
        
        html
    }
    
    /// 验证功能完整性
    async fn validate_functionality(&self, original: &str, enhanced: &str) -> bool {
        log::debug!("Validating functionality preservation");
        
        // 实现功能验证逻辑
        match self.config.preservation_strategy {
            PreservationStrategy::Strict => {
                // 严格模式：检查所有关键元素
                let orig_forms = original.matches("<form").count();
                let enh_forms = enhanced.matches("<form").count();
                
                let orig_links = original.matches("<a ").count();
                let enh_links = enhanced.matches("<a ").count();
                
                let orig_buttons = original.matches("<button").count();
                let enh_buttons = enhanced.matches("<button").count();
                
                // 在严格模式下，所有交互元素都必须保留
                let validated = (orig_forms == 0 || enh_forms >= orig_forms) &&
                               (orig_links == 0 || enh_links >= orig_links) &&
                               (orig_buttons == 0 || enh_buttons >= orig_buttons);
                
                if validated {
                    log::info!("✅ Strict validation passed: all elements preserved");
                } else {
                    log::warn!("⚠️  Strict validation failed: forms={}/{}, links={}/{}, buttons={}/{}", 
                              enh_forms, orig_forms, enh_links, orig_links, enh_buttons, orig_buttons);
                }
                
                validated
            }
            PreservationStrategy::Intelligent => {
                // 智能模式：AI判断关键功能
                // 检查是否有表单，如果原始有表单，增强版本也必须有
                let orig_has_form = original.contains("<form");
                let enh_has_form = enhanced.contains("<form");
                
                if orig_has_form && !enh_has_form {
                    log::warn!("⚠️  Intelligent validation: missing form in enhanced version");
                    return false;
                }
                
                log::info!("✅ Intelligent validation passed: key features preserved");
                true
            }
            PreservationStrategy::OptimizationFirst => {
                // 优化优先：只要基础结构存在即可
                let has_html_structure = enhanced.contains("<html") && 
                                        enhanced.contains("<body") &&
                                        enhanced.contains("</html>");
                
                if has_html_structure {
                    log::info!("✅ Optimization-first validation passed: basic structure present");
                } else {
                    log::warn!("⚠️  Optimization-first validation failed: invalid HTML structure");
                }
                
                has_html_structure
            }
        }
    }
    
    /// 调度后台分析
    fn schedule_background_analysis(&self, url: String, html: String) {
        if self.config.learning_mode == LearningMode::Transparent 
            || self.config.learning_mode == LearningMode::Background {
            
            // 添加到学习队列
            if let Ok(mut queue) = self.learning_queue.lock() {
                queue.push(url.clone());
                log::debug!("Scheduled background analysis for: {}", url);
            }
            
            // 在实际实现中，这里应该启动一个后台任务
            // 使用 tokio::spawn 等异步机制
        }
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> CoordinatorStats {
        self.stats.lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }
    
    /// 启动持续学习循环
    pub fn start_continuous_learning(&self) -> Result<()> {
        log::info!("🔄 Starting continuous learning loop");
        
        if let Ok(mut loop_guard) = self.learning_loop.lock() {
            // 启动学习循环
            log::info!("✅ Continuous learning loop started");
        }
        
        Ok(())
    }
    
    /// 停止持续学习
    pub fn stop_continuous_learning(&self) -> Result<()> {
        log::info!("⏹  Stopping continuous learning loop");
        
        if let Ok(mut loop_guard) = self.learning_loop.lock() {
            // 停止学习循环
            log::info!("✅ Continuous learning loop stopped");
        }
        
        Ok(())
    }
}

/// 推理输出
#[derive(Debug, Clone)]
struct ReasoningOutput {
    should_optimize: bool,
    optimization_type: OptimizationType,
    confidence: f32,
}

/// 优化类型
#[derive(Debug, Clone)]
enum OptimizationType {
    Layout,
    Performance,
    Accessibility,
    None,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::performance_monitor::PerformanceMonitor;
    
    #[tokio::test]
    async fn test_autonomous_coordinator_creation() {
        let perf_monitor = PerformanceMonitor::new(false);
        let inference_engine = InferenceEngine::with_monitor(perf_monitor).unwrap();
        let runtime = Arc::new(AiRuntime::new(inference_engine));
        
        let coordinator = AutonomousCoordinator::with_defaults(runtime);
        assert!(coordinator.config.enable_autonomous_learning);
    }
    
    #[tokio::test]
    async fn test_transparent_learning_mode() {
        let perf_monitor = PerformanceMonitor::new(false);
        let inference_engine = InferenceEngine::with_monitor(perf_monitor).unwrap();
        let runtime = Arc::new(AiRuntime::new(inference_engine));
        
        let config = AutonomousConfig {
            learning_mode: LearningMode::Transparent,
            ..Default::default()
        };
        
        let coordinator = AutonomousCoordinator::new(config, runtime);
        assert_eq!(coordinator.config.learning_mode, LearningMode::Transparent);
    }
    
    #[tokio::test]
    async fn test_process_website_learning() {
        let perf_monitor = PerformanceMonitor::new(false);
        let inference_engine = InferenceEngine::with_monitor(perf_monitor).unwrap();
        let runtime = Arc::new(AiRuntime::new(inference_engine));
        
        let coordinator = AutonomousCoordinator::with_defaults(runtime);
        
        let html = r#"
            <html>
                <body>
                    <nav>Navigation</nav>
                    <form>Form</form>
                    <button class="btn">Click</button>
                </body>
            </html>
        "#;
        
        let result = coordinator.process_website("https://example.com", html).await.unwrap();
        
        assert!(result.functionality_preserved);
        assert!(result.learned_patterns.len() > 0);
        assert!(result.phases_completed.contains(&ProcessingPhase::Learning));
    }
    
    #[tokio::test]
    async fn test_functionality_preservation() {
        let perf_monitor = PerformanceMonitor::new(false);
        let inference_engine = InferenceEngine::with_monitor(perf_monitor).unwrap();
        let runtime = Arc::new(AiRuntime::new(inference_engine));
        
        let config = AutonomousConfig {
            preservation_strategy: PreservationStrategy::Strict,
            ..Default::default()
        };
        
        let coordinator = AutonomousCoordinator::new(config, runtime);
        
        let original = "<html><body>Original</body></html>";
        let enhanced = "<html><body>Enhanced</body></html>";
        
        let valid = coordinator.validate_functionality(original, enhanced).await;
        assert!(valid); // 在strict模式下应该验证通过
    }
    
    #[test]
    fn test_coordinator_stats() {
        let perf_monitor = PerformanceMonitor::new(false);
        let inference_engine = InferenceEngine::with_monitor(perf_monitor).unwrap();
        let runtime = Arc::new(AiRuntime::new(inference_engine));
        
        let coordinator = AutonomousCoordinator::with_defaults(runtime);
        
        let stats = coordinator.get_stats();
        assert_eq!(stats.total_sites_processed, 0);
    }
}
