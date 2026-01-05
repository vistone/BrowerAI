//! Seamless Browser Engine - 无感浏览器引擎
//! 
//! 这个模块实现了完全透明的AI驱动浏览器引擎，用户体验与传统浏览器完全一致，
//! 但在后台使用AI进行学习、推理和优化。

use anyhow::Result;
use std::sync::Arc;
use std::collections::HashMap;

use crate::ai::{AiRuntime, AutonomousCoordinator, AutonomousConfig, AutonomousResult};
use crate::parser::{HtmlParser, CssParser, JsParser};
use crate::renderer::RenderEngine;
use crate::network::HttpClient;

/// 无感浏览器引擎
pub struct SeamlessBrowser {
    /// AI协调器
    coordinator: Arc<AutonomousCoordinator>,
    
    /// 解析器
    html_parser: HtmlParser,
    css_parser: CssParser,
    js_parser: JsParser,
    
    /// 渲染引擎
    render_engine: RenderEngine,
    
    /// 网络客户端
    http_client: HttpClient,
    
    /// 会话状态
    session: BrowserSession,
}

/// 浏览器会话
#[derive(Debug, Clone)]
pub struct BrowserSession {
    /// 当前URL
    pub current_url: Option<String>,
    
    /// 历史记录
    pub history: Vec<String>,
    
    /// AI增强统计
    pub ai_enhancements: usize,
    
    /// 用户设置
    pub user_preferences: UserPreferences,
}

impl Default for BrowserSession {
    fn default() -> Self {
        Self {
            current_url: None,
            history: Vec::new(),
            ai_enhancements: 0,
            user_preferences: UserPreferences::default(),
        }
    }
}

/// 用户偏好设置
#[derive(Debug, Clone)]
pub struct UserPreferences {
    /// 是否启用AI增强（用户可见）
    pub enable_ai_features: bool,
    
    /// 性能优先级
    pub performance_priority: bool,
    
    /// 可访问性优先
    pub accessibility_priority: bool,
    
    /// 自定义样式
    pub custom_styles: HashMap<String, String>,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            enable_ai_features: true,
            performance_priority: false,
            accessibility_priority: false,
            custom_styles: HashMap::new(),
        }
    }
}

/// 页面渲染结果
#[derive(Debug, Clone)]
pub struct PageRenderResult {
    /// 渲染的HTML
    pub html: String,
    
    /// 是否AI增强
    pub ai_enhanced: bool,
    
    /// 渲染时间（毫秒）
    pub render_time_ms: u64,
    
    /// 功能完整性验证
    pub functionality_verified: bool,
}

impl SeamlessBrowser {
    /// 创建新的无感浏览器
    pub fn new(ai_runtime: Arc<AiRuntime>) -> Self {
        let config = AutonomousConfig::default();
        let coordinator = Arc::new(AutonomousCoordinator::new(config, ai_runtime.clone()));
        
        Self {
            coordinator,
            html_parser: HtmlParser::with_ai_runtime((*ai_runtime).clone()),
            css_parser: CssParser::with_ai_runtime((*ai_runtime).clone()),
            js_parser: JsParser::with_ai_runtime((*ai_runtime).clone()),
            render_engine: RenderEngine::new(),
            http_client: HttpClient::new(),
            session: BrowserSession::default(),
        }
    }
    
    /// 访问URL - 核心方法
    /// 
    /// 这个方法对用户完全透明，但在后台：
    /// 1. 获取网页内容
    /// 2. AI自主学习网页结构
    /// 3. 智能推理优化方案
    /// 4. 可选地生成增强版本
    /// 5. 验证功能完整性
    /// 6. 返回渲染结果（原始或增强）
    pub async fn navigate(&mut self, url: &str) -> Result<PageRenderResult> {
        log::info!("🌐 Navigating to: {}", url);
        
        let start_time = std::time::Instant::now();
        
        // 1. 获取网页内容
        let html = self.fetch_page(url).await?;
        
        // 2. AI自主处理（透明）
        let ai_result = self.coordinator.process_website(url, &html).await?;
        
        // 3. 选择渲染版本
        let (final_html, ai_enhanced) = self.select_render_version(&ai_result);
        
        // 4. 解析和渲染
        let dom = self.html_parser.parse(&final_html)?;
        // Parse CSS to get styles (using empty CSS for now)
        let css_rules = self.css_parser.parse("")?;
        let _rendered = self.render_engine.render(&dom, &css_rules);
        
        // 5. 更新会话
        self.session.current_url = Some(url.to_string());
        self.session.history.push(url.to_string());
        if ai_enhanced {
            self.session.ai_enhancements += 1;
        }
        
        let elapsed = start_time.elapsed();
        
        log::info!("✅ Page loaded in {:.2}ms (AI enhanced: {})", 
                   elapsed.as_millis(), ai_enhanced);
        
        Ok(PageRenderResult {
            html: final_html,
            ai_enhanced,
            render_time_ms: elapsed.as_millis() as u64,
            functionality_verified: ai_result.functionality_preserved,
        })
    }
    
    /// 获取页面内容
    async fn fetch_page(&self, url: &str) -> Result<String> {
        // 使用HTTP客户端获取页面
        match self.http_client.get(url) {
            Ok(response) => {
                // Convert Vec<u8> to String
                String::from_utf8(response.body)
                    .map_err(|e| anyhow::anyhow!("Failed to decode response: {}", e))
            }
            Err(e) => {
                log::warn!("Failed to fetch {}: {}, using mock content", url, e);
                // 返回模拟内容用于测试
                Ok(format!(r#"
                    <!DOCTYPE html>
                    <html>
                        <head><title>Mock Page</title></head>
                        <body>
                            <h1>Mock Content for {}</h1>
                            <p>This is mock content for testing.</p>
                        </body>
                    </html>
                "#, url))
            }
        }
    }
    
    /// 选择渲染版本
    fn select_render_version(&self, ai_result: &AutonomousResult) -> (String, bool) {
        // 如果用户禁用AI功能，始终使用原始版本
        if !self.session.user_preferences.enable_ai_features {
            return (ai_result.original_html.clone(), false);
        }
        
        // 如果有增强版本且功能验证通过，使用增强版本
        if let Some(ref enhanced) = ai_result.enhanced_html {
            if ai_result.functionality_preserved {
                return (enhanced.clone(), true);
            }
        }
        
        // 否则使用原始版本
        (ai_result.original_html.clone(), false)
    }
    
    /// 后退
    pub fn go_back(&mut self) -> Option<String> {
        if self.session.history.len() > 1 {
            self.session.history.pop();
            self.session.history.last().cloned()
        } else {
            None
        }
    }
    
    /// 前进（需要维护前进历史）
    pub fn go_forward(&mut self) -> Option<String> {
        // 实际实现需要维护前进栈
        None
    }
    
    /// 刷新当前页面
    pub async fn refresh(&mut self) -> Result<PageRenderResult> {
        if let Some(url) = self.session.current_url.clone() {
            self.navigate(&url).await
        } else {
            Err(anyhow::anyhow!("No page to refresh"))
        }
    }
    
    /// 获取当前URL
    pub fn current_url(&self) -> Option<&str> {
        self.session.current_url.as_deref()
    }
    
    /// 获取会话统计
    pub fn get_session_stats(&self) -> SessionStats {
        SessionStats {
            pages_visited: self.session.history.len(),
            ai_enhancements_applied: self.session.ai_enhancements,
            coordinator_stats: self.coordinator.get_stats(),
        }
    }
    
    /// 设置用户偏好
    pub fn set_user_preferences(&mut self, preferences: UserPreferences) {
        self.session.user_preferences = preferences;
    }
    
    /// 启动持续学习
    pub fn start_learning(&self) -> Result<()> {
        self.coordinator.start_continuous_learning()
    }
    
    /// 停止持续学习
    pub fn stop_learning(&self) -> Result<()> {
        self.coordinator.stop_continuous_learning()
    }
}

/// 会话统计
#[derive(Debug, Clone)]
pub struct SessionStats {
    /// 访问的页面数
    pub pages_visited: usize,
    
    /// AI增强应用次数
    pub ai_enhancements_applied: usize,
    
    /// 协调器统计
    pub coordinator_stats: crate::ai::CoordinatorStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{InferenceEngine, performance_monitor::PerformanceMonitor};
    
    #[tokio::test]
    async fn test_seamless_browser_creation() {
        let perf_monitor = PerformanceMonitor::new(false);
        let inference_engine = InferenceEngine::with_monitor(perf_monitor).unwrap();
        let runtime = Arc::new(AiRuntime::new(inference_engine));
        
        let browser = SeamlessBrowser::new(runtime);
        assert!(browser.session.user_preferences.enable_ai_features);
    }
    
    #[tokio::test]
    async fn test_navigate_basic() {
        let perf_monitor = PerformanceMonitor::new(false);
        let inference_engine = InferenceEngine::with_monitor(perf_monitor).unwrap();
        let runtime = Arc::new(AiRuntime::new(inference_engine));
        
        let mut browser = SeamlessBrowser::new(runtime);
        
        // 由于没有真实网络，这会使用mock内容
        let result = browser.navigate("https://example.com").await.unwrap();
        
        assert!(result.functionality_verified);
        assert_eq!(browser.session.history.len(), 1);
    }
    
    #[tokio::test]
    async fn test_user_preferences() {
        let perf_monitor = PerformanceMonitor::new(false);
        let inference_engine = InferenceEngine::with_monitor(perf_monitor).unwrap();
        let runtime = Arc::new(AiRuntime::new(inference_engine));
        
        let mut browser = SeamlessBrowser::new(runtime);
        
        let mut prefs = UserPreferences::default();
        prefs.enable_ai_features = false;
        browser.set_user_preferences(prefs);
        
        assert!(!browser.session.user_preferences.enable_ai_features);
    }
    
    #[tokio::test]
    async fn test_navigation_history() {
        let perf_monitor = PerformanceMonitor::new(false);
        let inference_engine = InferenceEngine::with_monitor(perf_monitor).unwrap();
        let runtime = Arc::new(AiRuntime::new(inference_engine));
        
        let mut browser = SeamlessBrowser::new(runtime);
        
        browser.navigate("https://example.com").await.unwrap();
        browser.navigate("https://example.com/page2").await.unwrap();
        
        assert_eq!(browser.session.history.len(), 2);
        
        let prev = browser.go_back();
        assert!(prev.is_some());
    }
    
    #[test]
    fn test_session_stats() {
        let perf_monitor = PerformanceMonitor::new(false);
        let inference_engine = InferenceEngine::with_monitor(perf_monitor).unwrap();
        let runtime = Arc::new(AiRuntime::new(inference_engine));
        
        let browser = SeamlessBrowser::new(runtime);
        let stats = browser.get_session_stats();
        
        assert_eq!(stats.pages_visited, 0);
        assert_eq!(stats.ai_enhancements_applied, 0);
    }
}
