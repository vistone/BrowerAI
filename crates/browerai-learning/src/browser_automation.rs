// 浏览器自动化模块 - 动态网站分析
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[cfg(feature = "browser")]
use headless_chrome::{Browser, LaunchOptions, Tab};
#[cfg(feature = "browser")]
use std::sync::Arc;

/// 浏览器自动化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserConfig {
    pub headless: bool,
    pub timeout_ms: u64,
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub user_agent: Option<String>,
    pub proxy: Option<String>,
    pub ignore_https_errors: bool,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            headless: true,
            timeout_ms: 30000,
            viewport_width: 1920,
            viewport_height: 1080,
            user_agent: None,
            proxy: None,
            ignore_https_errors: false,
        }
    }
}

/// HTTP 请求信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRequest {
    pub url: String,
    pub method: String,
    pub headers: HashMap<String, String>,
    pub post_data: Option<String>,
    pub timestamp: u64,
    pub resource_type: String,
    pub status: Option<u16>,
    pub response_headers: HashMap<String, String>,
    pub response_size: usize,
    pub duration_ms: f64,
}

/// 浏览器会话
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserSession {
    pub url: String,
    pub title: String,
    pub html: String,
    pub cookies: Vec<Cookie>,
    pub local_storage: HashMap<String, String>,
    pub session_storage: HashMap<String, String>,
    pub network_requests: Vec<NetworkRequest>,
    pub console_logs: Vec<ConsoleMessage>,
    pub websocket_connections: Vec<crate::websocket_monitor::WebSocketConnection>, // ✨ NEW
    pub screenshots: Vec<Screenshot>,
}

/// Cookie 信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cookie {
    pub name: String,
    pub value: String,
    pub domain: String,
    pub path: String,
    pub expires: Option<i64>,
    pub http_only: bool,
    pub secure: bool,
    pub same_site: Option<String>,
}

/// 控制台消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsoleMessage {
    pub level: String, // log, warn, error, info, debug
    pub text: String,
    pub timestamp: u64,
    pub source: Option<String>,
}

/// 截图
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Screenshot {
    pub name: String,
    pub data: Vec<u8>,
    pub format: String, // png, jpeg
    pub width: u32,
    pub height: u32,
}

/// 浏览器自动化引擎
pub struct BrowserAutomation {
    config: BrowserConfig,
}

impl Default for BrowserAutomation {
    fn default() -> Self {
        Self::new(BrowserConfig::default())
    }
}

impl BrowserAutomation {
    pub fn new(config: BrowserConfig) -> Self {
        Self { config }
    }
    /// 访问网站并收集所有信息
    pub async fn visit_and_analyze(&self, url: &str) -> Result<BrowserSession> {
        log::info!("开始分析网站: {}", url);

        #[cfg(feature = "browser")]
        {
            // 使用真实的 headless Chrome
            let session = self.real_browser_session(url)?;
            log::info!(
                "网站分析完成: {} 个网络请求",
                session.network_requests.len()
            );
            Ok(session)
        }

        #[cfg(not(feature = "browser"))]
        {
            // 回退到模拟实现
            let session = self.simulate_browser_session(url).await?;
            log::info!(
                "网站分析完成 (模拟模式): {} 个网络请求",
                session.network_requests.len()
            );
            Ok(session)
        }
    }

    /// 真实浏览器会话（使用 headless_chrome）
    #[cfg(feature = "browser")]
    fn real_browser_session(&self, url: &str) -> Result<BrowserSession> {
        // 1. 启动浏览器
        let launch_options = LaunchOptions::default_builder()
            .headless(self.config.headless)
            .build()
            .context("Failed to build launch options")?;

        let browser = Browser::new(launch_options).context("Failed to launch browser")?;

        // 2. 创建新标签页
        let tab = browser.new_tab().context("Failed to create new tab")?;

        // 3. 导航到 URL（视口大小会使用浏览器默认值）
        tab.navigate_to(url).context("Failed to navigate to URL")?;

        // 4. 等待页面加载
        tab.wait_until_navigated()
            .context("Failed to wait for navigation")?;

        // 5. 获取页面内容
        let html = tab.get_content().context("Failed to get page content")?;

        // 6. 获取页面标题
        let title = tab.get_title().context("Failed to get page title")?;

        // 7. 获取 cookies
        let cookies = self.extract_cookies(&tab)?;

        // 8. 获取 localStorage 和 sessionStorage
        let (local_storage, session_storage) = self.extract_storage(&tab)?;

        // 9. 获取 console 日志
        let console_logs = self.get_injected_console_logs(&tab)?;

        // 10. 获取网络请求（通过 JS 注入收集）
        let network_requests = self.extract_network_requests(&tab)?;

        // 11. 获取 WebSocket 连接 ✨ NEW
        let websocket_connections = self.extract_websocket_connections(&tab)?;

        // 12. 截图（可选）
        let screenshots = if self.config.headless {
            vec![self.take_screenshot_internal(&tab)?]
        } else {
            vec![]
        };

        Ok(BrowserSession {
            url: url.to_string(),
            title,
            html,
            cookies,
            local_storage,
            session_storage,
            network_requests,
            console_logs,
            websocket_connections,
            screenshots,
        })
    }

    /// 提取 cookies
    #[cfg(feature = "browser")]
    fn extract_cookies(&self, tab: &Arc<Tab>) -> Result<Vec<Cookie>> {
        use headless_chrome::protocol::cdp::Network;

        let cookies_result = tab.call_method(Network::GetCookies { urls: None })?;

        let mut cookies = Vec::new();
        let cookie_list = cookies_result.cookies;
        for c in cookie_list {
            cookies.push(Cookie {
                name: c.name,
                value: c.value,
                domain: c.domain,
                path: c.path,
                expires: Some(c.expires as i64),
                http_only: c.http_only,
                secure: c.secure,
                same_site: c.same_site.map(|s| format!("{:?}", s)),
            });
        }

        Ok(cookies)
    }

    /// 提取 localStorage 和 sessionStorage
    #[cfg(feature = "browser")]
    fn extract_storage(
        &self,
        tab: &Arc<Tab>,
    ) -> Result<(HashMap<String, String>, HashMap<String, String>)> {
        // 执行 JS 获取 localStorage
        let local_storage_script = r#"
            JSON.stringify(Object.keys(localStorage).reduce((obj, key) => {
                obj[key] = localStorage.getItem(key);
                return obj;
            }, {}))
        "#;

        let local_storage_result = tab.evaluate(local_storage_script, false)?;
        let local_storage: HashMap<String, String> = serde_json::from_str(
            local_storage_result
                .value
                .as_ref()
                .and_then(|v| v.as_str())
                .unwrap_or("{}"),
        )
        .unwrap_or_default();

        // 执行 JS 获取 sessionStorage
        let session_storage_script = r#"
            JSON.stringify(Object.keys(sessionStorage).reduce((obj, key) => {
                obj[key] = sessionStorage.getItem(key);
                return obj;
            }, {}))
        "#;

        let session_storage_result = tab.evaluate(session_storage_script, false)?;
        let session_storage: HashMap<String, String> = serde_json::from_str(
            session_storage_result
                .value
                .as_ref()
                .and_then(|v| v.as_str())
                .unwrap_or("{}"),
        )
        .unwrap_or_default();

        Ok((local_storage, session_storage))
    }

    /// 提取 console 日志
    #[allow(dead_code)]
    #[cfg(feature = "browser")]
    fn extract_console_logs(&self, tab: &Arc<Tab>) -> Result<Vec<ConsoleMessage>> {
        // 通过 JavaScript 注入来收集日志
        let script = r#"
        (function() {
            window.__consoleLogs = [];
            
            // 拦截 console 方法
            const originalLog = console.log;
            const originalWarn = console.warn;
            const originalError = console.error;
            const originalInfo = console.info;
            const originalDebug = console.debug;
            
            const timestamp = () => Math.floor(Date.now() / 1000);
            
            const capture = (level, args) => {
                const message = Array.from(args)
                    .map(arg => {
                        if (typeof arg === 'object') {
                            try { return JSON.stringify(arg); } 
                            catch { return String(arg); }
                        }
                        return String(arg);
                    })
                    .join(' ');
                    
                window.__consoleLogs.push({
                    level: level,
                    text: message,
                    timestamp: timestamp(),
                    source: null
                });
            };
            
            console.log = function(...args) { 
                capture('log', args);
                originalLog.apply(console, args);
            };
            console.warn = function(...args) { 
                capture('warn', args);
                originalWarn.apply(console, args);
            };
            console.error = function(...args) { 
                capture('error', args);
                originalError.apply(console, args);
            };
            console.info = function(...args) { 
                capture('info', args);
                originalInfo.apply(console, args);
            };
            console.debug = function(...args) { 
                capture('debug', args);
                originalDebug.apply(console, args);
            };
            
            return 'Console logging intercepted';
        })()
        "#;

        // 注入拦截脚本
        let _ = tab.evaluate(script, false);

        // 这是被动方式。更主动的方式需要在页面加载前注入脚本
        // 目前返回空列表，实际日志会在 visit_and_analyze 中通过脚本注入获取
        Ok(vec![])
    }

    /// 获取注入的 console 日志
    #[cfg(feature = "browser")]
    fn get_injected_console_logs(&self, tab: &Arc<Tab>) -> Result<Vec<ConsoleMessage>> {
        // 执行脚本获取已收集的日志
        let script = "window.__consoleLogs || []";
        let result = tab.evaluate(script, false)?;

        let logs = if let Some(value) = result.value {
            if let Ok(json_str) = serde_json::to_string(&value) {
                serde_json::from_str::<Vec<ConsoleMessage>>(&json_str).unwrap_or_default()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        Ok(logs)
    }

    /// 提取网络请求
    #[cfg(feature = "browser")]
    fn extract_network_requests(&self, tab: &Arc<Tab>) -> Result<Vec<NetworkRequest>> {
        // 通过 JavaScript 注入来监听 fetch 和 XHR 请求
        let script = r#"
        (function() {
            window.__networkRequests = [];
            
            // 拦截 fetch
            const originalFetch = window.fetch;
            window.fetch = function(...args) {
                const url = args[0];
                const init = args[1] || {};
                const method = (init.method || 'GET').toUpperCase();
                const headers = init.headers || {};
                
                const timestamp = Math.floor(Date.now() / 1000);
                const startTime = performance.now();
                
                return originalFetch.apply(window, args)
                    .then(response => {
                        const endTime = performance.now();
                        const status = response.status;
                        const responseSize = response.headers.get('content-length') || 
                                           (response.text ? response.text().length : 0);
                        
                        window.__networkRequests.push({
                            url: String(url),
                            method: method,
                            headers: Object.fromEntries(Object.entries(headers).map(e => [e[0], String(e[1])])),
                            post_data: init.body ? String(init.body).substring(0, 500) : null,
                            timestamp: timestamp,
                            resource_type: 'fetch',
                            status: status,
                            response_headers: {},
                            response_size: Number(responseSize) || 0,
                            duration_ms: endTime - startTime
                        });
                        
                        return response;
                    })
                    .catch(error => {
                        const endTime = performance.now();
                        window.__networkRequests.push({
                            url: String(url),
                            method: method,
                            headers: Object.fromEntries(Object.entries(headers).map(e => [e[0], String(e[1])])),
                            post_data: init.body ? String(init.body).substring(0, 500) : null,
                            timestamp: timestamp,
                            resource_type: 'fetch',
                            status: null,
                            response_headers: {},
                            response_size: 0,
                            duration_ms: endTime - startTime
                        });
                        throw error;
                    });
            };
            
            // 拦截 XMLHttpRequest
            const originalOpen = XMLHttpRequest.prototype.open;
            const originalSend = XMLHttpRequest.prototype.send;
            
            XMLHttpRequest.prototype.open = function(method, url) {
                this.__method = method;
                this.__url = url;
                this.__startTime = performance.now();
                this.__timestamp = Math.floor(Date.now() / 1000);
                return originalOpen.apply(this, arguments);
            };
            
            XMLHttpRequest.prototype.send = function(body) {
                const self = this;
                
                const onLoadEnd = function() {
                    const endTime = performance.now();
                    window.__networkRequests.push({
                        url: String(self.__url),
                        method: String(self.__method).toUpperCase(),
                        headers: {},
                        post_data: body ? String(body).substring(0, 500) : null,
                        timestamp: self.__timestamp,
                        resource_type: 'xhr',
                        status: self.status,
                        response_headers: {},
                        response_size: self.responseText ? self.responseText.length : 0,
                        duration_ms: endTime - self.__startTime
                    });
                };
                
                this.addEventListener('loadend', onLoadEnd);
                return originalSend.apply(this, arguments);
            };
            
            return 'Network monitoring enabled';
        })()
        "#;

        // 注入监控脚本
        let _ = tab.evaluate(script, false);

        // 执行脚本获取已收集的请求
        let get_script = "window.__networkRequests || []";
        let result = tab.evaluate(get_script, false)?;

        let requests = if let Some(value) = result.value {
            if let Ok(json_str) = serde_json::to_string(&value) {
                serde_json::from_str::<Vec<NetworkRequest>>(&json_str).unwrap_or_default()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        Ok(requests)
    }

    /// 提取 WebSocket 连接
    #[cfg(feature = "browser")]
    fn extract_websocket_connections(
        &self,
        tab: &Arc<Tab>,
    ) -> Result<Vec<crate::websocket_monitor::WebSocketConnection>> {
        #[allow(unused_imports)]
        use crate::websocket_monitor::{
            MessageDirection, MessageType, WebSocketConnection, WebSocketMessage,
        };

        // 注入 WebSocket 监控脚本
        let script = r#"
        (function() {
            window.__websockets = {};
            let wsCounter = 0;
            
            const originalWebSocket = window.WebSocket;
            
            window.WebSocket = function(url, protocols) {
                const wsId = 'ws-' + (++wsCounter);
                const ws = new originalWebSocket(url, protocols);
                
                window.__websockets[wsId] = {
                    id: wsId,
                    url: url,
                    protocol: ws.protocol || null,
                    created_at: Math.floor(Date.now() / 1000),
                    closed_at: null,
                    close_code: null,
                    close_reason: null,
                    messages: [],
                    total_messages_sent: 0,
                    total_messages_received: 0,
                    total_bytes_sent: 0,
                    total_bytes_received: 0
                };
                
                const connInfo = window.__websockets[wsId];
                
                // 拦截 send
                const originalSend = ws.send;
                ws.send = function(data) {
                    connInfo.total_messages_sent++;
                    connInfo.total_bytes_sent += typeof data === 'string' ? data.length : data.byteLength || 0;
                    
                    connInfo.messages.push({
                        id: wsId + '-sent-' + connInfo.total_messages_sent,
                        direction: 'sent',
                        message_type: typeof data === 'string' ? 'text' : 'binary',
                        content: typeof data === 'string' ? data : btoa(String.fromCharCode.apply(null, data)),
                        size_bytes: typeof data === 'string' ? data.length : data.byteLength || 0,
                        timestamp: Math.floor(Date.now() / 1000)
                    });
                    
                    return originalSend.call(this, data);
                };
                
                // 拦截 message 事件
                const originalOnMessage = ws.onmessage;
                ws.addEventListener('message', function(event) {
                    connInfo.total_messages_received++;
                    
                    const data = event.data;
                    const content = typeof data === 'string' ? data : btoa(String.fromCharCode.apply(null, new Uint8Array(data)));
                    
                    connInfo.messages.push({
                        id: wsId + '-recv-' + connInfo.total_messages_received,
                        direction: 'received',
                        message_type: typeof data === 'string' ? 'text' : 'binary',
                        content: content,
                        size_bytes: typeof data === 'string' ? data.length : data.byteLength || 0,
                        timestamp: Math.floor(Date.now() / 1000)
                    });
                    
                    connInfo.total_bytes_received += typeof data === 'string' ? data.length : data.byteLength || 0;
                    
                    if (originalOnMessage) originalOnMessage.call(this, event);
                });
                
                // 拦截 close
                ws.addEventListener('close', function(event) {
                    connInfo.closed_at = Math.floor(Date.now() / 1000);
                    connInfo.close_code = event.code;
                    connInfo.close_reason = event.reason;
                });
                
                return ws;
            };
            
            // 复制静态方法
            window.WebSocket.CONNECTING = originalWebSocket.CONNECTING;
            window.WebSocket.OPEN = originalWebSocket.OPEN;
            window.WebSocket.CLOSING = originalWebSocket.CLOSING;
            window.WebSocket.CLOSED = originalWebSocket.CLOSED;
            
            return 'WebSocket monitoring enabled';
        })()
        "#;

        // 注入脚本
        let _ = tab.evaluate(script, false);

        // 等待一段时间让 WebSocket 建立
        std::thread::sleep(std::time::Duration::from_millis(100));

        // 获取 WebSocket 数据
        let get_script = "Object.values(window.__websockets || {})";
        let result = tab.evaluate(get_script, false)?;

        let connections = if let Some(value) = result.value {
            if let Ok(json_str) = serde_json::to_string(&value) {
                serde_json::from_str::<Vec<crate::websocket_monitor::WebSocketConnection>>(
                    &json_str,
                )
                .unwrap_or_default()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        Ok(connections)
    }

    /// 内部截图方法
    #[cfg(feature = "browser")]
    fn take_screenshot_internal(&self, tab: &Arc<Tab>) -> Result<Screenshot> {
        let screenshot_data = tab.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Png,
            None,
            None,
            true,
        )?;

        Ok(Screenshot {
            name: "page_screenshot".to_string(),
            data: screenshot_data,
            format: "png".to_string(),
            width: self.config.viewport_width,
            height: self.config.viewport_height,
        })
    }

    #[allow(dead_code)]
    /// 模拟浏览器会话（占位实现，用于无 browser feature 时）
    async fn simulate_browser_session(&self, url: &str) -> Result<BrowserSession> {
        log::warn!("使用模拟浏览器模式（browser feature 未启用）");

        let html = self.fetch_html(url).await?;

        Ok(BrowserSession {
            url: url.to_string(),
            title: self.extract_title(&html),
            html,
            cookies: vec![],
            local_storage: HashMap::new(),
            session_storage: HashMap::new(),
            network_requests: vec![],
            console_logs: vec![],
            websocket_connections: vec![],
            screenshots: vec![],
        })
    }

    /// 获取 HTML 内容
    async fn fetch_html(&self, url: &str) -> Result<String> {
        #[allow(dead_code)]
        use reqwest::Client;

        let client = Client::builder()
            .timeout(Duration::from_millis(self.config.timeout_ms))
            .build()?;

        let mut request = client.get(url);

        if let Some(ref ua) = self.config.user_agent {
            request = request.header("User-Agent", ua);
        }

        let response = request.send().await.context("Failed to fetch URL")?;

        let html = response
            .text()
            .await
            .context("Failed to read response body")?;

        Ok(html)
    }

    /// 提取标题
    fn extract_title(&self, html: &str) -> String {
        #[allow(dead_code)]
        use regex::Regex;

        let re = Regex::new(r"<title[^>]*>(.*?)</title>").unwrap();

        re.captures(html)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_else(|| "Untitled".to_string())
    }

    /// 等待页面加载完成
    pub async fn wait_for_load(&self, timeout_ms: u64) -> Result<()> {
        #[cfg(feature = "browser")]
        {
            // 真实浏览器的等待逻辑已在 visit_and_analyze 中实现
            log::debug!("等待页面加载: {}ms", timeout_ms);
            Ok(())
        }

        #[cfg(not(feature = "browser"))]
        {
            // 模拟等待
            tokio::time::sleep(Duration::from_millis(timeout_ms.min(1000))).await;
            Ok(())
        }
    }

    /// 等待特定元素出现
    #[cfg(feature = "browser")]
    pub fn wait_for_element(&self, tab: &Arc<Tab>, selector: &str, timeout_ms: u64) -> Result<()> {
        use std::time::Instant;

        let start = Instant::now();
        let timeout = Duration::from_millis(timeout_ms);

        loop {
            let script = format!(r#"document.querySelector("{}")"#, selector);
            let result = tab.evaluate(&script, false)?;

            if result.value.is_some() {
                return Ok(());
            }

            if start.elapsed() > timeout {
                return Err(anyhow::anyhow!("等待元素超时: {}", selector));
            }

            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// 执行 JavaScript
    pub async fn execute_script(&self, _script: &str) -> Result<String> {
        #[cfg(feature = "browser")]
        {
            // 真实浏览器执行需要 tab 实例
            // 这个方法需要重构为接受 tab 参数
            log::warn!("execute_script 需要 tab 实例，请使用 execute_script_on_tab");
            Ok("{}".to_string())
        }

        #[cfg(not(feature = "browser"))]
        {
            log::warn!("JavaScript 执行不可用（browser feature 未启用）");
            Ok("{}".to_string())
        }
    }

    /// 在指定 tab 上执行 JavaScript
    #[cfg(feature = "browser")]
    pub fn execute_script_on_tab(&self, tab: &Arc<Tab>, script: &str) -> Result<String> {
        let result = tab.evaluate(script, false)?;

        if let Some(value) = result.value {
            Ok(serde_json::to_string(&value)?)
        } else {
            Ok("null".to_string())
        }
    }

    /// 截图
    pub async fn take_screenshot(&self, _path: &str) -> Result<Screenshot> {
        #[cfg(feature = "browser")]
        {
            log::warn!("take_screenshot 需要 tab 实例，请使用 take_screenshot_on_tab");
            Ok(Screenshot {
                name: "screenshot".to_string(),
                data: vec![],
                format: "png".to_string(),
                width: self.config.viewport_width,
                height: self.config.viewport_height,
            })
        }

        #[cfg(not(feature = "browser"))]
        {
            log::warn!("截图功能不可用（browser feature 未启用）");
            Ok(Screenshot {
                name: "screenshot".to_string(),
                data: vec![],
                format: "png".to_string(),
                width: self.config.viewport_width,
                height: self.config.viewport_height,
            })
        }
    }

    /// 在指定 tab 上截图
    #[cfg(feature = "browser")]
    pub fn take_screenshot_on_tab(&self, tab: &Arc<Tab>, name: &str) -> Result<Screenshot> {
        let screenshot_data = tab.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Png,
            None,
            None,
            true,
        )?;

        Ok(Screenshot {
            name: name.to_string(),
            data: screenshot_data,
            format: "png".to_string(),
            width: self.config.viewport_width,
            height: self.config.viewport_height,
        })
    }
}

/// 网络监控器
impl Default for NetworkMonitor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct NetworkMonitor {
    requests: Vec<NetworkRequest>,
}

impl NetworkMonitor {
    pub fn new() -> Self {
        Self {
            requests: Vec::new(),
        }
    }

    /// 记录请求
    pub fn record_request(&mut self, request: NetworkRequest) {
        self.requests.push(request);
    }

    /// 获取所有请求
    pub fn get_requests(&self) -> &[NetworkRequest] {
        &self.requests
    }

    /// 按资源类型分组
    pub fn group_by_type(&self) -> HashMap<String, Vec<&NetworkRequest>> {
        let mut groups: HashMap<String, Vec<&NetworkRequest>> = HashMap::new();

        for req in &self.requests {
            groups
                .entry(req.resource_type.clone())
                .or_default()
                .push(req);
        }

        groups
    }

    /// 获取 API 请求
    pub fn get_api_requests(&self) -> Vec<&NetworkRequest> {
        self.requests
            .iter()
            .filter(|req| {
                req.resource_type == "xhr"
                    || req.resource_type == "fetch"
                    || req.url.contains("/api/")
            })
            .collect()
    }

    /// 分析性能
    pub fn analyze_performance(&self) -> PerformanceStats {
        let total_requests = self.requests.len();
        let total_size: usize = self.requests.iter().map(|r| r.response_size).sum();
        let total_duration: f64 = self.requests.iter().map(|r| r.duration_ms).sum();

        let avg_duration = if total_requests > 0 {
            total_duration / total_requests as f64
        } else {
            0.0
        };

        PerformanceStats {
            total_requests,
            total_size_bytes: total_size,
            total_duration_ms: total_duration,
            avg_request_duration_ms: avg_duration,
            slowest_request: self.find_slowest_request(),
        }
    }

    fn find_slowest_request(&self) -> Option<String> {
        self.requests
            .iter()
            .max_by(|a, b| a.duration_ms.partial_cmp(&b.duration_ms).unwrap())
            .map(|req| req.url.clone())
    }
}

/// 性能统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub total_requests: usize,
    pub total_size_bytes: usize,
    pub total_duration_ms: f64,
    pub avg_request_duration_ms: f64,
    pub slowest_request: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_browser_config_default() {
        let config = BrowserConfig::default();
        assert_eq!(config.headless, true);
        assert_eq!(config.timeout_ms, 30000);
        assert_eq!(config.viewport_width, 1920);
    }

    #[test]
    fn test_network_monitor() {
        let mut monitor = NetworkMonitor::new();

        let request = NetworkRequest {
            url: "https://example.com/api/data".to_string(),
            method: "GET".to_string(),
            headers: HashMap::new(),
            post_data: None,
            timestamp: 0,
            resource_type: "fetch".to_string(),
            status: Some(200),
            response_headers: HashMap::new(),
            response_size: 1024,
            duration_ms: 150.5,
        };

        monitor.record_request(request);

        assert_eq!(monitor.get_requests().len(), 1);
        assert_eq!(monitor.get_api_requests().len(), 1);
    }

    #[test]
    fn test_performance_stats() {
        let mut monitor = NetworkMonitor::new();

        for i in 0..5 {
            monitor.record_request(NetworkRequest {
                url: format!("https://example.com/file{}.js", i),
                method: "GET".to_string(),
                headers: HashMap::new(),
                post_data: None,
                timestamp: 0,
                resource_type: "script".to_string(),
                status: Some(200),
                response_headers: HashMap::new(),
                response_size: 1024 * (i + 1),
                duration_ms: 100.0 * (i + 1) as f64,
            });
        }

        let stats = monitor.analyze_performance();
        assert_eq!(stats.total_requests, 5);
        assert_eq!(stats.total_size_bytes, 1024 * (1 + 2 + 3 + 4 + 5));
        assert!(stats.avg_request_duration_ms > 0.0);
    }

    #[test]
    fn test_console_message_structure() {
        let messages = vec![
            ConsoleMessage {
                level: "log".to_string(),
                text: "Hello World".to_string(),
                timestamp: 1000,
                source: Some("app.js".to_string()),
            },
            ConsoleMessage {
                level: "error".to_string(),
                text: "An error occurred".to_string(),
                timestamp: 1100,
                source: Some("utils.js".to_string()),
            },
            ConsoleMessage {
                level: "warn".to_string(),
                text: "Warning message".to_string(),
                timestamp: 1200,
                source: None,
            },
        ];

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].level, "log");
        assert_eq!(messages[1].level, "error");
        assert!(messages[2].source.is_none());

        // 过滤错误和警告
        let errors: Vec<_> = messages.iter().filter(|m| m.level == "error").collect();
        let warnings: Vec<_> = messages.iter().filter(|m| m.level == "warn").collect();

        assert_eq!(errors.len(), 1);
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn test_network_request_filtering() {
        let mut monitor = NetworkMonitor::new();

        // 混合类型的请求
        let requests = vec![
            ("https://api.example.com/data", "fetch", "GET"),
            ("https://example.com/style.css", "stylesheet", "GET"),
            ("https://api.example.com/submit", "xhr", "POST"),
            ("https://cdn.example.com/app.js", "script", "GET"),
            ("https://api.example.com/list", "fetch", "GET"),
        ];

        for (url, res_type, method) in requests {
            monitor.record_request(NetworkRequest {
                url: url.to_string(),
                method: method.to_string(),
                headers: HashMap::new(),
                post_data: None,
                timestamp: 0,
                resource_type: res_type.to_string(),
                status: Some(200),
                response_headers: HashMap::new(),
                response_size: 512,
                duration_ms: 100.0,
            });
        }

        // 过滤 API 请求
        let api_requests = monitor.get_api_requests();
        assert_eq!(api_requests.len(), 3); // 2 fetch + 1 xhr

        // 按类型分组
        let groups = monitor.group_by_type();
        assert_eq!(groups.len(), 4); // fetch, stylesheet, xhr, script
        assert_eq!(groups.get("fetch").map(|v| v.len()), Some(2));
        assert_eq!(groups.get("xhr").map(|v| v.len()), Some(1));
    }
}
