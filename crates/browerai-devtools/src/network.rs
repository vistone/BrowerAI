//! Network Monitor - 网络监控
//!
//! 监控网络请求和响应：
//! - HTTP请求追踪
//! - 响应时间统计
//! - 请求/响应头查看
//! - 缓存分析

use std::collections::VecDeque;
use std::time::Duration;

/// 网络监控器
#[derive(Debug, Clone)]
pub struct NetworkMonitor {
    /// 配置
    config: NetworkConfig,
    /// 请求列表
    requests: VecDeque<NetworkRequest>,
    /// 响应列表
    responses: VecDeque<NetworkResponse>,
    /// 请求计数器
    request_counter: u64,
}

impl NetworkMonitor {
    /// 创建新的网络监控器
    pub fn new() -> Self {
        Self {
            config: NetworkConfig::default(),
            requests: VecDeque::new(),
            responses: VecDeque::new(),
            request_counter: 0,
        }
    }

    /// 使用配置创建监控器
    pub fn with_config(config: NetworkConfig) -> Self {
        Self {
            config,
            requests: VecDeque::new(),
            responses: VecDeque::new(),
            request_counter: 0,
        }
    }

    /// 记录请求
    pub fn record_request(&mut self, url: impl Into<String>, method: impl Into<String>) -> u64 {
        self.request_counter += 1;
        let id = self.request_counter;

        let request = NetworkRequest {
            id,
            url: url.into(),
            method: method.into(),
            headers: Vec::new(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            body_size: None,
        };

        self.requests.push_back(request);

        // 限制数量
        while self.requests.len() > self.config.max_requests {
            self.requests.pop_front();
        }

        id
    }

    /// 记录响应
    pub fn record_response(&mut self, request_id: u64, status: u16) {
        let response = NetworkResponse {
            request_id,
            status,
            headers: Vec::new(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            duration_ms: None,
            body_size: None,
            cached: false,
        };

        self.responses.push_back(response);

        // 限制数量
        while self.responses.len() > self.config.max_requests {
            self.responses.pop_front();
        }
    }

    /// 完成请求（记录响应时间）
    pub fn complete_request(&mut self, request_id: u64, status: u16, duration: Duration) {
        self.record_response(request_id, status);

        if let Some(response) = self
            .responses
            .iter_mut()
            .find(|r| r.request_id == request_id)
        {
            response.duration_ms = Some(duration.as_millis() as u64);
        }
    }

    /// 获取所有请求
    pub fn requests(&self) -> &VecDeque<NetworkRequest> {
        &self.requests
    }

    /// 获取所有响应
    pub fn responses(&self) -> &VecDeque<NetworkResponse> {
        &self.responses
    }

    /// 获取请求数量
    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    /// 获取响应数量
    pub fn response_count(&self) -> usize {
        self.responses.len()
    }

    /// 获取特定请求
    pub fn get_request(&self, id: u64) -> Option<&NetworkRequest> {
        self.requests.iter().find(|r| r.id == id)
    }

    /// 获取特定响应
    pub fn get_response(&self, request_id: u64) -> Option<&NetworkResponse> {
        self.responses.iter().find(|r| r.request_id == request_id)
    }

    /// 获取平均响应时间
    pub fn average_response_time(&self) -> Option<Duration> {
        let durations: Vec<_> = self
            .responses
            .iter()
            .filter_map(|r| r.duration_ms.map(Duration::from_millis))
            .collect();

        if durations.is_empty() {
            return None;
        }

        let total: Duration = durations.iter().sum();
        Some(total / durations.len() as u32)
    }

    /// 按状态码统计
    pub fn status_code_stats(&self) -> std::collections::HashMap<u16, usize> {
        let mut stats = std::collections::HashMap::new();

        for response in &self.responses {
            *stats.entry(response.status).or_insert(0) += 1;
        }

        stats
    }

    /// 获取总传输大小
    pub fn total_transfer_size(&self) -> usize {
        self.responses.iter().filter_map(|r| r.body_size).sum()
    }

    /// 清空所有数据
    pub fn clear(&mut self) {
        self.requests.clear();
        self.responses.clear();
        self.request_counter = 0;
    }

    /// 获取配置
    pub fn config(&self) -> &NetworkConfig {
        &self.config
    }
}

impl Default for NetworkMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// 网络请求
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NetworkRequest {
    /// 请求ID
    pub id: u64,
    /// URL
    pub url: String,
    /// HTTP方法
    pub method: String,
    /// 请求头
    pub headers: Vec<(String, String)>,
    /// 时间戳（毫秒，从UNIX纪元）
    pub timestamp_ms: u64,
    /// 请求体大小
    pub body_size: Option<usize>,
}

impl NetworkRequest {
    /// 创建新的请求
    pub fn new(id: u64, url: impl Into<String>, method: impl Into<String>) -> Self {
        Self {
            id,
            url: url.into(),
            method: method.into(),
            headers: Vec::new(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            body_size: None,
        }
    }

    /// 添加请求头
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((key.into(), value.into()));
        self
    }
}

/// 网络响应
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NetworkResponse {
    /// 对应的请求ID
    pub request_id: u64,
    /// HTTP状态码
    pub status: u16,
    /// 响应头
    pub headers: Vec<(String, String)>,
    /// 时间戳（毫秒，从UNIX纪元）
    pub timestamp_ms: u64,
    /// 响应时间（毫秒）
    pub duration_ms: Option<u64>,
    /// 响应体大小
    pub body_size: Option<usize>,
    /// 是否来自缓存
    pub cached: bool,
}

impl NetworkResponse {
    /// 创建新的响应
    pub fn new(request_id: u64, status: u16) -> Self {
        Self {
            request_id,
            status,
            headers: Vec::new(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            duration_ms: None,
            body_size: None,
            cached: false,
        }
    }

    /// 是否成功
    pub fn is_success(&self) -> bool {
        self.status >= 200 && self.status < 300
    }

    /// 是否重定向
    pub fn is_redirect(&self) -> bool {
        self.status >= 300 && self.status < 400
    }

    /// 是否客户端错误
    pub fn is_client_error(&self) -> bool {
        self.status >= 400 && self.status < 500
    }

    /// 是否服务器错误
    pub fn is_server_error(&self) -> bool {
        self.status >= 500 && self.status < 600
    }
}

/// 网络配置
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// 最大请求记录数
    pub max_requests: usize,
    /// 捕获请求体
    pub capture_request_body: bool,
    /// 捕获响应体
    pub capture_response_body: bool,
    /// 启用缓存分析
    pub enable_cache_analysis: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            max_requests: 1000,
            capture_request_body: false,
            capture_response_body: false,
            enable_cache_analysis: true,
        }
    }
}

/// 请求类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestType {
    /// 文档
    Document,
    /// 样式表
    Stylesheet,
    /// 脚本
    Script,
    /// 图像
    Image,
    /// 字体
    Font,
    /// XHR/Fetch
    XHR,
    /// WebSocket
    WebSocket,
    /// 其他
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_network_monitor_creation() {
        let monitor = NetworkMonitor::new();
        assert_eq!(monitor.request_count(), 0);
    }

    #[test]
    fn test_record_request() {
        let mut monitor = NetworkMonitor::new();
        let id = monitor.record_request("https://example.com", "GET");

        assert_eq!(id, 1);
        assert_eq!(monitor.request_count(), 1);

        let request = monitor.get_request(id).unwrap();
        assert_eq!(request.url, "https://example.com");
        assert_eq!(request.method, "GET");
    }

    #[test]
    fn test_record_response() {
        let mut monitor = NetworkMonitor::new();
        let id = monitor.record_request("https://example.com", "GET");
        monitor.record_response(id, 200);

        assert_eq!(monitor.response_count(), 1);

        let response = monitor.get_response(id).unwrap();
        assert_eq!(response.status, 200);
        assert!(response.is_success());
    }

    #[test]
    fn test_complete_request() {
        let mut monitor = NetworkMonitor::new();
        let id = monitor.record_request("https://example.com", "GET");

        thread::sleep(Duration::from_millis(10));
        monitor.complete_request(id, 200, Duration::from_millis(10));

        let response = monitor.get_response(id).unwrap();
        assert_eq!(response.duration_ms, Some(10));
    }

    #[test]
    fn test_average_response_time() {
        let mut monitor = NetworkMonitor::new();

        for i in 0..3 {
            let id = monitor.record_request(format!("https://example.com/{}", i), "GET");
            monitor.complete_request(id, 200, Duration::from_millis(10 * (i + 1) as u64));
        }

        let avg = monitor.average_response_time().unwrap();
        assert_eq!(avg, Duration::from_millis(20)); // (10 + 20 + 30) / 3 = 20
    }

    #[test]
    fn test_status_code_stats() {
        let mut monitor = NetworkMonitor::new();

        let id1 = monitor.record_request("https://example.com/1", "GET");
        monitor.record_response(id1, 200);

        let id2 = monitor.record_request("https://example.com/2", "GET");
        monitor.record_response(id2, 404);

        let id3 = monitor.record_request("https://example.com/3", "GET");
        monitor.record_response(id3, 200);

        let stats = monitor.status_code_stats();
        assert_eq!(stats.get(&200), Some(&2));
        assert_eq!(stats.get(&404), Some(&1));
    }

    #[test]
    fn test_response_status_checks() {
        let success = NetworkResponse::new(1, 200);
        assert!(success.is_success());
        assert!(!success.is_client_error());

        let redirect = NetworkResponse::new(1, 301);
        assert!(redirect.is_redirect());

        let client_error = NetworkResponse::new(1, 404);
        assert!(client_error.is_client_error());

        let server_error = NetworkResponse::new(1, 500);
        assert!(server_error.is_server_error());
    }

    #[test]
    fn test_max_requests_limit() {
        let mut monitor = NetworkMonitor::with_config(NetworkConfig {
            max_requests: 5,
            ..Default::default()
        });

        for i in 0..10 {
            monitor.record_request(format!("https://example.com/{}", i), "GET");
        }

        assert_eq!(monitor.request_count(), 5);
    }

    #[test]
    fn test_clear() {
        let mut monitor = NetworkMonitor::new();

        let id = monitor.record_request("https://example.com", "GET");
        monitor.record_response(id, 200);

        monitor.clear();

        assert_eq!(monitor.request_count(), 0);
        assert_eq!(monitor.response_count(), 0);
    }
}
