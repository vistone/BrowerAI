// WebSocket 监控模块
// 用于捕获和分析 WebSocket 连接和消息交互

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// WebSocket 连接信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketConnection {
    pub id: String,                      // 连接唯一标识
    pub url: String,                     // WebSocket URL
    pub protocol: Option<String>,        // 子协议
    pub created_at: u64,                 // 创建时间戳
    pub closed_at: Option<u64>,          // 关闭时间戳
    pub close_code: Option<u16>,         // 关闭代码
    pub close_reason: Option<String>,    // 关闭原因
    pub messages: Vec<WebSocketMessage>, // 交互消息
    pub total_messages_sent: usize,
    pub total_messages_received: usize,
    pub total_bytes_sent: usize,
    pub total_bytes_received: usize,
}

/// WebSocket 消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketMessage {
    pub id: String,                  // 消息唯一标识
    pub direction: MessageDirection, // 发送/接收
    pub message_type: MessageType,   // 文本/二进制
    pub content: String,             // 内容（二进制会编码为 base64）
    pub size_bytes: usize,           // 大小
    pub timestamp: u64,              // 时间戳
}

/// 消息方向
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageDirection {
    Sent,
    Received,
}

/// 消息类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageType {
    Text,
    Binary,
}

impl WebSocketConnection {
    /// 创建新连接
    pub fn new(id: String, url: String, protocol: Option<String>) -> Self {
        Self {
            id,
            url,
            protocol,
            created_at: timestamp(),
            closed_at: None,
            close_code: None,
            close_reason: None,
            messages: Vec::new(),
            total_messages_sent: 0,
            total_messages_received: 0,
            total_bytes_sent: 0,
            total_bytes_received: 0,
        }
    }

    /// 添加消息
    pub fn add_message(&mut self, message: WebSocketMessage) {
        match message.direction {
            MessageDirection::Sent => {
                self.total_messages_sent += 1;
                self.total_bytes_sent += message.size_bytes;
            }
            MessageDirection::Received => {
                self.total_messages_received += 1;
                self.total_bytes_received += message.size_bytes;
            }
        }
        self.messages.push(message);
    }

    /// 关闭连接
    pub fn close(&mut self, code: u16, reason: Option<String>) {
        self.closed_at = Some(timestamp());
        self.close_code = Some(code);
        self.close_reason = reason;
    }

    /// 获取连接时长（毫秒）
    pub fn duration_ms(&self) -> u64 {
        let end = self.closed_at.unwrap_or_else(timestamp);
        (end - self.created_at) * 1000
    }

    /// 获取消息速率（消息/秒）
    pub fn message_rate(&self) -> f64 {
        let duration_s = (self.duration_ms() as f64) / 1000.0;
        if duration_s == 0.0 {
            0.0
        } else {
            (self.total_messages_sent + self.total_messages_received) as f64 / duration_s
        }
    }

    /// 获取吞吐量（字节/秒）
    pub fn throughput_bps(&self) -> f64 {
        let duration_s = (self.duration_ms() as f64) / 1000.0;
        if duration_s == 0.0 {
            0.0
        } else {
            (self.total_bytes_sent + self.total_bytes_received) as f64 / duration_s
        }
    }
}

/// WebSocket 监控器
pub struct WebSocketMonitor {
    connections: HashMap<String, WebSocketConnection>,
}

impl WebSocketMonitor {
    /// 创建新监控器
    pub fn new() -> Self {
        Self {
            connections: HashMap::new(),
        }
    }

    /// 记录新连接
    pub fn record_connection(&mut self, connection: WebSocketConnection) {
        self.connections.insert(connection.id.clone(), connection);
    }

    /// 更新连接
    pub fn update_connection(&mut self, id: &str, connection: WebSocketConnection) {
        self.connections.insert(id.to_string(), connection);
    }

    /// 获取连接
    pub fn get_connection(&self, id: &str) -> Option<&WebSocketConnection> {
        self.connections.get(id)
    }

    /// 获取所有连接
    pub fn get_all_connections(&self) -> Vec<&WebSocketConnection> {
        self.connections.values().collect()
    }

    /// 获取活跃连接数
    pub fn active_connection_count(&self) -> usize {
        self.connections
            .values()
            .filter(|c| c.closed_at.is_none())
            .count()
    }

    /// 获取已关闭的连接数
    pub fn closed_connection_count(&self) -> usize {
        self.connections
            .values()
            .filter(|c| c.closed_at.is_some())
            .count()
    }

    /// 获取总消息数
    pub fn total_messages(&self) -> usize {
        self.connections
            .values()
            .map(|c| c.total_messages_sent + c.total_messages_received)
            .sum()
    }

    /// 获取总数据量
    pub fn total_bytes(&self) -> usize {
        self.connections
            .values()
            .map(|c| c.total_bytes_sent + c.total_bytes_received)
            .sum()
    }

    /// 分析统计
    pub fn analyze(&self) -> WebSocketStats {
        let connections = self.get_all_connections();
        let total_connections = connections.len();
        let active_connections = self.active_connection_count();
        let closed_connections = self.closed_connection_count();
        let total_messages = self.total_messages();
        let total_bytes = self.total_bytes();

        let avg_messages_per_connection = if total_connections > 0 {
            total_messages as f64 / total_connections as f64
        } else {
            0.0
        };

        let avg_bytes_per_message = if total_messages > 0 {
            total_bytes as f64 / total_messages as f64
        } else {
            0.0
        };

        let slowest_connection = connections
            .iter()
            .min_by(|a, b| a.throughput_bps().partial_cmp(&b.throughput_bps()).unwrap())
            .map(|c| c.url.clone());

        let fastest_connection = connections
            .iter()
            .max_by(|a, b| a.throughput_bps().partial_cmp(&b.throughput_bps()).unwrap())
            .map(|c| c.url.clone());

        WebSocketStats {
            total_connections,
            active_connections,
            closed_connections,
            total_messages,
            total_bytes,
            avg_messages_per_connection,
            avg_bytes_per_message,
            slowest_connection,
            fastest_connection,
        }
    }
}

impl Default for WebSocketMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// WebSocket 统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketStats {
    pub total_connections: usize,
    pub active_connections: usize,
    pub closed_connections: usize,
    pub total_messages: usize,
    pub total_bytes: usize,
    pub avg_messages_per_connection: f64,
    pub avg_bytes_per_message: f64,
    pub slowest_connection: Option<String>,
    pub fastest_connection: Option<String>,
}

// 辅助函数
fn timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_websocket_connection_creation() {
        let conn = WebSocketConnection::new(
            "ws-1".to_string(),
            "ws://localhost:8080".to_string(),
            Some("chat".to_string()),
        );

        assert_eq!(conn.id, "ws-1");
        assert_eq!(conn.url, "ws://localhost:8080");
        assert_eq!(conn.protocol, Some("chat".to_string()));
        assert_eq!(conn.total_messages_sent, 0);
        assert_eq!(conn.total_messages_received, 0);
        assert!(conn.closed_at.is_none());
    }

    #[test]
    fn test_websocket_message_tracking() {
        let mut conn =
            WebSocketConnection::new("ws-1".to_string(), "ws://localhost:8080".to_string(), None);

        let msg1 = WebSocketMessage {
            id: "msg-1".to_string(),
            direction: MessageDirection::Sent,
            message_type: MessageType::Text,
            content: "Hello".to_string(),
            size_bytes: 5,
            timestamp: 1000,
        };

        let msg2 = WebSocketMessage {
            id: "msg-2".to_string(),
            direction: MessageDirection::Received,
            message_type: MessageType::Text,
            content: "World".to_string(),
            size_bytes: 5,
            timestamp: 1001,
        };

        conn.add_message(msg1);
        conn.add_message(msg2);

        assert_eq!(conn.total_messages_sent, 1);
        assert_eq!(conn.total_messages_received, 1);
        assert_eq!(conn.total_bytes_sent, 5);
        assert_eq!(conn.total_bytes_received, 5);
        assert_eq!(conn.messages.len(), 2);
    }

    #[test]
    fn test_websocket_connection_close() {
        let mut conn =
            WebSocketConnection::new("ws-1".to_string(), "ws://localhost:8080".to_string(), None);

        conn.close(1000, Some("Normal closure".to_string()));

        assert!(conn.closed_at.is_some());
        assert_eq!(conn.close_code, Some(1000));
        assert_eq!(conn.close_reason, Some("Normal closure".to_string()));
    }

    #[test]
    fn test_websocket_monitor() {
        let mut monitor = WebSocketMonitor::new();

        let mut conn =
            WebSocketConnection::new("ws-1".to_string(), "ws://localhost:8080".to_string(), None);

        for i in 0..5 {
            let msg = WebSocketMessage {
                id: format!("msg-{}", i),
                direction: if i % 2 == 0 {
                    MessageDirection::Sent
                } else {
                    MessageDirection::Received
                },
                message_type: MessageType::Text,
                content: format!("Message {}", i),
                size_bytes: 10,
                timestamp: 1000 + i as u64,
            };
            conn.add_message(msg);
        }

        monitor.record_connection(conn);

        assert_eq!(monitor.active_connection_count(), 1);
        assert_eq!(monitor.closed_connection_count(), 0);
        assert_eq!(monitor.total_messages(), 5);
        assert_eq!(monitor.total_bytes(), 50);
    }

    #[test]
    fn test_websocket_stats() {
        let mut monitor = WebSocketMonitor::new();

        let mut conn = WebSocketConnection::new(
            "ws-1".to_string(),
            "ws://example.com/socket".to_string(),
            None,
        );

        for i in 0..10 {
            conn.add_message(WebSocketMessage {
                id: format!("msg-{}", i),
                direction: MessageDirection::Sent,
                message_type: MessageType::Text,
                content: format!("msg-{}", i),
                size_bytes: 20,
                timestamp: 1000 + i as u64,
            });
        }

        monitor.record_connection(conn);

        let stats = monitor.analyze();
        assert_eq!(stats.total_connections, 1);
        assert_eq!(stats.active_connections, 1);
        assert_eq!(stats.total_messages, 10);
        assert_eq!(stats.total_bytes, 200);
        assert_eq!(stats.avg_messages_per_connection, 10.0);
        assert_eq!(stats.avg_bytes_per_message, 20.0);
    }
}
