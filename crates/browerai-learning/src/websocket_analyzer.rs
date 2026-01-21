// WebSocket 和实时通信分析器
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketInfo {
    pub url: String,
    pub protocols: Vec<String>,
    pub connection_type: ConnectionType,
    pub messages: Vec<WebSocketMessage>,
    pub events: Vec<WebSocketEvent>,
    pub authentication: Option<WebSocketAuth>,
    pub heartbeat: Option<HeartbeatInfo>,
    pub reconnection: Option<ReconnectionStrategy>,
    pub line_number: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConnectionType {
    NativeWebSocket,
    SocketIO,
    SockJS,
    MQTT,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketMessage {
    pub timestamp: u64,
    pub direction: MessageDirection,
    pub message_type: MessageType,
    pub payload: String,
    pub size_bytes: usize,
    pub parsed_data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageDirection {
    Sent,
    Received,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageType {
    Text,
    Binary,
    Ping,
    Pong,
    Close,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketEvent {
    pub timestamp: u64,
    pub event_type: EventType,
    pub event_name: Option<String>,
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventType {
    Open,
    Close,
    Error,
    Message,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketAuth {
    pub method: AuthMethod,
    pub timing: AuthTiming,
    pub credentials: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuthMethod {
    QueryParams,
    InitialMessage,
    Headers,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuthTiming {
    BeforeConnection,
    DuringHandshake,
    AfterConnection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatInfo {
    pub interval_ms: u64,
    pub message: String,
    pub response_expected: bool,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconnectionStrategy {
    pub enabled: bool,
    pub max_attempts: Option<u32>,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_strategy: BackoffStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BackoffStrategy {
    Linear,
    Exponential,
    Fibonacci,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocketIOInfo {
    pub version: String,
    pub transport: Vec<String>,
    pub path: String,
    pub namespaces: Vec<String>,
    pub events: HashMap<String, Vec<EventHandler>>,
    pub ack_callbacks: HashMap<u32, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventHandler {
    pub event_name: String,
    pub handler_code: String,
    pub parameters: Vec<String>,
}

pub struct WebSocketAnalyzer {
    #[allow(dead_code)]
    capture_messages: bool,
    #[allow(dead_code)]
    max_message_size: usize,
}

impl Default for WebSocketAnalyzer {
    fn default() -> Self {
        Self::new(true, 1024 * 1024)
    }
}

impl WebSocketAnalyzer {
    pub fn new(capture_messages: bool, max_message_size: usize) -> Self {
        Self {
            capture_messages,
            max_message_size,
        }
    }

    pub fn extract_from_js(&self, js_code: &str) -> Result<Vec<WebSocketInfo>> {
        let mut connections = Vec::new();

        let lines: Vec<&str> = js_code.lines().collect();

        if let Some(ws_info) = self.parse_native_websocket(js_code, &lines)? {
            connections.push(ws_info);
        }

        if let Some(ws_info) = self.parse_socketio(js_code, &lines)? {
            connections.push(ws_info);
        }

        if let Some(ws_info) = self.parse_sockjs(js_code, &lines)? {
            connections.push(ws_info);
        }

        if let Some(ws_info) = self.parse_mqtt(js_code, &lines)? {
            connections.push(ws_info);
        }

        Ok(connections)
    }

    fn parse_native_websocket(
        &self,
        js_code: &str,
        lines: &[&str],
    ) -> Result<Option<WebSocketInfo>> {
        let ws_pattern =
            Regex::new(r#"new\s+WebSocket\s*\(\s*["']([^"']+)["']\s*(?:,\s*(\[.*?\]))?\s*\)"#)?;

        let mut captures: Vec<(String, Vec<String>, usize)> = Vec::new();
        use std::sync::LazyLock;
        static PROTOCOL_RE: LazyLock<Option<Regex>> =
            LazyLock::new(|| Regex::new(r#"['"]([^'"]+)['"]"#).ok());

        for (line_idx, line) in lines.iter().enumerate() {
            if let Some(caps) = ws_pattern.captures(line) {
                let url = caps
                    .get(1)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                let protocols_str = caps
                    .get(2)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                let protocols = if protocols_str.is_empty() {
                    vec![]
                } else {
                    match PROTOCOL_RE
                        .as_ref()
                        .and_then(|re| re.captures(&protocols_str))
                    {
                        Some(caps) => {
                            if let Some(cap) = caps.get(1) {
                                vec![cap.as_str().to_string()]
                            } else {
                                vec![]
                            }
                        }
                        None => vec![],
                    }
                };
                captures.push((url, protocols, line_idx));
            }
        }

        if captures.is_empty() {
            return Ok(None);
        }

        let (url, protocols, line_number) = captures.into_iter().next().unwrap();

        let reconnection = self.extract_reconnection_strategy(js_code);
        let heartbeat = self.extract_heartbeat_config(js_code);
        let authentication = self.extract_websocket_auth(js_code);

        let events = vec![WebSocketEvent {
            timestamp: 0,
            event_type: EventType::Open,
            event_name: Some("onopen".to_string()),
            data: None,
        }];

        Ok(Some(WebSocketInfo {
            url,
            protocols,
            connection_type: ConnectionType::NativeWebSocket,
            messages: vec![],
            events,
            authentication,
            heartbeat,
            reconnection,
            line_number: Some(line_number),
        }))
    }

    fn parse_socketio(&self, js_code: &str, lines: &[&str]) -> Result<Option<WebSocketInfo>> {
        let io_pattern = Regex::new(r#"io\s*\(\s*["']([^"']+)["']\s*(?:,\s*(\{[^}]*\}))?\s*\)"#)?;
        let socket_io_new_pattern = Regex::new(r#"new\s+SocketIO\s*\(\s*["']([^"']+)["']"#)?;
        let _version_pattern = Regex::new(r"socket\.io[^\\]*v?([0-9]+\.[0-9]+)")?;

        let mut found = false;
        let mut url = String::new();
        let mut line_number = 0;

        for (line_idx, line) in lines.iter().enumerate() {
            if let Some(caps) = io_pattern.captures(line) {
                url = caps
                    .get(1)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                line_number = line_idx;
                found = true;
                break;
            }
            if let Some(caps) = socket_io_new_pattern.captures(line) {
                url = caps
                    .get(1)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                line_number = line_idx;
                found = true;
                break;
            }
        }

        if !found {
            return Ok(None);
        }

        let protocols = vec!["polling".to_string(), "websocket".to_string()];

        let heartbeat = Some(HeartbeatInfo {
            interval_ms: 25000,
            message: "2".to_string(),
            response_expected: true,
            timeout_ms: Some(5000),
        });

        let reconnection = self.extract_socketio_reconnection(js_code);

        let _namespaces = self.extract_socketio_namespaces(js_code);
        let events = self.extract_socketio_events(js_code);

        Ok(Some(WebSocketInfo {
            url,
            protocols,
            connection_type: ConnectionType::SocketIO,
            messages: vec![],
            events,
            authentication: None,
            heartbeat,
            reconnection,
            line_number: Some(line_number),
        }))
    }

    fn parse_sockjs(&self, _js_code: &str, lines: &[&str]) -> Result<Option<WebSocketInfo>> {
        let sockjs_pattern = Regex::new(r#"new\s+SockJS\s*\(\s*["']([^"']+)["']"#)?;
        let sockjs_classic_pattern = Regex::new(r#"SockJS\s*\(\s*["']([^"']+)["']"#)?;

        let mut url = String::new();
        let mut line_number = 0;

        for (line_idx, line) in lines.iter().enumerate() {
            if let Some(caps) = sockjs_pattern.captures(line) {
                url = caps
                    .get(1)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                line_number = line_idx;
                break;
            }
            if let Some(caps) = sockjs_classic_pattern.captures(line) {
                url = caps
                    .get(1)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                line_number = line_idx;
                break;
            }
        }

        if url.is_empty() {
            return Ok(None);
        }

        Ok(Some(WebSocketInfo {
            url,
            protocols: vec![],
            connection_type: ConnectionType::SockJS,
            messages: vec![],
            events: vec![],
            authentication: None,
            heartbeat: None,
            reconnection: None,
            line_number: Some(line_number),
        }))
    }

    fn parse_mqtt(&self, _js_code: &str, lines: &[&str]) -> Result<Option<WebSocketInfo>> {
        let mqtt_pattern = Regex::new(r#"(?:new\s+)?MQTT(?:Client)?\s*\(\s*["']([^"']+)["']"#)?;
        let mqtt_ws_pattern = Regex::new(r#"MQTT\.connect\s*\(\s*["']([^"']+)["']"#)?;

        let mut url = String::new();
        let mut line_number = 0;

        for (line_idx, line) in lines.iter().enumerate() {
            if let Some(caps) = mqtt_pattern.captures(line) {
                url = caps
                    .get(1)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                line_number = line_idx;
                break;
            }
            if let Some(caps) = mqtt_ws_pattern.captures(line) {
                url = caps
                    .get(1)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                line_number = line_idx;
                break;
            }
        }

        if url.is_empty() {
            return Ok(None);
        }

        Ok(Some(WebSocketInfo {
            url,
            protocols: vec![],
            connection_type: ConnectionType::MQTT,
            messages: vec![],
            events: vec![],
            authentication: None,
            heartbeat: None,
            reconnection: None,
            line_number: Some(line_number),
        }))
    }

    fn extract_reconnection_strategy(&self, js_code: &str) -> Option<ReconnectionStrategy> {
        let _reconnect_pattern =
            Regex::new(r#"\.on\s*\(\s*["']close["']\s*,\s*[^)]*\breconnect\s*\([^)]*\)"#).ok()?;
        let max_attempts_pattern = Regex::new(r"max(?:imum)?[-_]?attempts?\s*[:=]\s*(\d+)").ok()?;
        let initial_delay_pattern =
            Regex::new(r"(?:initial[-_]?delay|reconnect[-_]?interval)\s*[:=]\s*(\d+)").ok()?;

        let has_reconnect = js_code.contains("reconnect")
            || js_code.contains("retry")
            || js_code.contains("reconnectInterval");

        if !has_reconnect {
            return None;
        }

        let max_attempts = max_attempts_pattern
            .captures(js_code)
            .and_then(|caps| caps.get(1).map(|m| m.as_str().parse().ok()))
            .flatten();

        let initial_delay = initial_delay_pattern
            .captures(js_code)
            .and_then(|caps| caps.get(1).map(|m| m.as_str().parse().ok()))
            .flatten()
            .unwrap_or(1000);

        let backoff = if js_code.contains("exponential") || js_code.contains("* 2") {
            BackoffStrategy::Exponential
        } else if js_code.contains("linear") || js_code.contains("+=") {
            BackoffStrategy::Linear
        } else {
            BackoffStrategy::Exponential
        };

        Some(ReconnectionStrategy {
            enabled: true,
            max_attempts,
            initial_delay_ms: initial_delay,
            max_delay_ms: 30000,
            backoff_strategy: backoff,
        })
    }

    fn extract_socketio_reconnection(&self, js_code: &str) -> Option<ReconnectionStrategy> {
        let reconnection_pattern = Regex::new(r"reconnection\s*[:=]\s*(true|false)").ok()?;
        let reconnection_delay_pattern = Regex::new(r"reconnectionDelay\s*[:=]\s*(\d+)").ok()?;
        let reconnection_attempts_pattern =
            Regex::new(r"reconnectionAttempts\s*[:=]\s*(\d+)").ok()?;

        let reconnection_enabled = reconnection_pattern
            .captures(js_code)
            .map(|caps| caps.get(1).map(|m| m.as_str() == "true").unwrap_or(true))
            .unwrap_or(true);

        let delay = reconnection_delay_pattern
            .captures(js_code)
            .and_then(|caps| caps.get(1).map(|m| m.as_str().parse().ok()))
            .flatten()
            .unwrap_or(1000);

        let attempts = reconnection_attempts_pattern
            .captures(js_code)
            .and_then(|caps| caps.get(1).map(|m| m.as_str().parse().ok()))
            .flatten();

        Some(ReconnectionStrategy {
            enabled: reconnection_enabled,
            max_attempts: attempts,
            initial_delay_ms: delay,
            max_delay_ms: 30000,
            backoff_strategy: BackoffStrategy::Exponential,
        })
    }

    fn extract_heartbeat_config(&self, js_code: &str) -> Option<HeartbeatInfo> {
        let ping_interval_pattern =
            Regex::new(r"(?:pingInterval|heartbeat(?:Interval)?)\s*[:=]\s*(\d+)").ok()?;
        let ping_timeout_pattern =
            Regex::new(r"(?:pingTimeout|heartbeatTimeout)\s*[:=]\s*(\d+)").ok()?;

        let has_heartbeat = js_code.contains("ping") && js_code.contains("heartbeat");

        if !has_heartbeat {
            return None;
        }

        let interval = ping_interval_pattern
            .captures(js_code)
            .and_then(|caps| caps.get(1).map(|m| m.as_str().parse().ok()))
            .flatten()
            .unwrap_or(25000);

        let timeout = ping_timeout_pattern
            .captures(js_code)
            .and_then(|caps| caps.get(1).map(|m| m.as_str().parse().ok()))
            .flatten();

        Some(HeartbeatInfo {
            interval_ms: interval,
            message: "ping".to_string(),
            response_expected: true,
            timeout_ms: timeout,
        })
    }

    fn extract_websocket_auth(&self, js_code: &str) -> Option<WebSocketAuth> {
        let token_in_url =
            Regex::new(r"ws[s]?://[^?]*[?&](?:token|auth|access_token)=([^&]+)").ok()?;
        let auth_header = Regex::new(
            r#"\.set(?:Header|Authorization)\s*\(\s*["']Authorization["']\s*,\s*["'][^"']+["']"#,
        )
        .ok()?;

        let mut credentials = HashMap::new();
        let mut method = AuthMethod::Headers;
        let mut timing = AuthTiming::DuringHandshake;

        if let Some(caps) = token_in_url.captures(js_code) {
            if let Some(token) = caps.get(1) {
                credentials.insert("token".to_string(), token.as_str().to_string());
                method = AuthMethod::QueryParams;
            }
        }

        if auth_header.is_match(js_code) {
            method = AuthMethod::Headers;
        }

        if method == AuthMethod::Headers
            && (js_code.contains("Sec-WebSocket-Protocol") || js_code.contains("protocols"))
        {
            timing = AuthTiming::DuringHandshake;
        }

        if credentials.is_empty() {
            return None;
        }

        Some(WebSocketAuth {
            method,
            timing,
            credentials,
        })
    }

    fn extract_socketio_namespaces(&self, js_code: &str) -> Vec<String> {
        let namespace_pattern = match Regex::new(r#"\.of\s*\(\s*["'](/[^"']+)["']\s*\)"#) {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };
        let io_namespace_pattern = match Regex::new(r#"io\s*\(\s*["'](/[^"']+)["']"#) {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };

        let mut namespaces: Vec<String> = Vec::new();

        for caps in namespace_pattern.captures_iter(js_code) {
            if let Some(ns) = caps.get(1) {
                namespaces.push(ns.as_str().to_string());
            }
        }

        for caps in io_namespace_pattern.captures_iter(js_code) {
            if let Some(ns) = caps.get(1) {
                let ns_str = ns.as_str().to_string();
                if !namespaces.contains(&ns_str) {
                    namespaces.push(ns_str);
                }
            }
        }

        namespaces
    }

    fn extract_socketio_events(&self, js_code: &str) -> Vec<WebSocketEvent> {
        let event_pattern =
            match Regex::new(r#"\.(on|once)\s*\(\s*["']([^"']+)["']\s*,\s*function"#) {
                Ok(p) => p,
                Err(_) => return Vec::new(),
            };
        let emit_pattern = match Regex::new(r#"\.(emit|send)\s*\(\s*["']([^"']+)["']"#) {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };

        let mut events = Vec::new();
        let mut event_names = std::collections::HashSet::new();

        for caps in event_pattern.captures_iter(js_code) {
            if let Some(event_name) = caps.get(2) {
                let name = event_name.as_str().to_string();
                if !event_names.contains(&name) {
                    event_names.insert(name.clone());
                    events.push(WebSocketEvent {
                        timestamp: 0,
                        event_type: EventType::Custom(name.clone()),
                        event_name: Some(name),
                        data: None,
                    });
                }
            }
        }

        for caps in emit_pattern.captures_iter(js_code) {
            if let Some(event_name) = caps.get(2) {
                let name = format!("emit:{}", event_name.as_str());
                if !event_names.contains(&name) {
                    event_names.insert(name.clone());
                    events.push(WebSocketEvent {
                        timestamp: 0,
                        event_type: EventType::Message,
                        event_name: Some(name),
                        data: None,
                    });
                }
            }
        }

        events
    }

    pub fn analyze_message_format(&self, messages: &[WebSocketMessage]) -> MessageFormatAnalysis {
        let mut analysis = MessageFormatAnalysis {
            total_messages: messages.len(),
            text_messages: 0,
            binary_messages: 0,
            ping_pong_messages: 0,
            avg_message_size: 0,
            common_patterns: HashMap::new(),
            data_types: HashMap::new(),
            protocols: HashMap::new(),
        };

        let mut total_size = 0;

        for msg in messages {
            match msg.message_type {
                MessageType::Text => analysis.text_messages += 1,
                MessageType::Binary => analysis.binary_messages += 1,
                MessageType::Ping | MessageType::Pong => analysis.ping_pong_messages += 1,
                _ => {}
            }

            total_size += msg.size_bytes;

            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&msg.payload) {
                self.analyze_json_structure(&json, &mut analysis.data_types);

                if let Some(protocol) = json.get("type").and_then(|t| t.as_str()) {
                    *analysis.protocols.entry(protocol.to_string()).or_insert(0) += 1;
                }
            }
        }

        if !messages.is_empty() {
            analysis.avg_message_size = total_size / messages.len();
        }

        analysis
    }

    fn analyze_json_structure(&self, json: &serde_json::Value, types: &mut HashMap<String, u32>) {
        if let serde_json::Value::Object(map) = json {
            for (key, value) in map {
                let type_name = match value {
                    serde_json::Value::String(_) => "string",
                    serde_json::Value::Number(_) => "number",
                    serde_json::Value::Bool(_) => "boolean",
                    serde_json::Value::Array(_) => "array",
                    serde_json::Value::Object(_) => "object",
                    serde_json::Value::Null => "null",
                };
                *types.entry(format!("{}: {}", key, type_name)).or_insert(0) += 1;
            }
        }
    }

    pub fn infer_event_patterns(&self, events: &[WebSocketEvent]) -> Vec<EventPattern> {
        let mut patterns = Vec::new();

        for window in events.windows(2) {
            if let [event1, event2] = window {
                let time_diff = (event2.timestamp as i64 - event1.timestamp as i64).abs();
                if time_diff < 1000 {
                    patterns.push(EventPattern {
                        pattern_type: PatternType::RequestResponse,
                        events: vec![event1.clone(), event2.clone()],
                        confidence: 0.8,
                    });
                }
            }
        }

        let event_types: Vec<_> = events
            .iter()
            .filter_map(|e| e.event_name.as_ref())
            .collect();

        if event_types
            .iter()
            .any(|n| n.contains("heartbeat") || n.contains("ping"))
        {
            patterns.push(EventPattern {
                pattern_type: PatternType::Heartbeat,
                events: events[..std::cmp::min(2, events.len())].to_vec(),
                confidence: 0.9,
            });
        }

        patterns
    }

    pub fn generate_reconnection_code(&self, strategy: &ReconnectionStrategy) -> String {
        let backoff_impl = match strategy.backoff_strategy {
            BackoffStrategy::Exponential => {
                "Math.min(initialDelay * Math.pow(2, attempts), maxDelay)".to_string()
            }
            BackoffStrategy::Linear => {
                "Math.min(initialDelay + attempts * 1000, maxDelay)".to_string()
            }
            BackoffStrategy::Fibonacci => {
                let a = strategy.initial_delay_ms;
                let _b = (strategy.initial_delay_ms as f64 * 1.618) as u64;
                format!("Math.min(getFibonacci({}) * attempts, maxDelay)", a)
            }
            BackoffStrategy::Custom => "initialDelay * (1 + attempts)".to_string(),
        };

        format!(
            r#"function connectWithRetry() {{
  let attempts = 0;
  const maxAttempts = {};
  const initialDelay = {};
  const maxDelay = {};
  
  function connect() {{
    try {{
      const ws = new WebSocket(url);
      
      ws.onopen = function() {{
        console.log('WebSocket connected');
        attempts = 0;
      }};
      
      ws.onclose = function(event) {{
        if (attempts >= maxAttempts) {{
          console.error('Max reconnection attempts reached');
          return;
        }}
        
        const delay = {};
        console.log(`Reconnecting in ${{delay}}ms (attempt ${{attempts + 1}}/${{maxAttempts}})`);
        
        setTimeout(function() {{
          attempts++;
          connect();
        }}, delay);
      }};
      
      ws.onerror = function(error) {{
        console.error('WebSocket error:', error);
      }};
    }} catch (e) {{
      console.error('Connection error:', e);
      setTimeout(connect, initialDelay);
    }}
  }}
  
  connect();
}}"#,
            strategy.max_attempts.unwrap_or(999),
            strategy.initial_delay_ms,
            strategy.max_delay_ms,
            backoff_impl
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageFormatAnalysis {
    pub total_messages: usize,
    pub text_messages: usize,
    pub binary_messages: usize,
    pub ping_pong_messages: usize,
    pub avg_message_size: usize,
    pub common_patterns: HashMap<String, u32>,
    pub data_types: HashMap<String, u32>,
    pub protocols: HashMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPattern {
    pub pattern_type: PatternType,
    pub events: Vec<WebSocketEvent>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PatternType {
    RequestResponse,
    Broadcast,
    Heartbeat,
    Stream,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_websocket_analyzer_creation() {
        let analyzer = WebSocketAnalyzer::new(true, 2048);
        assert_eq!(analyzer.capture_messages, true);
        assert_eq!(analyzer.max_message_size, 2048);
    }

    #[test]
    fn test_native_websocket_detection() {
        let analyzer = WebSocketAnalyzer::default();
        let js_code = r#"
            const ws = new WebSocket('wss://example.com/socket');
            ws.onopen = () => console.log('connected');
            ws.onmessage = (msg) => console.log(msg);
        "#;

        let result = analyzer.extract_from_js(js_code).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].connection_type, ConnectionType::NativeWebSocket);
        assert_eq!(result[0].url, "wss://example.com/socket");
    }

    #[test]
    fn test_websocket_with_protocols() {
        let analyzer = WebSocketAnalyzer::default();
        let js_code = r#"
            const ws = new WebSocket('wss://example.com', ['graphql-ws']);
        "#;

        let result = analyzer.extract_from_js(js_code).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].protocols, vec!["graphql-ws"]);
    }

    #[test]
    fn test_socketio_detection() {
        let analyzer = WebSocketAnalyzer::default();
        let js_code = r#"
            const socket = io('https://example.com', {
                reconnection: true,
                reconnectionAttempts: 5
            });
            socket.on('message', (data) => console.log(data));
        "#;

        let result = analyzer.extract_from_js(js_code).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].connection_type, ConnectionType::SocketIO);
    }

    #[test]
    fn test_socketio_with_namespace() {
        let analyzer = WebSocketAnalyzer::default();
        let js_code = r#"
            const mainSocket = io('/chat');
            const adminSocket = io('/admin');
        "#;

        let result = analyzer.extract_from_js(js_code).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_sockjs_detection() {
        let analyzer = WebSocketAnalyzer::default();
        let js_code = r#"
            const sock = new SockJS('https://example.com/sockjs');
        "#;

        let result = analyzer.extract_from_js(js_code).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].connection_type, ConnectionType::SockJS);
    }

    #[test]
    fn test_mqtt_detection() {
        let analyzer = WebSocketAnalyzer::default();
        let js_code = r#"
            const client = MQTT.connect('wss://broker.example.com:8083/mqtt');
        "#;

        let result = analyzer.extract_from_js(js_code).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].connection_type, ConnectionType::MQTT);
    }

    #[test]
    fn test_message_format_analysis() {
        let analyzer = WebSocketAnalyzer::default();
        let messages = vec![
            WebSocketMessage {
                timestamp: 1000,
                direction: MessageDirection::Sent,
                message_type: MessageType::Text,
                payload: r#"{"type":"ping"}"#.to_string(),
                size_bytes: 15,
                parsed_data: None,
            },
            WebSocketMessage {
                timestamp: 2000,
                direction: MessageDirection::Received,
                message_type: MessageType::Text,
                payload: r#"{"type":"pong","data":{}}"#.to_string(),
                size_bytes: 25,
                parsed_data: None,
            },
        ];

        let analysis = analyzer.analyze_message_format(&messages);
        assert_eq!(analysis.total_messages, 2);
        assert_eq!(analysis.text_messages, 2);
        assert_eq!(analysis.avg_message_size, 20);
        assert_eq!(analysis.protocols.get("ping").copied(), Some(1));
        assert_eq!(analysis.protocols.get("pong").copied(), Some(1));
    }

    #[test]
    fn test_reconnection_code_generation() {
        let analyzer = WebSocketAnalyzer::default();
        let strategy = ReconnectionStrategy {
            enabled: true,
            max_attempts: Some(5),
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_strategy: BackoffStrategy::Exponential,
        };

        let code = analyzer.generate_reconnection_code(&strategy);
        assert!(code.contains("maxAttempts = 5"));
        assert!(code.contains("initialDelay = 1000"));
        assert!(code.contains("Exponential"));
    }

    #[test]
    fn test_reconnection_strategy_extraction() {
        let analyzer = WebSocketAnalyzer::default();
        let js_code = r#"
            const ws = new WebSocket('wss://example.com');
            ws.onclose = function() {
                if (attempts < maxAttempts) {
                    setTimeout(connect, initialDelay * Math.pow(2, attempts));
                }
            };
        "#;

        let result = analyzer.extract_from_js(js_code).unwrap();
        assert!(result.len() > 0);
        if let Some(ws_info) = result.first() {
            if let Some(reconnect) = &ws_info.reconnection {
                assert!(reconnect.enabled);
                assert_eq!(reconnect.backoff_strategy, BackoffStrategy::Exponential);
            }
        }
    }
}
