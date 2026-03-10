/// Modern Web APIs for 2026 Standards
///
/// Implements essential browser APIs including:
/// - Console API with formatting
/// - Timer APIs (setTimeout/setInterval)
/// - URL and URLSearchParams
/// - Clipboard API
/// - Fetch API improvements (AbortController)
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

/// Console API implementation with multiple log levels and formatting
#[derive(Debug, Clone)]
pub struct ConsoleAPI {
    /// Log history
    logs: Vec<ConsoleEntry>,
    /// Maximum log history size
    max_logs: usize,
    /// Whether console is enabled
    enabled: bool,
}

#[derive(Debug, Clone)]
pub struct ConsoleEntry {
    pub level: LogLevel,
    pub message: String,
    pub timestamp: SystemTime,
    pub stack_trace: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogLevel {
    Log,
    Info,
    Warn,
    Error,
    Debug,
    Trace,
}

impl ConsoleAPI {
    pub fn new() -> Self {
        Self {
            logs: Vec::new(),
            max_logs: 1000,
            enabled: true,
        }
    }

    /// Log a message
    pub fn log(&mut self, message: String) {
        self.add_entry(LogLevel::Log, message, None);
    }

    /// Log an info message
    pub fn info(&mut self, message: String) {
        self.add_entry(LogLevel::Info, message, None);
    }

    /// Log a warning
    pub fn warn(&mut self, message: String) {
        self.add_entry(LogLevel::Warn, message, None);
    }

    /// Log an error
    pub fn error(&mut self, message: String) {
        self.add_entry(LogLevel::Error, message, None);
    }

    /// Log a debug message
    pub fn debug(&mut self, message: String) {
        self.add_entry(LogLevel::Debug, message, None);
    }

    /// Log with stack trace
    pub fn trace(&mut self, message: String, stack_trace: String) {
        self.add_entry(LogLevel::Trace, message, Some(stack_trace));
    }

    /// Format and log a table (simplified)
    pub fn table(&mut self, data: Vec<HashMap<String, String>>) {
        let mut table_str = String::from("Table:\n");
        for (i, row) in data.iter().enumerate() {
            table_str.push_str(&format!("Row {}: {:?}\n", i, row));
        }
        self.add_entry(LogLevel::Log, table_str, None);
    }

    /// Clear console
    pub fn clear(&mut self) {
        self.logs.clear();
    }

    /// Get all logs
    pub fn get_logs(&self) -> &[ConsoleEntry] {
        &self.logs
    }

    /// Get logs by level
    pub fn get_logs_by_level(&self, level: LogLevel) -> Vec<&ConsoleEntry> {
        self.logs.iter().filter(|e| e.level == level).collect()
    }

    fn add_entry(&mut self, level: LogLevel, message: String, stack_trace: Option<String>) {
        if !self.enabled {
            return;
        }

        let entry = ConsoleEntry {
            level,
            message,
            timestamp: SystemTime::now(),
            stack_trace,
        };

        self.logs.push(entry);

        // Limit log history
        if self.logs.len() > self.max_logs {
            self.logs.remove(0);
        }
    }
}

impl Default for ConsoleAPI {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer APIs - setTimeout and setInterval implementation
#[derive(Debug)]
pub struct TimerAPI {
    /// Active timers
    timers: HashMap<u32, Timer>,
    /// Next timer ID
    next_id: u32,
}

#[derive(Debug, Clone)]
pub struct Timer {
    pub id: u32,
    pub callback: String,
    pub delay: Duration,
    pub scheduled_time: Instant,
    pub timer_type: TimerType,
    pub is_active: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TimerType {
    Timeout,
    Interval,
}

impl TimerAPI {
    pub fn new() -> Self {
        Self {
            timers: HashMap::new(),
            next_id: 1,
        }
    }

    /// Set a timeout
    pub fn set_timeout(&mut self, callback: String, delay_ms: u64) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let timer = Timer {
            id,
            callback,
            delay: Duration::from_millis(delay_ms),
            scheduled_time: Instant::now(),
            timer_type: TimerType::Timeout,
            is_active: true,
        };

        self.timers.insert(id, timer);
        id
    }

    /// Set an interval
    pub fn set_interval(&mut self, callback: String, delay_ms: u64) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let timer = Timer {
            id,
            callback,
            delay: Duration::from_millis(delay_ms),
            scheduled_time: Instant::now(),
            timer_type: TimerType::Interval,
            is_active: true,
        };

        self.timers.insert(id, timer);
        id
    }

    /// Clear a timeout
    pub fn clear_timeout(&mut self, id: u32) {
        self.timers.remove(&id);
    }

    /// Clear an interval
    pub fn clear_interval(&mut self, id: u32) {
        self.timers.remove(&id);
    }

    /// Get pending timers that are ready to execute
    pub fn get_ready_timers(&mut self) -> Vec<Timer> {
        let now = Instant::now();
        let mut ready = Vec::new();

        for timer in self.timers.values_mut() {
            if timer.is_active && now.duration_since(timer.scheduled_time) >= timer.delay {
                ready.push(timer.clone());

                // Handle intervals - reschedule
                if timer.timer_type == TimerType::Interval {
                    timer.scheduled_time = now;
                } else {
                    // Timeouts are one-shot
                    timer.is_active = false;
                }
            }
        }

        // Remove inactive timeouts
        self.timers.retain(|_, t| t.is_active);

        ready
    }

    /// Clear all timers
    pub fn clear_all(&mut self) {
        self.timers.clear();
    }

    /// Get active timer count
    pub fn active_count(&self) -> usize {
        self.timers.values().filter(|t| t.is_active).count()
    }
}

impl Default for TimerAPI {
    fn default() -> Self {
        Self::new()
    }
}

/// URL API for URL parsing and manipulation
#[derive(Debug, Clone, PartialEq)]
pub struct URL {
    pub href: String,
    pub protocol: String,
    pub host: String,
    pub hostname: String,
    pub port: String,
    pub pathname: String,
    pub search: String,
    pub hash: String,
    pub origin: String,
}

impl URL {
    /// Parse a URL string
    pub fn parse(url: &str) -> Result<Self, String> {
        // Simple URL parsing (real implementation would use url crate)
        let url = url.trim();

        // Extract protocol
        let protocol = if let Some(idx) = url.find("://") {
            url[..idx].to_string()
        } else {
            return Err("Invalid URL: no protocol".to_string());
        };

        let after_protocol = &url[protocol.len() + 3..];

        // Extract host and port
        let host_end = after_protocol
            .find('/')
            .or_else(|| after_protocol.find('?'))
            .or_else(|| after_protocol.find('#'))
            .unwrap_or(after_protocol.len());

        let host_part = &after_protocol[..host_end];
        let (hostname, port) = if let Some(idx) = host_part.find(':') {
            (host_part[..idx].to_string(), host_part[idx + 1..].to_string())
        } else {
            (host_part.to_string(), String::new())
        };

        let host = if port.is_empty() {
            hostname.clone()
        } else {
            format!("{}:{}", hostname, port)
        };

        // Extract pathname, search, and hash
        let rest = &after_protocol[host_end..];
        let (pathname, search, hash) = Self::parse_path_components(rest);

        let origin = format!("{}://{}", protocol, host);

        Ok(Self {
            href: url.to_string(),
            protocol: format!("{}:", protocol),
            host,
            hostname,
            port,
            pathname,
            search,
            hash,
            origin,
        })
    }

    fn parse_path_components(path: &str) -> (String, String, String) {
        let mut pathname: String;
        let mut search = String::new();
        let mut hash = String::new();

        if let Some(hash_idx) = path.find('#') {
            hash = path[hash_idx..].to_string();
            let before_hash = &path[..hash_idx];

            if let Some(search_idx) = before_hash.find('?') {
                search = before_hash[search_idx..].to_string();
                pathname = before_hash[..search_idx].to_string();
            } else {
                pathname = before_hash.to_string();
            }
        } else if let Some(search_idx) = path.find('?') {
            search = path[search_idx..].to_string();
            pathname = path[..search_idx].to_string();
        } else {
            pathname = path.to_string();
        }

        if pathname.is_empty() {
            pathname = "/".to_string();
        }

        (pathname, search, hash)
    }
}

impl std::fmt::Display for URL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.href)
    }
}

/// URLSearchParams API for query string manipulation
#[derive(Debug, Clone)]
pub struct URLSearchParams {
    params: Vec<(String, String)>,
}

impl URLSearchParams {
    /// Create from query string
    pub fn new(query: &str) -> Self {
        let mut params = Vec::new();

        let query = query.trim_start_matches('?');
        if query.is_empty() {
            return Self { params };
        }

        for pair in query.split('&') {
            if let Some(idx) = pair.find('=') {
                let key = pair[..idx].to_string();
                let value = pair[idx + 1..].to_string();
                params.push((key, value));
            } else {
                params.push((pair.to_string(), String::new()));
            }
        }

        Self { params }
    }

    /// Append a parameter
    pub fn append(&mut self, key: String, value: String) {
        self.params.push((key, value));
    }

    /// Get a parameter
    pub fn get(&self, key: &str) -> Option<&String> {
        self.params
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Get all values for a key
    pub fn get_all(&self, key: &str) -> Vec<&String> {
        self.params
            .iter()
            .filter(|(k, _)| k == key)
            .map(|(_, v)| v)
            .collect()
    }

    /// Check if parameter exists
    pub fn has(&self, key: &str) -> bool {
        self.params.iter().any(|(k, _)| k == key)
    }

    /// Delete a parameter
    pub fn delete(&mut self, key: &str) {
        self.params.retain(|(k, _)| k != key);
    }

    /// Set a parameter (replace if exists)
    pub fn set(&mut self, key: String, value: String) {
        self.delete(&key);
        self.append(key, value);
    }
}

impl std::fmt::Display for URLSearchParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let query_string = self.params
            .iter()
            .map(|(k, v)| {
                if v.is_empty() {
                    k.clone()
                } else {
                    format!("{}={}", k, v)
                }
            })
            .collect::<Vec<_>>()
            .join("&");
        write!(f, "{}", query_string)
    }
}

impl URLSearchParams {
    /// Get all entries
    pub fn entries(&self) -> &[(String, String)] {
        &self.params
    }
}

/// Clipboard API for async read/write operations
#[derive(Debug, Clone)]
pub struct ClipboardAPI {
    /// Current clipboard text
    text: String,
    /// Clipboard history (simplified)
    history: Vec<ClipboardEntry>,
    /// Maximum history size
    max_history: usize,
}

#[derive(Debug, Clone)]
pub struct ClipboardEntry {
    pub text: String,
    pub timestamp: SystemTime,
}

impl ClipboardAPI {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            history: Vec::new(),
            max_history: 10,
        }
    }

    /// Write text to clipboard
    pub fn write_text(&mut self, text: String) -> Result<(), String> {
        self.text = text.clone();
        self.history.push(ClipboardEntry {
            text,
            timestamp: SystemTime::now(),
        });

        // Limit history
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        Ok(())
    }

    /// Read text from clipboard
    pub fn read_text(&self) -> String {
        self.text.clone()
    }

    /// Get clipboard history
    pub fn get_history(&self) -> &[ClipboardEntry] {
        &self.history
    }

    /// Clear clipboard
    pub fn clear(&mut self) {
        self.text.clear();
    }
}

impl Default for ClipboardAPI {
    fn default() -> Self {
        Self::new()
    }
}

/// AbortController for cancelling fetch requests
#[derive(Debug, Clone)]
pub struct AbortController {
    signal: AbortSignal,
}

#[derive(Debug, Clone)]
pub struct AbortSignal {
    pub aborted: bool,
    pub reason: Option<String>,
}

impl AbortController {
    pub fn new() -> Self {
        Self {
            signal: AbortSignal {
                aborted: false,
                reason: None,
            },
        }
    }

    /// Get the signal
    pub fn signal(&self) -> &AbortSignal {
        &self.signal
    }

    /// Abort the operation
    pub fn abort(&mut self, reason: Option<String>) {
        self.signal.aborted = true;
        self.signal.reason = reason;
    }
}

impl Default for AbortController {
    fn default() -> Self {
        Self::new()
    }
}

impl AbortSignal {
    pub fn is_aborted(&self) -> bool {
        self.aborted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_console_log() {
        let mut console = ConsoleAPI::new();
        console.log("Hello".to_string());
        console.info("Info message".to_string());
        console.warn("Warning".to_string());

        assert_eq!(console.get_logs().len(), 3);
        assert_eq!(console.get_logs()[0].level, LogLevel::Log);
        assert_eq!(console.get_logs()[1].level, LogLevel::Info);
        assert_eq!(console.get_logs()[2].level, LogLevel::Warn);
    }

    #[test]
    fn test_console_filter_by_level() {
        let mut console = ConsoleAPI::new();
        console.log("Log".to_string());
        console.error("Error".to_string());
        console.error("Error2".to_string());

        let errors = console.get_logs_by_level(LogLevel::Error);
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_set_timeout() {
        let mut timer = TimerAPI::new();
        let id = timer.set_timeout("console.log('hello')".to_string(), 100);

        assert!(id > 0);
        assert_eq!(timer.active_count(), 1);

        timer.clear_timeout(id);
        assert_eq!(timer.active_count(), 0);
    }

    #[test]
    fn test_set_interval() {
        let mut timer = TimerAPI::new();
        let id = timer.set_interval("console.log('tick')".to_string(), 1000);

        assert!(id > 0);
        assert_eq!(timer.active_count(), 1);

        timer.clear_interval(id);
        assert_eq!(timer.active_count(), 0);
    }

    #[test]
    fn test_url_parse() {
        let url = URL::parse("https://example.com:8080/path?query=1#hash").unwrap();

        assert_eq!(url.protocol, "https:");
        assert_eq!(url.hostname, "example.com");
        assert_eq!(url.port, "8080");
        assert_eq!(url.pathname, "/path");
        assert_eq!(url.search, "?query=1");
        assert_eq!(url.hash, "#hash");
        assert_eq!(url.origin, "https://example.com:8080");
    }

    #[test]
    fn test_url_parse_simple() {
        let url = URL::parse("http://localhost/").unwrap();

        assert_eq!(url.protocol, "http:");
        assert_eq!(url.hostname, "localhost");
        assert_eq!(url.port, "");
        assert_eq!(url.pathname, "/");
    }

    #[test]
    fn test_url_search_params() {
        let mut params = URLSearchParams::new("?foo=bar&baz=qux");

        assert_eq!(params.get("foo"), Some(&"bar".to_string()));
        assert_eq!(params.get("baz"), Some(&"qux".to_string()));
        assert!(params.has("foo"));

        params.set("foo".to_string(), "new".to_string());
        assert_eq!(params.get("foo"), Some(&"new".to_string()));

        params.delete("baz");
        assert!(!params.has("baz"));
    }

    #[test]
    fn test_url_search_params_to_string() {
        let mut params = URLSearchParams::new("");
        params.append("key1".to_string(), "value1".to_string());
        params.append("key2".to_string(), "value2".to_string());

        let query = params.to_string();
        assert_eq!(query, "key1=value1&key2=value2");
    }

    #[test]
    fn test_clipboard() {
        let mut clipboard = ClipboardAPI::new();

        clipboard.write_text("Hello World".to_string()).unwrap();
        assert_eq!(clipboard.read_text(), "Hello World");

        clipboard.write_text("New text".to_string()).unwrap();
        assert_eq!(clipboard.read_text(), "New text");
        assert_eq!(clipboard.get_history().len(), 2);

        clipboard.clear();
        assert_eq!(clipboard.read_text(), "");
    }

    #[test]
    fn test_abort_controller() {
        let mut controller = AbortController::new();

        assert!(!controller.signal().is_aborted());

        controller.abort(Some("User cancelled".to_string()));

        assert!(controller.signal().is_aborted());
        assert_eq!(
            controller.signal().reason,
            Some("User cancelled".to_string())
        );
    }
}
