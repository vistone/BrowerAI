//! Console - 控制台
//!
//! 提供日志记录和执行功能：
//! - 日志级别（Log/Info/Warn/Error/Debug）
//! - 命令执行
//! - 历史记录
//! - 格式化输出

use browerai_core::Result;
use std::collections::VecDeque;

/// 控制台
#[derive(Debug, Clone)]
pub struct Console {
    /// 配置
    config: ConsoleConfig,
    /// 消息列表
    messages: VecDeque<ConsoleMessage>,
    /// 命令历史
    command_history: VecDeque<String>,
}

impl Console {
    /// 创建新的控制台
    pub fn new() -> Self {
        Self {
            config: ConsoleConfig::default(),
            messages: VecDeque::new(),
            command_history: VecDeque::new(),
        }
    }

    /// 使用配置创建控制台
    pub fn with_config(config: ConsoleConfig) -> Self {
        Self {
            config,
            messages: VecDeque::new(),
            command_history: VecDeque::new(),
        }
    }

    /// 记录日志
    fn log_message(&mut self, level: LogLevel, message: impl Into<String>) {
        let msg = ConsoleMessage {
            level,
            message: message.into(),
            timestamp: std::time::SystemTime::now(),
            source: None,
        };

        self.messages.push_back(msg);

        // 限制消息数量
        while self.messages.len() > self.config.max_messages {
            self.messages.pop_front();
        }
    }

    /// 记录普通日志
    pub fn log(&mut self, message: impl Into<String>) {
        self.log_message(LogLevel::Log, message);
    }

    /// 记录信息
    pub fn info(&mut self, message: impl Into<String>) {
        self.log_message(LogLevel::Info, message);
    }

    /// 记录警告
    pub fn warn(&mut self, message: impl Into<String>) {
        self.log_message(LogLevel::Warn, message);
    }

    /// 记录错误
    pub fn error(&mut self, message: impl Into<String>) {
        self.log_message(LogLevel::Error, message);
    }

    /// 记录调试信息
    pub fn debug(&mut self, message: impl Into<String>) {
        if self.config.enable_debug {
            self.log_message(LogLevel::Debug, message);
        }
    }

    /// 执行命令
    pub fn execute(&mut self, command: &str) -> Result<String> {
        // 记录到历史
        self.command_history.push_back(command.to_string());
        while self.command_history.len() > self.config.max_history {
            self.command_history.pop_front();
        }

        // 简化实现：返回命令本身
        log::info!("Executing console command: {}", command);

        Ok(format!("Executed: {}", command))
    }

    /// 获取所有消息
    pub fn messages(&self) -> &VecDeque<ConsoleMessage> {
        &self.messages
    }

    /// 获取特定级别的消息
    pub fn messages_by_level(&self, level: LogLevel) -> Vec<&ConsoleMessage> {
        self.messages.iter().filter(|m| m.level == level).collect()
    }

    /// 获取消息数量
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// 获取命令历史
    pub fn command_history(&self) -> &VecDeque<String> {
        &self.command_history
    }

    /// 清空消息
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// 清空历史
    pub fn clear_history(&mut self) {
        self.command_history.clear();
    }

    /// 获取配置
    pub fn config(&self) -> &ConsoleConfig {
        &self.config
    }
}

impl Default for Console {
    fn default() -> Self {
        Self::new()
    }
}

/// 控制台消息
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsoleMessage {
    /// 日志级别
    pub level: LogLevel,
    /// 消息内容
    pub message: String,
    /// 时间戳
    pub timestamp: std::time::SystemTime,
    /// 来源
    pub source: Option<String>,
}

/// 日志级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LogLevel {
    /// 普通日志
    Log,
    /// 信息
    Info,
    /// 警告
    Warn,
    /// 错误
    Error,
    /// 调试
    Debug,
}

impl LogLevel {
    /// 获取级别名称
    pub fn name(&self) -> &'static str {
        match self {
            LogLevel::Log => "LOG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Debug => "DEBUG",
        }
    }

    /// 获取严重度（数值越大越严重）
    pub fn severity(&self) -> u8 {
        match self {
            LogLevel::Debug => 0,
            LogLevel::Log => 1,
            LogLevel::Info => 2,
            LogLevel::Warn => 3,
            LogLevel::Error => 4,
        }
    }
}

/// 控制台配置
#[derive(Debug, Clone)]
pub struct ConsoleConfig {
    /// 最大消息数
    pub max_messages: usize,
    /// 最大历史记录数
    pub max_history: usize,
    /// 启用调试
    pub enable_debug: bool,
    /// 捕获panic
    pub capture_panic: bool,
}

impl Default for ConsoleConfig {
    fn default() -> Self {
        Self {
            max_messages: 1000,
            max_history: 100,
            enable_debug: true,
            capture_panic: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_console_creation() {
        let console = Console::new();
        assert_eq!(console.message_count(), 0);
    }

    #[test]
    fn test_log_levels() {
        let mut console = Console::new();

        console.log("Test log");
        console.info("Test info");
        console.warn("Test warn");
        console.error("Test error");

        assert_eq!(console.message_count(), 4);

        let errors = console.messages_by_level(LogLevel::Error);
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_debug_enabled() {
        let mut console = Console::with_config(ConsoleConfig {
            enable_debug: true,
            ..Default::default()
        });

        console.debug("Debug message");
        assert_eq!(console.message_count(), 1);
    }

    #[test]
    fn test_debug_disabled() {
        let mut console = Console::with_config(ConsoleConfig {
            enable_debug: false,
            ..Default::default()
        });

        console.debug("Debug message");
        assert_eq!(console.message_count(), 0);
    }

    #[test]
    fn test_max_messages() {
        let mut console = Console::with_config(ConsoleConfig {
            max_messages: 5,
            ..Default::default()
        });

        for i in 0..10 {
            console.log(format!("Message {}", i));
        }

        assert_eq!(console.message_count(), 5);
    }

    #[test]
    fn test_command_execution() {
        let mut console = Console::new();

        let result = console.execute("help");
        assert!(result.is_ok());
        assert!(result.unwrap().contains("help"));

        assert_eq!(console.command_history().len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut console = Console::new();
        console.log("Test");
        console.clear();

        assert_eq!(console.message_count(), 0);
    }

    #[test]
    fn test_log_level_severity() {
        assert!(LogLevel::Error.severity() > LogLevel::Warn.severity());
        assert!(LogLevel::Warn.severity() > LogLevel::Info.severity());
        assert!(LogLevel::Info.severity() > LogLevel::Debug.severity());
    }
}
