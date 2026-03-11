//! BrowerAI DevTools
//!
//! 开发者工具集，提供：
//! - DOM检查器 (DOM Inspector)
//! - 控制台API (Console API)
//! - 性能分析器 (Profiler)
//! - 网络监控 (Network Monitor)
//!
//! # 架构
//! ```text
//! DevTools
//! ├── Inspector: DOM树查看和编辑
//! ├── Console: 日志记录和执行
//! ├── Profiler: 性能分析
//! └── Network: 网络请求监控
//! ```
//!
//! # 示例
//! ```
//! use browerai_devtools::DevTools;
//!
//! let mut devtools = DevTools::new();
//! devtools.console().log("Hello from BrowerAI!");
//! ```

#![warn(missing_docs)]

use browerai_core::Result;

pub mod console;
pub mod inspector;
pub mod network;
pub mod profiler;

pub use console::{Console, ConsoleMessage, LogLevel};
pub use inspector::{DomInspector, InspectionResult, NodeInfo};
pub use network::{NetworkMonitor, NetworkRequest, NetworkResponse};
pub use profiler::{ProfileSummary, Profiler, TimingMark};

/// 开发者工具主入口
#[derive(Debug)]
pub struct DevTools {
    /// DOM检查器
    inspector: DomInspector,
    /// 控制台
    console: Console,
    /// 性能分析器
    profiler: Profiler,
    /// 网络监控
    network: NetworkMonitor,
    /// 是否启用
    enabled: bool,
}

impl DevTools {
    /// 创建新的开发者工具实例
    ///
    /// # 示例
    /// ```
    /// use browerai_devtools::DevTools;
    ///
    /// let devtools = DevTools::new();
    /// ```
    pub fn new() -> Self {
        Self {
            inspector: DomInspector::new(),
            console: Console::new(),
            profiler: Profiler::new(),
            network: NetworkMonitor::new(),
            enabled: true,
        }
    }

    /// 使用配置创建开发者工具
    pub fn with_config(config: DevToolsConfig) -> Self {
        Self {
            inspector: DomInspector::with_config(config.inspector_config),
            console: Console::with_config(config.console_config),
            profiler: Profiler::with_config(config.profiler_config),
            network: NetworkMonitor::with_config(config.network_config),
            enabled: config.enabled,
        }
    }

    /// 获取DOM检查器
    pub fn inspector(&self) -> &DomInspector {
        &self.inspector
    }

    /// 获取控制台（可变引用）
    pub fn console(&mut self) -> &mut Console {
        &mut self.console
    }

    /// 获取性能分析器
    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }

    /// 获取网络监控
    pub fn network(&self) -> &NetworkMonitor {
        &self.network
    }

    /// 检查是否启用
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// 启用/禁用开发者工具
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// 清除所有数据
    pub fn clear_all(&mut self) {
        self.console.clear();
        self.profiler.clear();
        self.network.clear();
    }

    /// 导出所有数据为JSON
    pub fn export_json(&self) -> Result<String> {
        let export = DevToolsExport {
            console_logs: self.console.messages().iter().cloned().collect(),
            network_requests: self.network.requests().iter().cloned().collect(),
            profile_summary: self.profiler.summary(),
        };

        serde_json::to_string_pretty(&export)
            .map_err(|e| browerai_core::BrowserError::parse(e.to_string()))
    }

    /// 获取统计信息
    pub fn stats(&self) -> DevToolsStats {
        DevToolsStats {
            console_message_count: self.console.message_count(),
            network_request_count: self.network.request_count(),
            profiler_sample_count: self.profiler.sample_count(),
            enabled: self.enabled,
        }
    }
}

impl Default for DevTools {
    fn default() -> Self {
        Self::new()
    }
}

/// 开发者工具配置
#[derive(Debug, Clone)]
pub struct DevToolsConfig {
    /// 是否启用
    pub enabled: bool,
    /// 检查器配置
    pub inspector_config: inspector::InspectorConfig,
    /// 控制台配置
    pub console_config: console::ConsoleConfig,
    /// 分析器配置
    pub profiler_config: profiler::ProfilerConfig,
    /// 网络配置
    pub network_config: network::NetworkConfig,
}

impl Default for DevToolsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            inspector_config: inspector::InspectorConfig::default(),
            console_config: console::ConsoleConfig::default(),
            profiler_config: profiler::ProfilerConfig::default(),
            network_config: network::NetworkConfig::default(),
        }
    }
}

/// 开发者工具导出数据
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DevToolsExport {
    /// 控制台日志
    pub console_logs: Vec<ConsoleMessage>,
    /// 网络请求
    pub network_requests: Vec<NetworkRequest>,
    /// 性能摘要
    pub profile_summary: profiler::ProfileSummary,
}

/// 开发者工具统计
#[derive(Debug, Clone, Copy, Default)]
pub struct DevToolsStats {
    /// 控制台消息数
    pub console_message_count: usize,
    /// 网络请求数
    pub network_request_count: usize,
    /// 性能样本数
    pub profiler_sample_count: usize,
    /// 是否启用
    pub enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_devtools_creation() {
        let devtools = DevTools::new();
        assert!(devtools.is_enabled());
    }

    #[test]
    fn test_devtools_console() {
        let mut devtools = DevTools::new();
        devtools.console().log("Test message");

        assert_eq!(devtools.stats().console_message_count, 1);
    }

    #[test]
    fn test_devtools_clear() {
        let mut devtools = DevTools::new();
        devtools.console().log("Test");
        devtools.clear_all();

        assert_eq!(devtools.stats().console_message_count, 0);
    }

    #[test]
    fn test_devtools_export() {
        let mut devtools = DevTools::new();
        devtools.console().log("Test message");

        let json = devtools.export_json();
        assert!(json.is_ok());
        assert!(json.unwrap().contains("Test message"));
    }

    #[test]
    fn test_devtools_enable_disable() {
        let mut devtools = DevTools::new();
        assert!(devtools.is_enabled());

        devtools.set_enabled(false);
        assert!(!devtools.is_enabled());
    }
}
