//! BrowerAI Core - 核心类型、错误处理和 Traits
//!
//! 这是 BrowerAI 的基础 crate，提供：
//! - 核心错误类型和 Result 别名
//! - 解析器、渲染器、AI 模型的 Traits
//! - 共享类型和配置
//!
//! 设计原则：
//! - 零外部依赖（除基础库外）
//! - 类型安全，编译期错误检查
//! - 可扩展的 trait 系统

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod error;
pub mod traits;
pub mod types;
pub mod config;
pub mod metrics;
pub mod source_loc;

// 重新导出最常用的类型
pub use error::{BrowserError, Result, ErrorKind};
pub use traits::{Parser, Renderer, AiModel, Deobfuscator, Learner};
pub use types::{CodeType, BrowserConfig, FeatureFlags};
pub use config::{AiConfig, CacheConfig, NetworkConfig};
pub use metrics::{Metric, MetricType, MetricsDashboard};
pub use source_loc::{SourceLocation, SourceSpan, SourceInfo};

/// 版本信息
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// 构建信息
pub const BUILD_INFO: &str = concat!(
    "BrowerAI Core v", env!("CARGO_PKG_VERSION"),
    " (", env!("CARGO_PKG_REPOSITORY"), ")"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
        assert!(VERSION.starts_with("0.2"));
    }

    #[test]
    fn test_build_info() {
        assert!(BUILD_INFO.contains("BrowerAI Core"));
    }
}
