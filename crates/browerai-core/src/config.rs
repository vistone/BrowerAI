//! 配置系统
//!
//! 提供 BrowerAI 各组件的配置类型

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// AI 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiConfig {
    /// 是否启用 AI
    pub enabled: bool,
    /// 模型目录
    pub model_dir: PathBuf,
    /// 默认模型名称
    pub default_model: String,
    /// 推理线程数
    pub inference_threads: usize,
    /// 批处理大小
    pub batch_size: usize,
    /// 是否启用 GPU
    pub use_gpu: bool,
    /// GPU 设备 ID
    pub gpu_device_id: i32,
    /// 是否启用热重载
    pub enable_hot_reload: bool,
    /// 热重载检查间隔（秒）
    pub hot_reload_interval_secs: u64,
    /// 回退策略
    pub fallback_policy: FallbackPolicy,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_dir: PathBuf::from("models"),
            default_model: "fast_enhanced.onnx".to_string(),
            inference_threads: 4,
            batch_size: 32,
            use_gpu: false,
            gpu_device_id: 0,
            enable_hot_reload: true,
            hot_reload_interval_secs: 30,
            fallback_policy: FallbackPolicy::UseBaseParser,
        }
    }
}

impl AiConfig {
    /// 启用 AI
    pub fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// 设置模型目录
    pub fn with_model_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.model_dir = dir.into();
        self
    }

    /// 启用 GPU
    pub fn with_gpu(mut self, device_id: i32) -> Self {
        self.use_gpu = true;
        self.gpu_device_id = device_id;
        self
    }
}

/// 回退策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FallbackPolicy {
    /// 使用基础解析器
    UseBaseParser,
    /// 返回错误
    ReturnError,
    /// 重试
    Retry,
}

/// 缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// 是否启用 L1 缓存（内存）
    pub enable_l1: bool,
    /// L1 缓存大小（条目数）
    pub l1_capacity: usize,
    /// 是否启用 L2 缓存（Redis）
    pub enable_l2: bool,
    /// L2 缓存（Redis）URL
    pub l2_redis_url: Option<String>,
    /// 是否启用 L3 缓存（磁盘）
    pub enable_l3: bool,
    /// L3 缓存目录
    pub l3_dir: PathBuf,
    /// 默认 TTL（秒）
    pub default_ttl_secs: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_l1: true,
            l1_capacity: 10000,
            enable_l2: false,
            l2_redis_url: None,
            enable_l3: false,
            l3_dir: PathBuf::from(".cache"),
            default_ttl_secs: 3600,
        }
    }
}

impl CacheConfig {
    /// 启用 L2 缓存（Redis）
    pub fn with_redis(mut self, url: impl Into<String>) -> Self {
        self.enable_l2 = true;
        self.l2_redis_url = Some(url.into());
        self
    }

    /// 启用 L3 缓存（磁盘）
    pub fn with_disk_cache(mut self, dir: impl Into<PathBuf>) -> Self {
        self.enable_l3 = true;
        self.l3_dir = dir.into();
        self
    }
}

/// 网络配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// 连接超时（秒）
    pub connect_timeout_secs: u64,
    /// 请求超时（秒）
    pub request_timeout_secs: u64,
    /// 是否跟随重定向
    pub follow_redirects: bool,
    /// 最大重定向次数
    pub max_redirects: usize,
    /// 用户代理
    pub user_agent: String,
    /// 是否验证 SSL 证书
    pub verify_ssl: bool,
    /// 代理配置
    pub proxy: Option<ProxyConfig>,
    /// 请求头
    pub default_headers: HashMap<String, String>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            connect_timeout_secs: 10,
            request_timeout_secs: 30,
            follow_redirects: true,
            max_redirects: 10,
            user_agent: format!("BrowerAI/{}", crate::VERSION),
            verify_ssl: true,
            proxy: None,
            default_headers: HashMap::new(),
        }
    }
}

/// 代理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfig {
    /// 代理类型
    pub proxy_type: ProxyType,
    /// 代理主机
    pub host: String,
    /// 代理端口
    pub port: u16,
    /// 用户名（如果需要认证）
    pub username: Option<String>,
    /// 密码（如果需要认证）
    pub password: Option<String>,
}

/// 代理类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProxyType {
    /// HTTP 代理
    Http,
    /// HTTPS 代理
    Https,
    /// SOCKS5 代理
    Socks5,
}

/// 解析配置
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParseConfig {
    /// HTML 解析配置
    pub html: HtmlParseConfig,
    /// CSS 解析配置
    pub css: CssParseConfig,
    /// JS 解析配置
    pub js: JsParseConfig,
}

/// HTML 解析配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlParseConfig {
    /// 是否忽略解析错误
    pub ignore_errors: bool,
    /// 是否解析脚本内容
    pub parse_scripts: bool,
    /// 是否解析样式内容
    pub parse_styles: bool,
}

impl Default for HtmlParseConfig {
    fn default() -> Self {
        Self {
            ignore_errors: true,
            parse_scripts: true,
            parse_styles: true,
        }
    }
}

/// CSS 解析配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CssParseConfig {
    /// 是否忽略无效规则
    pub ignore_invalid_rules: bool,
    /// 是否展开嵌套规则
    pub expand_nesting: bool,
}

impl Default for CssParseConfig {
    fn default() -> Self {
        Self {
            ignore_invalid_rules: true,
            expand_nesting: true,
        }
    }
}

/// JS 解析配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsParseConfig {
    /// 是否解析 TypeScript
    pub parse_typescript: bool,
    /// 是否解析 JSX
    pub parse_jsx: bool,
    /// 目标 ECMAScript 版本
    pub target_version: EcmaVersion,
}

impl Default for JsParseConfig {
    fn default() -> Self {
        Self {
            parse_typescript: true,
            parse_jsx: true,
            target_version: EcmaVersion::ES2022,
        }
    }
}

/// ECMAScript 版本
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EcmaVersion {
    /// ES5
    ES5,
    /// ES6/ES2015
    ES2015,
    /// ES2017
    ES2017,
    /// ES2019
    ES2019,
    /// ES2020
    ES2020,
    /// ES2021
    ES2021,
    /// ES2022
    ES2022,
    /// ES2023
    ES2023,
    /// ESNext
    ESNext,
}

/// 渲染配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderConfig {
    /// 视口宽度
    pub viewport_width: u32,
    /// 视口高度
    pub viewport_height: u32,
    /// 设备像素比
    pub device_pixel_ratio: f32,
    /// 是否启用 CSS 动画
    pub enable_animations: bool,
    /// 是否启用 WebGL
    pub enable_webgl: bool,
    /// 输出格式
    pub output_format: OutputFormat,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            viewport_width: 1920,
            viewport_height: 1080,
            device_pixel_ratio: 1.0,
            enable_animations: true,
            enable_webgl: false,
            output_format: OutputFormat::Bitmap,
        }
    }
}

/// 输出格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// 位图（PNG/JPEG）
    Bitmap,
    /// PDF
    Pdf,
    /// SVG
    Svg,
    /// 文本
    Text,
}

/// 学习配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// 是否启用学习
    pub enabled: bool,
    /// 数据目录
    pub data_dir: PathBuf,
    /// 最大并发学习数
    pub max_concurrent: usize,
    /// 学习超时（秒）
    pub timeout_secs: u64,
    /// 质量阈值（0.0 - 1.0）
    pub quality_threshold: f32,
    /// 是否自动保存
    pub auto_save: bool,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            data_dir: PathBuf::from("data/learning"),
            max_concurrent: 4,
            timeout_secs: 300,
            quality_threshold: 0.7,
            auto_save: true,
        }
    }
}

/// 全局配置
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlobalConfig {
    /// AI 配置
    pub ai: AiConfig,
    /// 缓存配置
    pub cache: CacheConfig,
    /// 网络配置
    pub network: NetworkConfig,
    /// 解析配置
    pub parse: ParseConfig,
    /// 渲染配置
    pub render: RenderConfig,
    /// 学习配置
    pub learning: LearningConfig,
    /// 额外配置
    pub extra: HashMap<String, serde_json::Value>,
}

impl GlobalConfig {
    /// 从文件加载配置（需要启用 config-file feature）
    #[cfg(feature = "config-file")]
    pub fn from_file(path: impl AsRef<std::path::Path>) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content).map_err(|e| {
            crate::error::BrowserError::config(format!("Failed to parse config: {}", e))
        })?;
        Ok(config)
    }

    /// 保存配置到文件（需要启用 config-file feature）
    #[cfg(feature = "config-file")]
    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> crate::error::Result<()> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            crate::error::BrowserError::config(format!("Failed to serialize config: {}", e))
        })?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_config_default() {
        let config = AiConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.inference_threads, 4);
    }

    #[test]
    fn test_cache_config_with_redis() {
        let config = CacheConfig::default().with_redis("redis://localhost:6379");
        assert!(config.enable_l2);
        assert_eq!(
            config.l2_redis_url,
            Some("redis://localhost:6379".to_string())
        );
    }

    #[test]
    fn test_network_config_default() {
        let config = NetworkConfig::default();
        assert!(config.follow_redirects);
        assert_eq!(config.max_redirects, 10);
    }

    #[test]
    #[cfg(feature = "config-file")]
    fn test_global_config_save_load() {
        let config = GlobalConfig::default();
        let temp_path = "/tmp/test_browerai_config.toml";

        config.save_to_file(temp_path).unwrap();
        let loaded = GlobalConfig::from_file(temp_path).unwrap();

        assert_eq!(config.ai.enabled, loaded.ai.enabled);
    }
}
