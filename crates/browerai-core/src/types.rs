//! 核心类型定义
//!
//! 提供 BrowerAI 共享的基础类型

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 代码类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodeType {
    /// HTML
    Html,
    /// CSS
    Css,
    /// JavaScript
    JavaScript,
    /// TypeScript
    TypeScript,
    /// JSON
    Json,
    /// WebAssembly
    Wasm,
    /// 未知类型
    Unknown,
}

impl CodeType {
    /// 从文件扩展名检测代码类型
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "html" | "htm" => CodeType::Html,
            "css" => CodeType::Css,
            "js" => CodeType::JavaScript,
            "ts" | "tsx" => CodeType::TypeScript,
            "json" => CodeType::Json,
            "wasm" => CodeType::Wasm,
            _ => CodeType::Unknown,
        }
    }

    /// 获取 MIME 类型
    pub fn mime_type(&self) -> &'static str {
        match self {
            CodeType::Html => "text/html",
            CodeType::Css => "text/css",
            CodeType::JavaScript => "application/javascript",
            CodeType::TypeScript => "application/typescript",
            CodeType::Json => "application/json",
            CodeType::Wasm => "application/wasm",
            CodeType::Unknown => "application/octet-stream",
        }
    }

    /// 是否是 Web 前端代码
    pub fn is_frontend(&self) -> bool {
        matches!(self, CodeType::Html | CodeType::Css | CodeType::JavaScript | CodeType::TypeScript)
    }
}

impl std::fmt::Display for CodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodeType::Html => write!(f, "HTML"),
            CodeType::Css => write!(f, "CSS"),
            CodeType::JavaScript => write!(f, "JavaScript"),
            CodeType::TypeScript => write!(f, "TypeScript"),
            CodeType::Json => write!(f, "JSON"),
            CodeType::Wasm => write!(f, "WebAssembly"),
            CodeType::Unknown => write!(f, "Unknown"),
        }
    }
}

/// 浏览器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserConfig {
    /// 视口宽度
    pub viewport_width: u32,
    /// 视口高度
    pub viewport_height: u32,
    /// 设备像素比
    pub device_pixel_ratio: f32,
    /// 用户代理字符串
    pub user_agent: String,
    /// 功能标志
    pub features: FeatureFlags,
    /// 超时配置（秒）
    pub timeout_secs: u64,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            viewport_width: 1920,
            viewport_height: 1080,
            device_pixel_ratio: 1.0,
            user_agent: format!("BrowerAI/{}", crate::VERSION),
            features: FeatureFlags::default(),
            timeout_secs: 30,
        }
    }
}

impl BrowserConfig {
    /// 创建移动端配置
    pub fn mobile() -> Self {
        Self {
            viewport_width: 375,
            viewport_height: 667,
            device_pixel_ratio: 2.0,
            user_agent: format!("BrowerAI/{}/Mobile", crate::VERSION),
            ..Default::default()
        }
    }

    /// 创建平板配置
    pub fn tablet() -> Self {
        Self {
            viewport_width: 768,
            viewport_height: 1024,
            device_pixel_ratio: 2.0,
            user_agent: format!("BrowerAI/{}/Tablet", crate::VERSION),
            ..Default::default()
        }
    }

    /// 启用 AI 功能
    pub fn with_ai(mut self) -> Self {
        self.features.enable_ai = true;
        self
    }

    /// 启用缓存
    pub fn with_cache(mut self) -> Self {
        self.features.enable_cache = true;
        self
    }
}

/// 功能标志
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    /// 启用 AI 增强
    pub enable_ai: bool,
    /// 启用缓存
    pub enable_cache: bool,
    /// 启用 JavaScript 执行
    pub enable_javascript: bool,
    /// 启用图片加载
    pub enable_images: bool,
    /// 启用 CSS 动画
    pub enable_animations: bool,
    /// 启用 WebGL
    pub enable_webgl: bool,
    /// 启用 WebAssembly
    pub enable_wasm: bool,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            enable_ai: false,
            enable_cache: true,
            enable_javascript: true,
            enable_images: true,
            enable_animations: true,
            enable_webgl: false,
            enable_wasm: false,
        }
    }
}

/// 网站功能类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WebsiteFeature {
    /// 搜索
    Search,
    /// 登录
    Login,
    /// 注册
    Register,
    /// 购物车
    ShoppingCart,
    /// 支付
    Payment,
    /// 导航
    Navigation,
    /// 内容展示
    ContentDisplay,
    /// 表单提交
    FormSubmission,
    /// 媒体播放
    MediaPlayback,
    /// 文件上传
    FileUpload,
    /// 社交互动
    SocialInteraction,
    /// 数据可视化
    DataVisualization,
}

impl WebsiteFeature {
    /// 是否是核心功能（必须保留）
    pub fn is_core(&self) -> bool {
        matches!(
            self,
            WebsiteFeature::Login
                | WebsiteFeature::Payment
                | WebsiteFeature::ShoppingCart
                | WebsiteFeature::FormSubmission
        )
    }

    /// 获取功能优先级（1-10）
    pub fn priority(&self) -> u8 {
        match self {
            WebsiteFeature::Payment => 10,
            WebsiteFeature::Login => 9,
            WebsiteFeature::ShoppingCart => 8,
            WebsiteFeature::FormSubmission => 7,
            WebsiteFeature::Search => 6,
            WebsiteFeature::Navigation => 5,
            _ => 3,
        }
    }
}

/// 页面类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PageType {
    /// 首页
    Homepage,
    /// 产品列表
    ProductList,
    /// 产品详情
    ProductDetail,
    /// 文章
    Article,
    /// 表单
    Form,
    /// 仪表板
    Dashboard,
    /// 搜索
    Search,
    /// 登录
    Login,
    /// 结算
    Checkout,
    /// 个人资料
    Profile,
    /// 未知
    Unknown,
}

/// 混淆技术类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObfuscationTechnique {
    /// 字符串数组
    StringArray,
    /// 控制流平坦化
    ControlFlowFlattening,
    /// 死代码注入
    DeadCodeInjection,
    /// 标识符混淆
    IdentifierMangling,
    /// 不透明谓词
    OpaquePredicate,
    /// 编码混淆
    EncodingObfuscation,
    /// 自我保护
    SelfDefending,
    /// 域名锁定
    DomainLock,
    /// 反调试
    DebugProtection,
    /// WebAssembly 打包
    WasmPacking,
    /// 数组旋转
    ArrayRotation,
    /// 仅压缩
    MinifyOnly,
}

impl ObfuscationTechnique {
    /// 获取技术名称
    pub fn name(&self) -> &'static str {
        match self {
            ObfuscationTechnique::StringArray => "字符串数组",
            ObfuscationTechnique::ControlFlowFlattening => "控制流平坦化",
            ObfuscationTechnique::DeadCodeInjection => "死代码注入",
            ObfuscationTechnique::IdentifierMangling => "标识符混淆",
            ObfuscationTechnique::OpaquePredicate => "不透明谓词",
            ObfuscationTechnique::EncodingObfuscation => "编码混淆",
            ObfuscationTechnique::SelfDefending => "自我保护",
            ObfuscationTechnique::DomainLock => "域名锁定",
            ObfuscationTechnique::DebugProtection => "反调试",
            ObfuscationTechnique::WasmPacking => "WASM打包",
            ObfuscationTechnique::ArrayRotation => "数组旋转",
            ObfuscationTechnique::MinifyOnly => "仅压缩",
        }
    }

    /// 检测难度（1-10）
    pub fn detection_difficulty(&self) -> u8 {
        match self {
            ObfuscationTechnique::MinifyOnly => 1,
            ObfuscationTechnique::StringArray => 3,
            ObfuscationTechnique::IdentifierMangling => 3,
            ObfuscationTechnique::EncodingObfuscation => 4,
            ObfuscationTechnique::DeadCodeInjection => 5,
            ObfuscationTechnique::ArrayRotation => 5,
            ObfuscationTechnique::ControlFlowFlattening => 7,
            ObfuscationTechnique::OpaquePredicate => 7,
            ObfuscationTechnique::SelfDefending => 8,
            ObfuscationTechnique::DebugProtection => 8,
            ObfuscationTechnique::DomainLock => 9,
            ObfuscationTechnique::WasmPacking => 10,
        }
    }
}

/// 内容类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    /// 内容类型
    pub content_type: CodeType,
    /// 原始内容
    pub raw: String,
    /// 来源 URL
    pub source_url: Option<String>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl Content {
    /// 创建新的内容
    pub fn new(content_type: CodeType, raw: impl Into<String>) -> Self {
        Self {
            content_type,
            raw: raw.into(),
            source_url: None,
            metadata: HashMap::new(),
        }
    }

    /// 设置来源 URL
    pub fn with_source(mut self, url: impl Into<String>) -> Self {
        self.source_url = Some(url.into());
        self
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// 内容大小（字节）
    pub fn size(&self) -> usize {
        self.raw.len()
    }

    /// 行数
    pub fn line_count(&self) -> usize {
        self.raw.lines().count()
    }
}

/// 版本信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    /// 主版本号
    pub major: u32,
    /// 次版本号
    pub minor: u32,
    /// 修订号
    pub patch: u32,
    /// 预发布标识
    pub prerelease: Option<String>,
    /// 构建元数据
    pub build: Option<String>,
}

impl VersionInfo {
    /// 解析版本字符串
    pub fn parse(version: &str) -> Option<Self> {
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() < 3 {
            return None;
        }

        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;
        
        // 处理 patch 可能包含 prerelease
        let patch_part = parts[2];
        let (patch, prerelease) = if let Some(idx) = patch_part.find('-') {
            let (patch_str, pre) = patch_part.split_at(idx);
            (patch_str.parse().ok()?, Some(pre[1..].to_string()))
        } else {
            (patch_part.parse().ok()?, None)
        };

        Some(Self {
            major,
            minor,
            patch,
            prerelease,
            build: None,
        })
    }

    /// 转换为字符串
    pub fn to_string(&self) -> String {
        let mut s = format!("{}.{}.{}", self.major, self.minor, self.patch);
        if let Some(ref pre) = self.prerelease {
            s.push_str(&format!("-{}", pre));
        }
        if let Some(ref build) = self.build {
            s.push_str(&format!("+{}", build));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_type_from_extension() {
        assert_eq!(CodeType::from_extension("html"), CodeType::Html);
        assert_eq!(CodeType::from_extension("css"), CodeType::Css);
        assert_eq!(CodeType::from_extension("js"), CodeType::JavaScript);
        assert_eq!(CodeType::from_extension("unknown"), CodeType::Unknown);
    }

    #[test]
    fn test_browser_config_default() {
        let config = BrowserConfig::default();
        assert_eq!(config.viewport_width, 1920);
        assert_eq!(config.viewport_height, 1080);
        assert!(!config.features.enable_ai);
        assert!(config.features.enable_javascript);
    }

    #[test]
    fn test_website_feature_priority() {
        assert_eq!(WebsiteFeature::Payment.priority(), 10);
        assert_eq!(WebsiteFeature::Login.priority(), 9);
        assert!(WebsiteFeature::Payment.is_core());
    }

    #[test]
    fn test_version_info_parse() {
        let v = VersionInfo::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);

        let v = VersionInfo::parse("0.2.0-alpha").unwrap();
        assert_eq!(v.prerelease, Some("alpha".to_string()));
    }
}
