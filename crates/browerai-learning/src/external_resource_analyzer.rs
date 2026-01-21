/// 外部资源依赖管理
///
/// 处理现代网站的复杂外部依赖：
/// - 跨域资源和 CDN
/// - API 调用和认证
/// - 第三方脚本和库
/// - 资源的加载顺序和依赖关系
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 外部资源类型
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// JavaScript 脚本（可能被混淆）
    Script,
    /// CSS 样式表
    Stylesheet,
    /// 第三方库（jQuery, React 等）
    Library,
    /// API 端点
    ApiEndpoint,
    /// 图片资源
    Image,
    /// 字体资源
    Font,
    /// WebAssembly 模块
    WebAssembly,
    /// Service Worker 脚本
    ServiceWorker,
    /// Web Worker 脚本
    WebWorker,
    /// WebGL 资源
    WebGL,
    /// IndexedDB 数据库
    IndexedDB,
    /// LocalStorage 数据
    LocalStorage,
    /// 其他类型
    Other,
}

/// 资源依赖关系
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceDependency {
    /// 资源 URL
    pub url: String,
    /// 资源类型
    pub resource_type: ResourceType,
    /// 是否是跨域资源
    pub cross_origin: bool,
    /// CORS 配置
    pub cors_mode: CorsMode,
    /// 依赖的其他资源
    pub dependencies: Vec<String>,
    /// 加载顺序（值越小越早加载）
    pub load_order: u32,
    /// 是否是关键资源（阻塞渲染）
    pub is_critical: bool,
    /// 资源大小（字节）
    pub size_bytes: usize,
    /// 加载时间（毫秒）
    pub load_time_ms: f64,
    /// 缓存策略
    pub cache_strategy: CacheStrategy,
    /// 认证需求
    pub auth_required: bool,
    /// 认证方式
    pub auth_type: AuthType,
}

/// CORS 模式
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CorsMode {
    /// 不需要 CORS（同域）
    SameOrigin,
    /// CORS 允许
    Cors,
    /// CORS 不允许
    NoCors,
    /// 不确定
    Unknown,
}

/// 缓存策略
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CacheStrategy {
    /// 不缓存
    NoCache,
    /// 缓存一小时
    ShortTerm,
    /// 缓存一天
    MediumTerm,
    /// 缓存一年
    LongTerm,
    /// 永久缓存（版本控制）
    Permanent,
}

/// 认证类型
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum AuthType {
    /// 无需认证
    None,
    /// Bearer Token
    BearerToken,
    /// API Key
    ApiKey,
    /// 基本认证（用户名密码）
    BasicAuth,
    /// OAuth 2.0
    OAuth2,
    /// JWT
    Jwt,
    /// 自定义认证
    Custom,
    /// 未知
    Unknown,
}

/// 外部资源依赖图
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExternalResourceGraph {
    /// 所有资源
    pub resources: HashMap<String, ResourceDependency>,

    /// 域名统计
    pub domains: HashMap<String, DomainStats>,

    /// 加载链（按顺序）
    pub load_chain: Vec<String>,

    /// 关键资源列表
    pub critical_resources: Vec<String>,

    /// 跨域资源列表
    pub cross_origin_resources: Vec<String>,

    /// 需要认证的资源
    pub auth_required_resources: Vec<String>,

    /// 总资源大小
    pub total_size_bytes: usize,

    /// 总加载时间
    pub total_load_time_ms: f64,

    /// 独特域名数
    pub unique_domains: usize,
}

/// 域名统计
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainStats {
    /// 域名
    pub domain: String,
    /// 该域名下的资源数
    pub resource_count: usize,
    /// 总字节数
    pub total_bytes: usize,
    /// 是否跨域
    pub is_cross_origin: bool,
    /// 加载速度（毫秒）
    pub avg_load_time_ms: f64,
    /// 该域名支持的认证方式
    pub auth_types: Vec<AuthType>,
}

/// 外部资源分析器
pub struct ExternalResourceAnalyzer;

impl ExternalResourceAnalyzer {
    /// 分析网页中的所有外部资源
    pub fn analyze_resources(html: &str, js_code: &str) -> Result<ExternalResourceGraph> {
        log::info!("🔗 分析外部资源依赖关系...");

        let mut resources = HashMap::new();
        let mut domains = HashMap::new();
        let mut critical_resources = Vec::new();
        let mut cross_origin_resources = Vec::new();
        let mut auth_required_resources = Vec::new();

        // 第1步：从 HTML 中提取资源
        Self::extract_html_resources(html, &mut resources, &mut critical_resources)?;

        // 第2步：从 JavaScript 中提取 API 调用和动态加载
        Self::extract_js_resources(js_code, &mut resources)?;

        // 第3步：分析资源的依赖关系
        Self::analyze_dependencies(&resources)?;

        // 第4步：计算加载顺序
        let load_chain = Self::calculate_load_chain(&resources)?;

        // 第5步：统计域名信息
        Self::analyze_domains(&resources, &mut domains)?;

        // 第6步：识别跨域和认证资源
        for (url, resource) in &resources {
            if resource.cross_origin {
                cross_origin_resources.push(url.clone());
            }
            if resource.auth_required {
                auth_required_resources.push(url.clone());
            }
        }

        let total_size_bytes = resources.values().map(|r| r.size_bytes).sum();
        let total_load_time_ms = resources.values().map(|r| r.load_time_ms).sum();

        log::info!(
            "  ✓ 发现 {} 个外部资源，{} 个域名，跨域 {} 个，需认证 {} 个",
            resources.len(),
            domains.len(),
            cross_origin_resources.len(),
            auth_required_resources.len()
        );

        Ok(ExternalResourceGraph {
            resources,
            domains: domains.clone(),
            load_chain,
            critical_resources,
            cross_origin_resources,
            auth_required_resources,
            total_size_bytes,
            total_load_time_ms,
            unique_domains: domains.len(),
        })
    }

    /// 从 HTML 中提取资源
    fn extract_html_resources(
        html: &str,
        resources: &mut HashMap<String, ResourceDependency>,
        critical_resources: &mut Vec<String>,
    ) -> Result<()> {
        // 提取 script 标签
        let script_pattern = regex::Regex::new(r#"<script[^>]*src="([^"]+)"[^>]*>"#)?;
        for cap in script_pattern.captures_iter(html) {
            let url = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            let is_async = html[..cap.get(0).unwrap().start()].contains("async");

            resources.insert(
                url.to_string(),
                ResourceDependency {
                    url: url.to_string(),
                    resource_type: ResourceType::Script,
                    cross_origin: !url.contains("localhost"),
                    cors_mode: CorsMode::Unknown,
                    dependencies: Vec::new(),
                    load_order: if is_async { 50 } else { 10 },
                    is_critical: !is_async,
                    size_bytes: 0,
                    load_time_ms: 0.0,
                    cache_strategy: CacheStrategy::MediumTerm,
                    auth_required: false,
                    auth_type: AuthType::None,
                },
            );

            if !is_async {
                critical_resources.push(url.to_string());
            }
        }

        // 提取 link 标签（CSS、字体等）
        let link_pattern = regex::Regex::new(r#"<link[^>]*href="([^"]+)"[^>]*>"#)?;
        for cap in link_pattern.captures_iter(html) {
            let url = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            resources.insert(
                url.to_string(),
                ResourceDependency {
                    url: url.to_string(),
                    resource_type: ResourceType::Stylesheet,
                    cross_origin: !url.contains("localhost"),
                    cors_mode: CorsMode::Unknown,
                    dependencies: Vec::new(),
                    load_order: 5,
                    is_critical: true,
                    size_bytes: 0,
                    load_time_ms: 0.0,
                    cache_strategy: CacheStrategy::LongTerm,
                    auth_required: false,
                    auth_type: AuthType::None,
                },
            );
            critical_resources.push(url.to_string());
        }

        // 提取 img 标签
        let img_pattern = regex::Regex::new(r#"<img[^>]*src="([^"]+)"[^>]*>"#)?;
        for cap in img_pattern.captures_iter(html) {
            let url = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            resources.insert(
                url.to_string(),
                ResourceDependency {
                    url: url.to_string(),
                    resource_type: ResourceType::Image,
                    cross_origin: !url.contains("localhost"),
                    cors_mode: CorsMode::NoCors,
                    dependencies: Vec::new(),
                    load_order: 60,
                    is_critical: false,
                    size_bytes: 0,
                    load_time_ms: 0.0,
                    cache_strategy: CacheStrategy::LongTerm,
                    auth_required: false,
                    auth_type: AuthType::None,
                },
            );
        }

        Ok(())
    }

    /// 从 JavaScript 中提取资源
    fn extract_js_resources(
        js_code: &str,
        resources: &mut HashMap<String, ResourceDependency>,
    ) -> Result<()> {
        // 提取 fetch() 和 XMLHttpRequest 调用
        let fetch_pattern = regex::Regex::new(r#"fetch\s*\(\s*["']([^"']+)["']"#)?;
        for cap in fetch_pattern.captures_iter(js_code) {
            let url = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            resources
                .entry(url.to_string())
                .or_insert_with(|| ResourceDependency {
                    url: url.to_string(),
                    resource_type: ResourceType::ApiEndpoint,
                    cross_origin: true,
                    cors_mode: CorsMode::Cors,
                    dependencies: Vec::new(),
                    load_order: 100,
                    is_critical: false,
                    size_bytes: 0,
                    load_time_ms: 0.0,
                    cache_strategy: CacheStrategy::NoCache,
                    auth_required: false,
                    auth_type: AuthType::Unknown,
                });
        }

        // 提取 import 语句（动态加载）
        let import_pattern = regex::Regex::new(r#"import\s*\(\s*["']([^"']+)["']"#)?;
        for cap in import_pattern.captures_iter(js_code) {
            let url = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            resources
                .entry(url.to_string())
                .or_insert_with(|| ResourceDependency {
                    url: url.to_string(),
                    resource_type: ResourceType::Script,
                    cross_origin: !url.contains("localhost"),
                    cors_mode: CorsMode::Unknown,
                    dependencies: Vec::new(),
                    load_order: 50,
                    is_critical: false,
                    size_bytes: 0,
                    load_time_ms: 0.0,
                    cache_strategy: CacheStrategy::MediumTerm,
                    auth_required: false,
                    auth_type: AuthType::None,
                });
        }

        // 提取 new Worker() 调用
        let worker_pattern = regex::Regex::new(r#"new\s+Worker\s*\(\s*["']([^"']+)["']"#)?;
        for cap in worker_pattern.captures_iter(js_code) {
            let url = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            resources.insert(
                url.to_string(),
                ResourceDependency {
                    url: url.to_string(),
                    resource_type: ResourceType::WebWorker,
                    cross_origin: false,
                    cors_mode: CorsMode::SameOrigin,
                    dependencies: Vec::new(),
                    load_order: 40,
                    is_critical: false,
                    size_bytes: 0,
                    load_time_ms: 0.0,
                    cache_strategy: CacheStrategy::MediumTerm,
                    auth_required: false,
                    auth_type: AuthType::None,
                },
            );
        }

        // 提取 WebAssembly 加载
        let wasm_pattern = regex::Regex::new(
            r#"WebAssembly\.(?:instantiate|instantiateStreaming)\s*\(\s*["']([^"']+)["']|wasm\(\s*["']([^"']+)["']"#,
        )?;
        for cap in wasm_pattern.captures_iter(js_code) {
            let url = cap
                .get(1)
                .or_else(|| cap.get(2))
                .map(|m| m.as_str())
                .unwrap_or("");

            if url.is_empty() {
                continue;
            }

            resources.insert(
                url.to_string(),
                ResourceDependency {
                    url: url.to_string(),
                    resource_type: ResourceType::WebAssembly,
                    cross_origin: false,
                    cors_mode: CorsMode::SameOrigin,
                    dependencies: Vec::new(),
                    load_order: 20,
                    is_critical: true,
                    size_bytes: 0,
                    load_time_ms: 0.0,
                    cache_strategy: CacheStrategy::LongTerm,
                    auth_required: false,
                    auth_type: AuthType::None,
                },
            );
        }

        Ok(())
    }

    /// 分析资源之间的依赖关系
    fn analyze_dependencies(_resources: &HashMap<String, ResourceDependency>) -> Result<()> {
        // 这里可以添加更复杂的依赖关系分析
        // 例如：哪些 API 依赖哪些认证令牌等
        log::debug!("分析资源依赖关系");
        Ok(())
    }

    /// 计算资源加载顺序
    fn calculate_load_chain(
        resources: &HashMap<String, ResourceDependency>,
    ) -> Result<Vec<String>> {
        let mut chain: Vec<_> = resources
            .iter()
            .map(|(url, dep)| (url.clone(), dep.load_order))
            .collect();

        chain.sort_by_key(|(_url, order)| *order);

        Ok(chain.into_iter().map(|(url, _)| url).collect())
    }

    /// 分析域名统计
    fn analyze_domains(
        resources: &HashMap<String, ResourceDependency>,
        domains: &mut HashMap<String, DomainStats>,
    ) -> Result<()> {
        for (url, resource) in resources {
            let domain = Self::extract_domain(url);

            domains
                .entry(domain.clone())
                .and_modify(|stats| {
                    stats.resource_count += 1;
                    stats.total_bytes += resource.size_bytes;
                    stats.avg_load_time_ms = (stats.avg_load_time_ms + resource.load_time_ms) / 2.0;
                    if resource.auth_required && !stats.auth_types.contains(&resource.auth_type) {
                        stats.auth_types.push(resource.auth_type.clone());
                    }
                })
                .or_insert_with(|| DomainStats {
                    domain,
                    resource_count: 1,
                    total_bytes: resource.size_bytes,
                    is_cross_origin: resource.cross_origin,
                    avg_load_time_ms: resource.load_time_ms,
                    auth_types: if resource.auth_required {
                        vec![resource.auth_type.clone()]
                    } else {
                        Vec::new()
                    },
                });
        }

        Ok(())
    }

    /// 从 URL 中提取域名
    fn extract_domain(url: &str) -> String {
        if let Ok(parsed) = url.parse::<url::Url>() {
            parsed.host_str().unwrap_or("unknown").to_string()
        } else {
            "unknown".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_domain() {
        let url = "https://example.com/api/users";
        let domain = ExternalResourceAnalyzer::extract_domain(url);
        assert_eq!(domain, "example.com");
    }

    #[test]
    fn test_resource_dependency_creation() {
        let dep = ResourceDependency {
            url: "https://api.example.com/data".to_string(),
            resource_type: ResourceType::ApiEndpoint,
            cross_origin: true,
            cors_mode: CorsMode::Cors,
            dependencies: Vec::new(),
            load_order: 100,
            is_critical: false,
            size_bytes: 1024,
            load_time_ms: 250.0,
            cache_strategy: CacheStrategy::NoCache,
            auth_required: true,
            auth_type: AuthType::BearerToken,
        };

        assert_eq!(dep.resource_type, ResourceType::ApiEndpoint);
        assert!(dep.cross_origin);
        assert!(dep.auth_required);
    }
}
