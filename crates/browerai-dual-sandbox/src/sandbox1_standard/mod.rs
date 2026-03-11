//! 沙盒1: 标准渲染引擎
//!
//! 像 Chrome 一样完整解析网站，获取所有资源

pub use crate::common::*;
use anyhow::{Context, Result};
use std::collections::HashMap;

/// 标准渲染沙盒
pub struct StandardSandbox {
    /// HTTP 客户端
    client: reqwest::Client,
}

/// 渲染后的页面
#[derive(Debug, Clone)]
pub struct RenderedPage {
    /// 原始 URL
    pub url: String,
    /// HTML 内容
    pub html: String,
    /// CSS 资源
    pub css_resources: Vec<CssResource>,
    /// JS 资源
    pub js_resources: Vec<JsResource>,
    /// DOM 树
    pub dom_tree: DomTree,
    /// 渲染统计
    pub stats: RenderStats,
}

/// CSS 资源
#[derive(Debug, Clone)]
pub struct CssResource {
    /// URL
    pub url: String,
    /// 内容
    pub content: String,
    /// 解析后的规则
    pub rules: Vec<CssRule>,
}

/// JS 资源
#[derive(Debug, Clone)]
pub struct JsResource {
    /// URL
    pub url: String,
    /// 内容
    pub content: String,
    /// 是否混淆
    pub is_obfuscated: bool,
    /// 提取的函数
    pub functions: Vec<JsFunction>,
}

/// DOM 树
#[derive(Debug, Clone)]
pub struct DomTree {
    /// 根节点
    pub root: DomNode,
    /// 所有节点 (用于快速查找)
    pub all_nodes: Vec<DomNode>,
}

impl DomTree {
    /// 获取节点数量
    pub fn node_count(&self) -> usize {
        self.all_nodes.len()
    }

    /// 查找节点 (通过选择器)
    pub fn query_selector(&self, selector: &str) -> Vec<&DomNode> {
        // 简化实现 - 实际应该使用选择器引擎
        self.all_nodes
            .iter()
            .filter(|n| n.tag_name == selector.trim_start_matches("."))
            .collect()
    }
}

/// DOM 节点
#[derive(Debug, Clone)]
pub struct DomNode {
    /// 标签名
    pub tag_name: String,
    /// 属性
    pub attributes: HashMap<String, String>,
    /// 子节点
    pub children: Vec<DomNode>,
    /// 文本内容
    pub text_content: Option<String>,
    /// 计算样式 (从 CSS 继承)
    pub computed_styles: HashMap<String, String>,
}

/// JS 函数
#[derive(Debug, Clone)]
pub struct JsFunction {
    /// 函数名
    pub name: String,
    /// 参数
    pub params: Vec<String>,
    /// 函数体
    pub body: String,
    /// 是否是事件处理器
    pub is_event_handler: bool,
    /// 关联的 DOM 元素
    pub attached_elements: Vec<String>,
}

/// 渲染统计
#[derive(Debug, Clone, Default)]
pub struct RenderStats {
    /// 下载字节数
    pub bytes_downloaded: usize,
    /// CSS 文件数
    pub css_files: usize,
    /// JS 文件数
    pub js_files: usize,
    /// 图片数
    pub images: usize,
    /// 渲染时间 (ms)
    pub render_time_ms: u64,
}

impl StandardSandbox {
    /// 创建标准沙盒
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    /// 渲染网站 - 获取完整资源
    pub async fn render(&self, url: &str) -> Result<RenderedPage> {
        let start = std::time::Instant::now();

        // 1. 获取主 HTML
        let html = self.fetch_html(url).await?;

        // 2. 解析 HTML 提取资源链接
        let (css_urls, js_urls) = self.extract_resource_urls(&html, url);

        // 3. 下载所有 CSS
        let css_resources = self.fetch_all_css(&css_urls).await?;

        // 4. 下载所有 JS
        let js_resources = self.fetch_all_js(&js_urls).await?;

        // 5. 构建 DOM 树
        let dom_tree = self.build_dom_tree(&html)?;

        // 6. 计算渲染统计
        let stats = RenderStats {
            bytes_downloaded: html.len()
                + css_resources.iter().map(|c| c.content.len()).sum::<usize>()
                + js_resources.iter().map(|j| j.content.len()).sum::<usize>(),
            css_files: css_resources.len(),
            js_files: js_resources.len(),
            images: 0, // TODO
            render_time_ms: start.elapsed().as_millis() as u64,
        };

        Ok(RenderedPage {
            url: url.to_string(),
            html,
            css_resources,
            js_resources,
            dom_tree,
            stats,
        })
    }

    /// 获取 HTML（带User-Agent避免反爬虫）
    async fn fetch_html(&self, url: &str) -> Result<String> {
        let response = self
            .client
            .get(url)
            .header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0",
            )
            .header(
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .header("Accept-Language", "en-US,en;q=0.5")
            .send()
            .await
            .with_context(|| format!("Failed to fetch HTML from {}", url))?;

        let html = response
            .text()
            .await
            .context("Failed to read HTML response")?;

        Ok(html)
    }

    /// 从 HTML 提取资源 URL（解码HTML实体）
    fn extract_resource_urls(&self, html: &str, base_url: &str) -> (Vec<String>, Vec<String>) {
        let mut css_urls = Vec::new();
        let mut js_urls = Vec::new();

        // 简单的正则提取
        // 提取 <link rel="stylesheet" href="...">
        for cap in
            regex::Regex::new(r#"<link[^>]*rel=["']stylesheet["'][^>]*href=["']([^"']+)["']"#)
                .unwrap()
                .captures_iter(html)
        {
            let url = self.decode_html_entities(&cap[1]);
            let url = self.resolve_url(&url, base_url);
            css_urls.push(url);
        }

        // 提取 <script src="...">
        for cap in regex::Regex::new(r#"<script[^>]*src=["']([^"']+)["']"#)
            .unwrap()
            .captures_iter(html)
        {
            let url = self.decode_html_entities(&cap[1]);
            let url = self.resolve_url(&url, base_url);
            js_urls.push(url);
        }

        (css_urls, js_urls)
    }

    /// 解码HTML实体 (&amp; -> &)
    fn decode_html_entities(&self, text: &str) -> String {
        text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
    }

    /// 解析相对 URL
    fn resolve_url(&self, url: &str, base: &str) -> String {
        if url.starts_with("http://") || url.starts_with("https://") {
            url.to_string()
        } else if url.starts_with("//") {
            format!("https:{}", url)
        } else if url.starts_with('/') {
            // 绝对路径
            let base_url = url::Url::parse(base).unwrap();
            format!(
                "{}://{}{}",
                base_url.scheme(),
                base_url.host_str().unwrap(),
                url
            )
        } else {
            // 相对路径
            format!("{}/{}", base.trim_end_matches('/'), url)
        }
    }

    /// 获取所有 CSS
    async fn fetch_all_css(&self, urls: &[String]) -> Result<Vec<CssResource>> {
        let mut resources = Vec::new();

        for url in urls {
            match self.fetch_css(url).await {
                Ok(resource) => resources.push(resource),
                Err(e) => log::warn!("Failed to fetch CSS {}: {}", url, e),
            }
        }

        Ok(resources)
    }

    /// 获取单个 CSS（带User-Agent）
    async fn fetch_css(&self, url: &str) -> Result<CssResource> {
        let response = self
            .client
            .get(url)
            .header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0",
            )
            .header("Accept", "text/css,*/*;q=0.1")
            .send()
            .await
            .with_context(|| format!("Failed to fetch CSS from {}", url))?;

        let content = response
            .text()
            .await
            .context("Failed to read CSS response")?;

        // 解析 CSS 规则
        let rules = self.parse_css_rules(&content, url);

        Ok(CssResource {
            url: url.to_string(),
            content,
            rules,
        })
    }

    /// 解析 CSS 规则
    fn parse_css_rules(&self, css: &str, _source: &str) -> Vec<CssRule> {
        let rules = Vec::new();

        // 使用 cssparser 库解析
        // 简化实现 - 实际应该完整解析
        let mut input = cssparser::ParserInput::new(css);
        let _parser = cssparser::Parser::new(&mut input);

        // TODO: 完整 CSS 解析

        rules
    }

    /// 获取所有 JS
    async fn fetch_all_js(&self, urls: &[String]) -> Result<Vec<JsResource>> {
        let mut resources = Vec::new();

        for url in urls {
            match self.fetch_js(url).await {
                Ok(resource) => resources.push(resource),
                Err(e) => log::warn!("Failed to fetch JS {}: {}", url, e),
            }
        }

        Ok(resources)
    }

    /// 获取单个 JS（带User-Agent）
    async fn fetch_js(&self, url: &str) -> Result<JsResource> {
        let response = self
            .client
            .get(url)
            .header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0",
            )
            .header("Accept", "*/*")
            .send()
            .await
            .with_context(|| format!("Failed to fetch JS from {}", url))?;

        let content = response
            .text()
            .await
            .context("Failed to read JS response")?;

        // 检测是否混淆
        let is_obfuscated = self.detect_obfuscation(&content);

        // 提取函数
        let functions = self.extract_js_functions(&content);

        Ok(JsResource {
            url: url.to_string(),
            content,
            is_obfuscated,
            functions,
        })
    }

    /// 检测 JS 混淆
    fn detect_obfuscation(&self, js: &str) -> bool {
        let indicators = ["_0x", "eval(", "Function(", "atob(", "charCodeAt("];

        let count = indicators.iter().filter(|&i| js.contains(i)).count();
        count >= 3
    }

    /// 提取 JS 函数
    fn extract_js_functions(&self, js: &str) -> Vec<JsFunction> {
        let mut functions = Vec::new();

        // 简化实现 - 实际应该用 AST 解析
        // 提取 function name() { ... }
        let re = regex::Regex::new(r#"function\s+(\w+)\s*\(([^)]*)\)\s*\{"#).unwrap();
        for cap in re.captures_iter(js) {
            functions.push(JsFunction {
                name: cap[1].to_string(),
                params: cap[2].split(',').map(|s| s.trim().to_string()).collect(),
                body: String::new(),     // TODO: 提取函数体
                is_event_handler: false, // TODO: 检测
                attached_elements: Vec::new(),
            });
        }

        functions
    }

    /// 构建 DOM 树
    fn build_dom_tree(&self, _html: &str) -> Result<DomTree> {
        // 使用 html5ever 解析
        // 简化实现
        let root = DomNode {
            tag_name: "html".to_string(),
            attributes: HashMap::new(),
            children: Vec::new(),
            text_content: None,
            computed_styles: HashMap::new(),
        };

        Ok(DomTree {
            root,
            all_nodes: Vec::new(),
        })
    }
}

impl Default for StandardSandbox {
    fn default() -> Self {
        Self::new()
    }
}
