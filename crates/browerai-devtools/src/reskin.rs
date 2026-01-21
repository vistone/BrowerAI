use anyhow::Result;

/// Configuration for runtime reskin. Default is disabled to avoid any side-effects.
#[derive(Debug, Clone)]
pub struct ReskinConfig {
    /// Turn on/off the runtime reskin pipeline.
    pub enabled: bool,
    /// Only apply to specific host (e.g., "www.jd.com").
    pub target_host: &'static str,
    /// Theme definition for generated CSS.
    pub theme: ReskinTheme,
}

impl Default for ReskinConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target_host: "www.jd.com",
            theme: ReskinTheme::default(),
        }
    }
}

/// Theme tokens for generated CSS.
#[derive(Debug, Clone)]
pub struct ReskinTheme {
    pub primary: &'static str,
    pub secondary: &'static str,
    pub accent: &'static str,
    pub bg: &'static str,
    pub panel: &'static str,
    pub text: &'static str,
    pub muted: &'static str,
    pub border: &'static str,
}

impl Default for ReskinTheme {
    fn default() -> Self {
        Self {
            primary: "#4f46e5",
            secondary: "#ec4899",
            accent: "#06b6d4",
            bg: "#0b1225",
            panel: "#121a33",
            text: "#e5e7eb",
            muted: "#94a3b8",
            border: "#1f2a48",
        }
    }
}

/// Result of reskin generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReskinResult {
    /// Style tag string to inject (contains `<style>` ... `</style>`).
    pub style_tag: String,
}

/// Runtime reskin pipeline: learns minimal semantics, infers targets, generates overlay CSS.
///
/// It **never** rewrites href/action/src/JS 逻辑，只提供样式覆盖层。
pub struct ReskinPipeline {
    config: ReskinConfig,
}

impl ReskinPipeline {
    pub fn new(config: ReskinConfig) -> Self {
        Self { config }
    }

    /// Apply reskin for a given URL. If disabled or host 不匹配，返回 None。
    pub fn apply(&self, url: &str) -> Result<Option<ReskinResult>> {
        if !self.config.enabled {
            return Ok(None);
        }
        let host = Self::extract_host(url).unwrap_or_default();
        if host != self.config.target_host {
            return Ok(None);
        }

        let css = self.generate_css();
        let style_tag = format!("<style data-browerai-reskin>{}</style>", css);
        Ok(Some(ReskinResult { style_tag }))
    }

    fn extract_host(url: &str) -> Option<String> {
        url::Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(|s| s.to_string()))
    }

    /// Generate overlay CSS (角色识别使用通用选择器，不改动功能属性)。
    fn generate_css(&self) -> String {
        let t = &self.config.theme;
        format!(
            r#"
:root {{
  --primary: {primary};
  --secondary: {secondary};
  --accent: {accent};
  --bg: {bg};
  --panel: {panel};
  --text: {text};
  --muted: {muted};
  --border: {border};
}}
html, body {{ background: linear-gradient(120deg, var(--bg), #0e1533)) !important; color: var(--text) !important; }}
header, .header, .site-topbar, [class*="top"] {{ background: rgba(11,18,37,.85) !important; backdrop-filter: blur(12px) !important; border-bottom: 1px solid var(--border) !important; }}
a {{ color: var(--text) !important; }}
a:hover {{ color: var(--accent) !important; }}
input[type="text"], input[type="search"], input[type="password"], textarea {{
  background: rgba(18,26,51,.7) !important; color: var(--text) !important; border: 1px solid var(--border) !important; border-radius: 10px !important;
}}
button, [role="button"], .btn {{
  background: linear-gradient(90deg,var(--primary),var(--accent)) !important; color: #0b1225 !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important;
}}
nav, .nav, [class*="nav"] {{ background: rgba(18,26,51,.6) !important; border-top: 1px solid var(--border) !important; border-bottom: 1px solid var(--border) !important; }}
nav a, .nav a {{ color: var(--muted) !important; padding: .4rem .6rem !important; border-radius: 8px !important; }}
nav a:hover, .nav a:hover {{ color: var(--primary) !important; background: rgba(79,70,229,.12) !important; }}
.gl-item, [class*="item"], [class*="sku"], [class*="product"], [class*="search-result"] {{
  border-radius: 12px !important; overflow: hidden !important; border: 1px solid var(--border) !important; background: var(--panel) !important;
}}
.gl-item .p-name, [class*="name"], [class*="title"] {{ color: var(--text) !important; }}
.gl-item .p-price, [class*="price"] {{ font-weight: 900 !important; background: linear-gradient(90deg,var(--accent),var(--primary)) !important; -webkit-background-clip:text !important; background-clip:text !important; color: transparent !important; }}
.gl-item .p-commit, [class*="commit"], [class*="rating"] {{ color: var(--muted) !important; }}
footer, .footer, [class*="footer"] {{ border-top: 1px solid var(--border) !important; background: rgba(18,26,51,.7) !important; color: var(--muted) !important; }}
.login-wrap, [class*="login"], form {{ border-radius: 12px !important; background: rgba(18,26,51,.6) !important; border: 1px solid var(--border) !important; }}
"#,
            primary = t.primary,
            secondary = t.secondary,
            accent = t.accent,
            bg = t.bg,
            panel = t.panel,
            text = t.text,
            muted = t.muted,
            border = t.border,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_returns_none() {
        let pipeline = ReskinPipeline::new(ReskinConfig::default());
        let res = pipeline.apply("https://www.jd.com/").unwrap();
        assert!(res.is_none());
    }

    #[test]
    fn host_mismatch_returns_none() {
        let mut cfg = ReskinConfig::default();
        cfg.enabled = true;
        cfg.target_host = "example.com";
        let pipeline = ReskinPipeline::new(cfg);
        let res = pipeline.apply("https://www.jd.com/").unwrap();
        assert!(res.is_none());
    }

    #[test]
    fn match_returns_style() {
        let mut cfg = ReskinConfig::default();
        cfg.enabled = true;
        let pipeline = ReskinPipeline::new(cfg);
        let res = pipeline.apply("https://www.jd.com/").unwrap();
        assert!(res.is_some());
        let style = res.unwrap();
        assert!(style.style_tag.contains("data-browerai-reskin"));
        assert!(style.style_tag.contains("--primary"));
        assert!(style.style_tag.contains("background-clip:text"));
    }
}
