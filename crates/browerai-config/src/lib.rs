use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserConfig {
    pub headless: bool,
    pub viewport: ViewportConfig,
    pub timeout: u64,
    pub user_agent: String,
    pub storage: StorageConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewportConfig {
    pub width: u32,
    pub height: u32,
    pub device_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub cache_dir: PathBuf,
    pub max_cache_size: u64,
    pub cookie_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiConfig {
    pub enabled: bool,
    pub model_dir: PathBuf,
    pub default_model: String,
    pub onnx_enabled: bool,
    pub candle_enabled: bool,
    pub inference_threads: usize,
    pub memory_limit: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParserConfig {
    pub html: HtmlParserConfig,
    pub css: CssParserConfig,
    pub js: JsParserConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlParserConfig {
    pub strict_mode: bool,
    pub enable_html5: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CssParserConfig {
    pub enable_vendor_prefixes: bool,
    pub max_selectors: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsParserConfig {
    pub strict_mode: bool,
    pub max_tokens: usize,
}

/// JavaScript 反混淆配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeobfuscationConfig {
    /// 是否启用自动反混淆
    pub enabled: bool,
    /// 混淆检测阈值 (0.0 - 1.0)
    pub detection_threshold: f32,
    /// 是否使用AI反混淆 (需要加载PyTorch模型)
    pub use_ai: bool,
    /// AI模型路径 (相对于项目根目录)
    pub model_path: Option<PathBuf>,
    /// 词汇表路径
    pub vocab_path: Option<PathBuf>,
    /// 是否记录反混淆日志
    pub log_enabled: bool,
}

impl Default for DeobfuscationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            detection_threshold: 0.3,
            use_ai: false,
            model_path: None,
            vocab_path: None,
            log_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RendererConfig {
    pub enable_gpu: bool,
    pub max_layout_threads: usize,
    pub enable_predictive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub max_connections: u32,
    pub connection_timeout: u64,
    pub max_redirects: u32,
    pub user_agent: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub browser: BrowserConfig,
    pub ai: Option<AiConfig>,
    pub parser: ParserConfig,
    pub renderer: RendererConfig,
    pub network: NetworkConfig,
    /// JavaScript反混淆配置
    #[serde(default)]
    pub deobfuscation: DeobfuscationConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            browser: BrowserConfig {
                headless: true,
                viewport: ViewportConfig {
                    width: 1920,
                    height: 1080,
                    device_scale: 1.0,
                },
                timeout: 30000,
                user_agent: "BrowerAI/0.1".to_string(),
                storage: StorageConfig {
                    cache_dir: PathBuf::from(".browerai-cache"),
                    max_cache_size: 1024 * 1024 * 1024,
                    cookie_enabled: true,
                },
            },
            ai: None,
            parser: ParserConfig {
                html: HtmlParserConfig {
                    strict_mode: false,
                    enable_html5: true,
                },
                css: CssParserConfig {
                    enable_vendor_prefixes: true,
                    max_selectors: 10000,
                },
                js: JsParserConfig {
                    strict_mode: false,
                    max_tokens: 1000000,
                },
            },
            renderer: RendererConfig {
                enable_gpu: false,
                max_layout_threads: 4,
                enable_predictive: false,
            },
            network: NetworkConfig {
                max_connections: 10,
                connection_timeout: 30000,
                max_redirects: 5,
                user_agent: "BrowerAI/0.1".to_string(),
            },
            deobfuscation: DeobfuscationConfig::default(),
        }
    }
}

pub struct ConfigLoader {
    config_path: PathBuf,
}

impl ConfigLoader {
    pub fn new(config_path: impl AsRef<std::path::Path>) -> Self {
        Self {
            config_path: config_path.as_ref().to_path_buf(),
        }
    }

    pub fn load(&self) -> anyhow::Result<Config> {
        if !self.config_path.exists() {
            return Ok(Config::default());
        }

        let content = fs::read_to_string(&self.config_path)?;
        self.parse(&content)
    }

    pub fn from_env() -> anyhow::Result<Config> {
        let config = Config {
            browser: BrowserConfig {
                headless: std::env::var("BROWERAI_HEADLESS")
                    .map(|v| v != "false")
                    .unwrap_or(true),
                viewport: ViewportConfig {
                    width: std::env::var("BROWERAI_WIDTH")
                        .unwrap_or_else(|_| "1920".into())
                        .parse()?,
                    height: std::env::var("BROWERAI_HEIGHT")
                        .unwrap_or_else(|_| "1080".into())
                        .parse()?,
                    device_scale: 1.0,
                },
                timeout: std::env::var("BROWERAI_TIMEOUT")
                    .unwrap_or_else(|_| "30000".into())
                    .parse()?,
                user_agent: std::env::var("BROWERAI_USER_AGENT")
                    .unwrap_or_else(|_| "BrowerAI/0.1".to_string()),
                storage: StorageConfig {
                    cache_dir: std::env::var("BROWERAI_CACHE_DIR")
                        .map(Into::into)
                        .unwrap_or_else(|_| std::env::temp_dir().join("browerai")),
                    max_cache_size: 1024 * 1024 * 1024,
                    cookie_enabled: true,
                },
            },
            ai: None,
            parser: ParserConfig {
                html: HtmlParserConfig {
                    strict_mode: false,
                    enable_html5: true,
                },
                css: CssParserConfig {
                    enable_vendor_prefixes: true,
                    max_selectors: 10000,
                },
                js: JsParserConfig {
                    strict_mode: false,
                    max_tokens: 1000000,
                },
            },
            renderer: RendererConfig {
                enable_gpu: false,
                max_layout_threads: 4,
                enable_predictive: false,
            },
            network: NetworkConfig {
                max_connections: 10,
                connection_timeout: 30000,
                max_redirects: 5,
                user_agent: "BrowerAI/0.1".to_string(),
            },
            deobfuscation: DeobfuscationConfig {
                enabled: std::env::var("BROWERAI_DEOBFUSCATION_ENABLED")
                    .map(|v| v != "false")
                    .unwrap_or(true),
                detection_threshold: std::env::var("BROWERAI_DEOBFUSCATION_THRESHOLD")
                    .unwrap_or_else(|_| "0.3".into())
                    .parse()
                    .unwrap_or(0.3),
                use_ai: std::env::var("BROWERAI_DEOBFUSCATION_USE_AI")
                    .map(|v| v == "true")
                    .unwrap_or(false),
                model_path: std::env::var("BROWERAI_DEOBFUSCATION_MODEL_PATH")
                    .ok()
                    .map(Into::into),
                vocab_path: std::env::var("BROWERAI_DEOBFUSCATION_VOCAB_PATH")
                    .ok()
                    .map(Into::into),
                log_enabled: std::env::var("BROWERAI_DEOBFUSCATION_LOG")
                    .map(|v| v != "false")
                    .unwrap_or(true),
            },
        };
        Ok(config)
    }

    fn parse(&self, content: &str) -> anyhow::Result<Config> {
        let ext = self.config_path.extension().and_then(|e| e.to_str());
        if ext == Some("toml") {
            toml::from_str(content).map_err(anyhow::Error::msg)
        } else if ext == Some("json") {
            serde_json::from_str(content).map_err(anyhow::Error::msg)
        } else {
            anyhow::bail!("Unsupported config format: {:?}", self.config_path)
        }
    }

    pub fn save(&self, config: &Config) -> anyhow::Result<()> {
        let content: String =
            if self.config_path.extension().and_then(|e| e.to_str()) == Some("toml") {
                toml::to_string_pretty(config)?
            } else {
                serde_json::to_string_pretty(config)?
            };

        fs::write(&self.config_path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.browser.headless);
        assert_eq!(config.browser.viewport.width, 1920);
        assert!(config.parser.html.enable_html5);
    }

    #[test]
    fn test_config_from_env() {
        std::env::set_var("BROWERAI_WIDTH", "1280");
        std::env::set_var("BROWERAI_HEIGHT", "720");
        std::env::set_var("BROWERAI_HEADLESS", "false");

        let config = ConfigLoader::from_env().unwrap();
        assert!(!config.browser.headless);
        assert_eq!(config.browser.viewport.width, 1280);
        assert_eq!(config.browser.viewport.height, 720);

        std::env::remove_var("BROWERAI_WIDTH");
        std::env::remove_var("BROWERAI_HEIGHT");
        std::env::remove_var("BROWERAI_HEADLESS");
    }

    #[test]
    fn test_config_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.toml");

        let loader = ConfigLoader::new(&config_path);
        let mut config = Config::default();
        config.browser.viewport.width = 2560;

        loader.save(&config).unwrap();
        let loaded = loader.load().unwrap();

        assert_eq!(loaded.browser.viewport.width, 2560);
    }

    #[test]
    fn test_viewport_config() {
        let viewport = ViewportConfig {
            width: 1920,
            height: 1080,
            device_scale: 2.0,
        };
        assert!(viewport.device_scale > 1.0);
    }

    #[test]
    fn test_ai_config() {
        let ai_config = AiConfig {
            enabled: true,
            model_dir: PathBuf::from("/models"),
            default_model: "bert-base".to_string(),
            onnx_enabled: true,
            candle_enabled: false,
            inference_threads: 4,
            memory_limit: 4 * 1024 * 1024 * 1024,
        };
        assert!(ai_config.enabled);
        assert!(ai_config.onnx_enabled);
    }
}
