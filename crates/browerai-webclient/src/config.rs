use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// WebClient 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    /// 用户ID
    pub user_id: String,
    /// 输出目录
    pub output_dir: PathBuf,
    /// 缓存目录
    pub cache_dir: PathBuf,
    /// 是否启用缓存
    pub enable_cache: bool,
    /// 调试模式
    pub debug_mode: bool,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            user_id: "default_user".to_string(),
            output_dir: PathBuf::from("./output"),
            cache_dir: PathBuf::from("./cache"),
            enable_cache: true,
            debug_mode: false,
        }
    }
}

impl ClientConfig {
    pub fn with_user_id(mut self, user_id: String) -> Self {
        self.user_id = user_id;
        self
    }

    pub fn with_output_dir(mut self, dir: PathBuf) -> Self {
        self.output_dir = dir;
        self
    }

    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug_mode = debug;
        self
    }
}
