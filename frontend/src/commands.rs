use serde::{Deserialize, Serialize};

/// CLI命令
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    /// 处理URL: browerai process <url>
    Process {
        url: String,
        user_id: Option<String>,
    },
    /// 显示历史: browerai history
    History { user_id: Option<String> },
    /// 配置: browerai config <key> <value>
    Config { key: String, value: String },
    /// 清空缓存: browerai clear-cache
    ClearCache,
    /// 显示统计: browerai stats
    Stats { user_id: Option<String> },
}

/// 处理结果
#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessResult {
    pub success: bool,
    pub url: String,
    pub user_id: String,
    pub message: String,
    pub output_path: Option<String>,
    pub processing_time_ms: u64,
}

impl ProcessResult {
    pub fn success(url: String, user_id: String, output_path: String, time_ms: u64) -> Self {
        Self {
            success: true,
            url,
            user_id,
            message: "处理成功".to_string(),
            output_path: Some(output_path),
            processing_time_ms: time_ms,
        }
    }

    pub fn error(url: String, user_id: String, error: String, time_ms: u64) -> Self {
        Self {
            success: false,
            url,
            user_id,
            message: error,
            output_path: None,
            processing_time_ms: time_ms,
        }
    }
}
