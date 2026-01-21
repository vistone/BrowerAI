use serde::{Deserialize, Serialize};

/// 用户会话
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSession {
    /// 用户ID
    pub user_id: String,
    /// 会话历史
    pub history: Vec<SessionEntry>,
    /// 用户偏好设置
    pub preferences: UserPreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEntry {
    pub url: String,
    pub timestamp: String,
    pub result_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// 偏好颜色主题
    pub preferred_colors: Vec<String>,
    /// 布局偏好
    pub layout_preference: String,
    /// 字体大小
    pub font_size_multiplier: f32,
    /// 对比度
    pub contrast_level: i32,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            preferred_colors: vec!["#007bff".to_string(), "#28a745".to_string()],
            layout_preference: "compact".to_string(),
            font_size_multiplier: 1.0,
            contrast_level: 50,
        }
    }
}

impl UserSession {
    pub fn new(user_id: String) -> Self {
        Self {
            user_id,
            history: Vec::new(),
            preferences: UserPreferences::default(),
        }
    }

    pub fn add_entry(&mut self, url: String, result_path: String) {
        use chrono::Local;
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        self.history.push(SessionEntry {
            url,
            timestamp,
            result_path,
        });
    }
}
