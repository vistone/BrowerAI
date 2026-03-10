//! 自动化观察系统
//! 使用无头浏览器自动探索网站并收集行为数据

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod explorer;
pub mod observer;
pub mod strategy;
pub mod reporter;

pub use explorer::AutoExplorer;
pub use observer::BehaviorObserver;
pub use strategy::{ExplorationStrategy, PriorityStrategy};
pub use reporter::ExplorationReporter;

/// 观察记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub timestamp: DateTime<Utc>,
    pub event_type: String,
    pub target: ElementInfo,
    pub page_url: String,
    pub details: HashMap<String, serde_json::Value>,
    pub before_state: Option<PageState>,
    pub after_state: Option<PageState>,
}

/// 元素信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementInfo {
    pub tag: String,
    pub id: Option<String>,
    pub classes: Vec<String>,
    pub attributes: HashMap<String, String>,
    pub text_content: Option<String>,
    pub selector: String,
    pub bounding_box: Option<BoundingBox>,
    pub is_visible: bool,
    pub is_interactive: bool,
}

/// 边界框
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// 页面状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageState {
    pub url: String,
    pub title: String,
    pub dom_snapshot: String,
    pub visible_elements: Vec<ElementInfo>,
    pub console_logs: Vec<String>,
    pub network_requests: Vec<NetworkRequest>,
}

/// 网络请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRequest {
    pub url: String,
    pub method: String,
    pub status: Option<u16>,
    pub timestamp: DateTime<Utc>,
}

/// 探索报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationReport {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub target_url: String,
    pub pages_explored: Vec<PageExploration>,
    pub total_observations: usize,
    pub unique_behaviors: Vec<BehaviorPattern>,
    pub coverage: CoverageReport,
    pub errors: Vec<ExplorationError>,
}

/// 页面探索记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageExploration {
    pub url: String,
    pub title: String,
    pub visit_count: usize,
    pub interactions: Vec<InteractionRecord>,
    pub elements_found: Vec<ElementInfo>,
    pub explored_elements: Vec<String>,
}

/// 交互记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionRecord {
    pub timestamp: DateTime<Utc>,
    pub action: InteractionAction,
    pub target: ElementInfo,
    pub result: InteractionResult,
    pub duration_ms: u64,
}

/// 交互动作
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionAction {
    Click,
    Hover,
    Input { value: String },
    Scroll { direction: ScrollDirection, amount: u32 },
    KeyPress { key: String },
    Focus,
    Blur,
}

/// 滚动方向
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScrollDirection {
    Up,
    Down,
    Left,
    Right,
}

/// 交互结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionResult {
    pub success: bool,
    pub state_changed: bool,
    pub navigation_occurred: bool,
    pub new_url: Option<String>,
    pub errors: Vec<String>,
}

/// 行为模式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorPattern {
    pub pattern_id: String,
    pub pattern_type: PatternType,
    pub trigger: InteractionAction,
    pub typical_targets: Vec<String>,
    pub effects: Vec<Effect>,
    pub frequency: usize,
    pub confidence: f64,
}

/// 模式类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    ClickToNavigate,
    ClickToToggle,
    ClickToSubmit,
    HoverToReveal,
    InputWithDebounce,
    FormSubmission,
    InfiniteScroll,
    TabSwitch,
    ModalOpen,
    ModalClose,
    DropdownOpen,
    DropdownSelect,
    Custom(String),
}

/// 效果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Effect {
    pub effect_type: EffectType,
    pub target: Option<String>,
    pub details: HashMap<String, serde_json::Value>,
}

/// 效果类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectType {
    Navigation,
    VisibilityChange,
    ContentChange,
    StyleChange,
    Animation,
    NetworkRequest,
    StateUpdate,
}

/// 覆盖率报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub total_elements: usize,
    pub explored_elements: usize,
    pub coverage_percentage: f64,
    pub by_type: HashMap<String, TypeCoverage>,
    pub unexplored_elements: Vec<String>,
}

/// 类型覆盖率
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeCoverage {
    pub total: usize,
    pub explored: usize,
    pub percentage: f64,
}

/// 探索错误
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationError {
    pub timestamp: DateTime<Utc>,
    pub url: String,
    pub action: String,
    pub error_message: String,
    pub recoverable: bool,
}

/// 探索配置
#[derive(Debug, Clone)]
pub struct ExplorationConfig {
    pub max_pages: usize,
    pub max_time_seconds: u64,
    pub max_depth: usize,
    pub wait_after_action_ms: u64,
    pub wait_for_navigation_ms: u64,
    pub respect_robots_txt: bool,
    pub allowed_domains: Vec<String>,
    pub blocked_urls: Vec<regex::Regex>,
    pub viewport: ViewportConfig,
    pub user_agent: Option<String>,
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        Self {
            max_pages: 50,
            max_time_seconds: 300,
            max_depth: 3,
            wait_after_action_ms: 500,
            wait_for_navigation_ms: 5000,
            respect_robots_txt: true,
            allowed_domains: vec![],
            blocked_urls: vec![],
            viewport: ViewportConfig::default(),
            user_agent: None,
        }
    }
}

/// 视口配置
#[derive(Debug, Clone)]
pub struct ViewportConfig {
    pub width: u32,
    pub height: u32,
    pub device_scale_factor: f64,
}

impl Default for ViewportConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            device_scale_factor: 1.0,
        }
    }
}
