//! 复杂交互模式库
//! 识别和实现复杂的Web交互模式

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod drag_drop;
pub mod infinite_scroll;
pub mod virtual_list;
pub mod tree_view;
pub mod rich_editor;
pub mod data_table;
pub mod pattern_recognizer;
pub mod code_generator;

pub use drag_drop::DragDropPattern;
pub use infinite_scroll::InfiniteScrollPattern;
pub use virtual_list::VirtualListPattern;
pub use tree_view::TreeViewPattern;
pub use rich_editor::RichEditorPattern;
pub use data_table::DataTablePattern;
pub use pattern_recognizer::PatternRecognizer;
pub use code_generator::PatternCodeGenerator;

/// 复杂交互模式类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ComplexPatternType {
    DragAndDrop,
    InfiniteScroll,
    VirtualList,
    TreeView,
    RichEditor,
    DataTable,
    Carousel,
    Tabs,
    Accordion,
    ContextMenu,
    Tooltip,
    Popover,
    Skeleton,
    LazyImage,
    Masonry,
    SplitPane,
    Resizable,
    Sortable,
    Swipeable,
    PinchZoom,
}

/// 交互模式定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPattern {
    pub pattern_type: ComplexPatternType,
    pub name: String,
    pub description: String,
    pub triggers: Vec<PatternTrigger>,
    pub behaviors: Vec<PatternBehavior>,
    pub state_machine: PatternStateMachine,
    pub confidence: f64,
}

/// 模式触发器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTrigger {
    pub trigger_type: TriggerType,
    pub selector: String,
    pub conditions: Vec<TriggerCondition>,
}

/// 触发类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerType {
    MouseDown,
    MouseMove,
    MouseUp,
    MouseEnter,
    MouseLeave,
    Click,
    DoubleClick,
    RightClick,
    DragStart,
    DragEnd,
    Drop,
    Scroll,
    KeyDown,
    KeyUp,
    TouchStart,
    TouchMove,
    TouchEnd,
    Resize,
    Intersection,
    Custom(String),
}

/// 触发条件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerCondition {
    pub property: String,
    pub operator: ConditionOperator,
    pub value: serde_json::Value,
}

/// 条件操作符
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    Contains,
    StartsWith,
    EndsWith,
    Matches,
}

/// 模式行为
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternBehavior {
    pub behavior_type: BehaviorType,
    pub target: String,
    pub animation: Option<AnimationConfig>,
    pub callback: Option<String>,
}

/// 行为类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BehaviorType {
    Move,
    Resize,
    Show,
    Hide,
    AddClass,
    RemoveClass,
    ToggleClass,
    SetStyle,
    SetAttribute,
    RemoveAttribute,
    InsertElement,
    RemoveElement,
    LoadData,
    SubmitForm,
    Navigate,
    Focus,
    Blur,
    Select,
    Copy,
    Paste,
}

/// 动画配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationConfig {
    pub duration_ms: u32,
    pub easing: String,
    pub properties: Vec<String>,
}

/// 模式状态机
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStateMachine {
    pub initial_state: String,
    pub states: Vec<PatternState>,
    pub transitions: Vec<StateTransition>,
}

/// 模式状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternState {
    pub name: String,
    pub description: String,
    pub entry_actions: Vec<PatternBehavior>,
    pub exit_actions: Vec<PatternBehavior>,
}

/// 状态转换
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub from_state: String,
    pub to_state: String,
    pub trigger: String,
    pub guard: Option<String>,
    pub actions: Vec<PatternBehavior>,
}

/// 生成的代码
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCode {
    pub pattern_type: ComplexPatternType,
    pub language: CodeLanguage,
    pub component_name: String,
    pub code: String,
    pub css: Option<String>,
    pub tests: Option<String>,
    pub documentation: String,
}

/// 代码语言
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodeLanguage {
    TypeScript,
    JavaScript,
    React,
    Vue,
    Svelte,
    Rust,
}

/// 交互模式库
pub struct InteractionPatternLibrary {
    patterns: HashMap<ComplexPatternType, Box<dyn PatternImplementation>>,
}

/// 模式实现 trait
pub trait PatternImplementation: Send + Sync {
    fn pattern_type(&self) -> ComplexPatternType;
    fn recognize(&self, observations: &[auto_observer::Observation]) -> Option<InteractionPattern>;
    fn generate_code(&self, pattern: &InteractionPattern, language: CodeLanguage) -> Result<GeneratedCode>;
    fn get_template(&self) -> &str;
}

impl InteractionPatternLibrary {
    pub fn new() -> Self {
        let mut patterns: HashMap<ComplexPatternType, Box<dyn PatternImplementation>> = HashMap::new();
        
        // 注册所有模式实现
        patterns.insert(ComplexPatternType::DragAndDrop, Box::new(DragDropPattern::new()));
        patterns.insert(ComplexPatternType::InfiniteScroll, Box::new(InfiniteScrollPattern::new()));
        patterns.insert(ComplexPatternType::VirtualList, Box::new(VirtualListPattern::new()));
        patterns.insert(ComplexPatternType::TreeView, Box::new(TreeViewPattern::new()));
        patterns.insert(ComplexPatternType::RichEditor, Box::new(RichEditorPattern::new()));
        patterns.insert(ComplexPatternType::DataTable, Box::new(DataTablePattern::new()));
        
        Self { patterns }
    }

    /// 识别所有模式
    pub fn recognize_patterns(&self, observations: &[auto_observer::Observation]) -> Vec<InteractionPattern> {
        let mut recognized = Vec::new();

        for (_, implementation) in &self.patterns {
            if let Some(pattern) = implementation.recognize(observations) {
                if pattern.confidence >= 0.7 {
                    recognized.push(pattern);
                }
            }
        }

        recognized
    }

    /// 生成模式代码
    pub fn generate_pattern_code(
        &self,
        pattern_type: ComplexPatternType,
        pattern: &InteractionPattern,
        language: CodeLanguage,
    ) -> Result<GeneratedCode> {
        if let Some(implementation) = self.patterns.get(&pattern_type) {
            implementation.generate_code(pattern, language)
        } else {
            anyhow::bail!("Pattern type {:?} not found", pattern_type)
        }
    }

    /// 获取所有支持的类型
    pub fn supported_types(&self) -> Vec<ComplexPatternType> {
        self.patterns.keys().cloned().collect()
    }
}

impl Default for InteractionPatternLibrary {
    fn default() -> Self {
        Self::new()
    }
}
