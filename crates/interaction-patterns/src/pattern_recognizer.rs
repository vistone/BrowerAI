//! 模式识别器
//! 从观察数据中识别交互模式

use crate::*;
use auto_observer::{Observation, InteractionRecord};

/// 模式识别器
pub struct PatternRecognizer {
    library: InteractionPatternLibrary,
}

impl PatternRecognizer {
    pub fn new() -> Self {
        Self {
            library: InteractionPatternLibrary::new(),
        }
    }

    /// 从观察记录中识别模式
    pub fn recognize_from_observations(&self, observations: &[Observation]) -> Vec<InteractionPattern> {
        self.library.recognize_patterns(observations)
    }

    /// 从交互记录中识别模式
    pub fn recognize_from_interactions(&self, interactions: &[InteractionRecord]) -> Vec<InteractionPattern> {
        // 将交互记录转换为观察记录
        let observations: Vec<Observation> = interactions.iter()
            .map(|i| self.interaction_to_observation(i))
            .collect();

        self.recognize_from_observations(&observations)
    }

    fn interaction_to_observation(&self, interaction: &InteractionRecord) -> Observation {
        Observation {
            timestamp: interaction.timestamp,
            event_type: format!("{:?}", interaction.action),
            target: interaction.target.clone(),
            page_url: String::new(), // 需要从上下文获取
            details: HashMap::new(),
            before_state: None,
            after_state: None,
        }
    }

    /// 分析模式特征
    pub fn analyze_pattern_features(&self, observations: &[Observation]) -> PatternFeatures {
        let mut features = PatternFeatures::default();

        for obs in observations {
            match obs.event_type.as_str() {
                "mousedown" | "touchstart" => features.has_drag_start = true,
                "mousemove" | "touchmove" => {
                    if features.has_drag_start {
                        features.has_drag_move = true;
                    }
                }
                "mouseup" | "touchend" => {
                    if features.has_drag_move {
                        features.has_drag_end = true;
                    }
                }
                "scroll" => {
                    features.scroll_count += 1;
                    features.scroll_positions.push(obs.details.get("scrollY")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as f64);
                }
                "mutation" => {
                    features.mutation_count += 1;
                    if let Some(added) = obs.details.get("addedCount").and_then(|v| v.as_u64()) {
                        features.items_added += added;
                    }
                }
                _ => {}
            }
        }

        // 分析滚动模式
        if features.scroll_count > 5 {
            features.scroll_pattern = self.analyze_scroll_pattern(&features.scroll_positions);
        }

        features
    }

    fn analyze_scroll_pattern(&self, positions: &[f64]) -> ScrollPattern {
        if positions.len() < 3 {
            return ScrollPattern::None;
        }

        // 计算滚动方向变化
        let mut direction_changes = 0;
        let mut last_direction = 0i8; // -1: up, 0: none, 1: down

        for i in 1..positions.len() {
            let diff = positions[i] - positions[i - 1];
            let direction = if diff > 0.0 { 1 } else if diff < 0.0 { -1 } else { 0 };

            if direction != 0 && direction != last_direction && last_direction != 0 {
                direction_changes += 1;
            }

            if direction != 0 {
                last_direction = direction;
            }
        }

        // 根据方向变化判断模式
        if direction_changes == 0 && last_direction == 1 {
            ScrollPattern::ContinuousDown
        } else if direction_changes == 0 && last_direction == -1 {
            ScrollPattern::ContinuousUp
        } else if direction_changes > positions.len() as u32 / 3 {
            ScrollPattern::Erratic
        } else {
            ScrollPattern::Mixed
        }
    }
}

impl Default for PatternRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

/// 模式特征
#[derive(Debug, Default)]
pub struct PatternFeatures {
    pub has_drag_start: bool,
    pub has_drag_move: bool,
    pub has_drag_end: bool,
    pub scroll_count: u32,
    pub scroll_positions: Vec<f64>,
    pub scroll_pattern: ScrollPattern,
    pub mutation_count: u32,
    pub items_added: u64,
}

/// 滚动模式
#[derive(Debug, Default)]
pub enum ScrollPattern {
    #[default]
    None,
    ContinuousUp,
    ContinuousDown,
    Mixed,
    Erratic,
}
