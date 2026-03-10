//! 视觉分析器
//! 协调各种分析任务

use crate::*;
use anyhow::Result;

/// 视觉分析器
pub struct VisualAnalyzer {
    config: VisualLearningConfig,
}

impl VisualAnalyzer {
    pub fn new(config: &VisualLearningConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// 分析图像
    pub fn analyze(&self, _image: &image::DynamicImage) -> Result<VisualAnalysis> {
        // 这里可以协调多个分析器的调用
        // 目前作为 VisualLearningEngine 的辅助
        unimplemented!("Use VisualLearningEngine for full analysis")
    }

    /// 比较两个视觉分析
    pub fn compare_analysis(&self, original: &VisualAnalysis, generated: &VisualAnalysis) -> ComparisonResult {
        let mut differences = Vec::new();
        let mut scores = HashMap::new();

        // 比较组件数量
        let component_score = self.compare_components(&original.components, &generated.components);
        scores.insert("components".to_string(), component_score);
        if component_score < 0.8 {
            differences.push(Difference {
                category: "components".to_string(),
                description: format!(
                    "组件数量不匹配: 原始{}个, 生成{}个",
                    original.components.len(),
                    generated.components.len()
                ),
                severity: Severity::Warning,
            });
        }

        // 比较颜色方案
        let color_score = self.compare_color_schemes(&original.color_scheme, &generated.color_scheme);
        scores.insert("colors".to_string(), color_score);
        if color_score < 0.8 {
            differences.push(Difference {
                category: "colors".to_string(),
                description: "颜色方案差异较大".to_string(),
                severity: Severity::Warning,
            });
        }

        // 比较布局
        let layout_score = self.compare_layouts(&original.layout, &generated.layout);
        scores.insert("layout".to_string(), layout_score);
        if layout_score < 0.7 {
            differences.push(Difference {
                category: "layout".to_string(),
                description: "布局结构差异较大".to_string(),
                severity: Severity::Error,
            });
        }

        // 计算总分
        let overall_score = scores.values().sum::<f64>() / scores.len() as f64;

        ComparisonResult {
            overall_score,
            scores,
            differences,
            passed: overall_score >= 0.8,
        }
    }

    /// 比较组件
    fn compare_components(&self, original: &[VisualComponent], generated: &[VisualComponent]) -> f64 {
        if original.is_empty() && generated.is_empty() {
            return 1.0;
        }
        if original.is_empty() || generated.is_empty() {
            return 0.0;
        }

        let mut matched = 0;
        for orig in original {
            let _weight = orig.confidence;

            // 查找匹配的生成组件
            let match_score = generated.iter()
                .map(|gen| self.calculate_component_similarity(orig, gen))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);

            if match_score > 0.7 {
                matched += 1;
            }
        }

        let coverage = matched as f64 / original.len() as f64;
        let precision = matched as f64 / generated.len().max(1) as f64;

        2.0 * (coverage * precision) / (coverage + precision + 0.001)
    }

    /// 计算组件相似度
    fn calculate_component_similarity(&self, a: &VisualComponent, b: &VisualComponent) -> f64 {
        let mut score = 0.0;

        // 类型匹配
        if a.component_type == b.component_type {
            score += 0.4;
        }

        // 位置匹配
        let pos_sim = self.calculate_position_similarity(&a.bounding_box, &b.bounding_box);
        score += pos_sim * 0.3;

        // 尺寸匹配
        let size_sim = self.calculate_size_similarity(&a.bounding_box, &b.bounding_box);
        score += size_sim * 0.2;

        // 样式匹配
        let style_sim = self.calculate_style_similarity(&a.visual_style, &b.visual_style);
        score += style_sim * 0.1;

        score
    }

    /// 计算位置相似度
    fn calculate_position_similarity(&self, a: &BoundingBox, b: &BoundingBox) -> f64 {
        let dx = (a.x as f64 - b.x as f64).abs();
        let dy = (a.y as f64 - b.y as f64).abs();
        
        let distance = (dx * dx + dy * dy).sqrt();
        let threshold = 50.0;

        if distance > threshold {
            0.0
        } else {
            1.0 - (distance / threshold)
        }
    }

    /// 计算尺寸相似度
    fn calculate_size_similarity(&self, a: &BoundingBox, b: &BoundingBox) -> f64 {
        let width_ratio = (a.width as f64 / b.width as f64)
            .min(b.width as f64 / a.width as f64);
        let height_ratio = (a.height as f64 / b.height as f64)
            .min(b.height as f64 / a.height as f64);

        width_ratio.min(height_ratio)
    }

    /// 计算样式相似度
    fn calculate_style_similarity(&self, a: &VisualStyle, b: &VisualStyle) -> f64 {
        let mut score = 0.0;
        let mut checks = 0;

        // 背景色
        if let (Some(ref c1), Some(ref c2)) = (&a.background_color, &b.background_color) {
            score += self.calculate_color_similarity(c1, c2);
            checks += 1;
        }

        // 圆角
        if a.border_radius == b.border_radius {
            score += 1.0;
        } else {
            let diff = (a.border_radius as i16 - b.border_radius as i16).abs();
            score += 1.0 - (diff as f64 / 20.0).min(1.0);
        }
        checks += 1;

        if checks > 0 {
            score / checks as f64
        } else {
            0.0
        }
    }

    /// 计算颜色相似度
    fn calculate_color_similarity(&self, a: &Color, b: &Color) -> f64 {
        let dr = (a.r as i16 - b.r as i16).abs() as f64;
        let dg = (a.g as i16 - b.g as i16).abs() as f64;
        let db = (a.b as i16 - b.b as i16).abs() as f64;

        let distance = (dr * dr + dg * dg + db * db).sqrt();
        let max_distance = (255.0f64 * 255.0 * 3.0).sqrt();

        1.0 - (distance / max_distance)
    }

    /// 比较颜色方案
    fn compare_color_schemes(&self, original: &ColorScheme, generated: &ColorScheme) -> f64 {
        let mut score = 0.0;
        let mut checks = 0;

        // 比较主色
        if let (Some(ref o), Some(ref g)) = (&original.primary, &generated.primary) {
            score += self.calculate_color_similarity(o, g);
            checks += 1;
        }

        // 比较背景色
        if let (Some(ref o), Some(ref g)) = (&original.background, &generated.background) {
            score += self.calculate_color_similarity(o, g);
            checks += 1;
        }

        // 比较文本色
        if let (Some(ref o), Some(ref g)) = (&original.text_primary, &generated.text_primary) {
            score += self.calculate_color_similarity(o, g);
            checks += 1;
        }

        if checks > 0 {
            score / checks as f64
        } else {
            0.0
        }
    }

    /// 比较布局
    fn compare_layouts(&self, original: &LayoutInfo, generated: &LayoutInfo) -> f64 {
        let mut score = 0.0;

        // 布局类型
        if original.layout_type == generated.layout_type {
            score += 0.4;
        }

        // 区域数量
        let section_ratio = (original.sections.len() as f64 / generated.sections.len().max(1) as f64)
            .min(generated.sections.len() as f64 / original.sections.len().max(1) as f64);
        score += section_ratio * 0.3;

        // 网格列数
        if original.grid_columns == generated.grid_columns {
            score += 0.3;
        }

        score
    }
}

use std::collections::HashMap;

/// 比较结果
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub overall_score: f64,
    pub scores: HashMap<String, f64>,
    pub differences: Vec<Difference>,
    pub passed: bool,
}

/// 差异
#[derive(Debug, Clone)]
pub struct Difference {
    pub category: String,
    pub description: String,
    pub severity: Severity,
}

/// 严重程度
#[derive(Debug, Clone)]
pub enum Severity {
    Info,
    Warning,
    Error,
}
