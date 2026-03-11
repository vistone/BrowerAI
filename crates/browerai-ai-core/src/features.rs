//! Feature Extraction - 特征提取
//!
//! 从代码、DOM、CSS等提取AI可用的特征向量，包括：
//! - 代码特征（复杂度、结构）
//! - DOM特征（树深度、节点数）
//! - CSS特征（选择器复杂度、属性数）
//! - 渲染特征（布局复杂度、绘制复杂度）

use browerai_core::Result;
use std::collections::HashMap;

/// 特征提取器
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// 配置
    #[allow(dead_code)]
    config: FeatureExtractorConfig,
    /// 特征缓存
    cache: HashMap<String, FeatureVector>,
}

impl FeatureExtractor {
    /// 创建新的特征提取器
    pub fn new() -> Self {
        Self {
            config: FeatureExtractorConfig::default(),
            cache: HashMap::new(),
        }
    }

    /// 使用配置创建特征提取器
    pub fn with_config(config: FeatureExtractorConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
        }
    }

    /// 提取特征
    pub fn extract(&self, input: &str, feature_type: FeatureType) -> Result<FeatureVector> {
        match feature_type {
            FeatureType::Code => self.extract_code_features(input),
            FeatureType::Dom => self.extract_dom_features(input),
            FeatureType::Css => self.extract_css_features(input),
            FeatureType::Render => self.extract_render_features(input),
            FeatureType::Mixed => self.extract_mixed_features(input),
        }
    }

    /// 提取代码特征
    fn extract_code_features(&self, code: &str) -> Result<FeatureVector> {
        let mut features = HashMap::new();

        // 基本统计特征
        features.insert("length".to_string(), code.len() as f32);
        features.insert("line_count".to_string(), code.lines().count() as f32);

        // 关键字频率
        let keywords = [
            "function", "var", "let", "const", "if", "for", "while", "return",
        ];
        for keyword in &keywords {
            let count = code.matches(keyword).count() as f32;
            features.insert(format!("keyword_{}", keyword), count);
        }

        // 复杂度指标
        let bracket_depth = self.calculate_max_bracket_depth(code);
        features.insert("max_bracket_depth".to_string(), bracket_depth as f32);

        Ok(FeatureVector {
            feature_type: FeatureType::Code,
            features,
            normalized: false,
        })
    }

    /// 提取DOM特征
    fn extract_dom_features(&self, html: &str) -> Result<FeatureVector> {
        let mut features = HashMap::new();

        // 基本统计
        features.insert("length".to_string(), html.len() as f32);

        // 标签统计
        let tag_open_count = html.matches('<').count() as f32;
        let tag_close_count = html.matches('>').count() as f32;
        features.insert("tag_open_count".to_string(), tag_open_count);
        features.insert("tag_close_count".to_string(), tag_close_count);

        // 常见标签
        let common_tags = ["div", "span", "p", "a", "img", "script", "style"];
        for tag in &common_tags {
            let count = html.matches(&format!("<{} ", tag)).count() as f32;
            features.insert(format!("tag_{}", tag), count);
        }

        Ok(FeatureVector {
            feature_type: FeatureType::Dom,
            features,
            normalized: false,
        })
    }

    /// 提取CSS特征
    fn extract_css_features(&self, css: &str) -> Result<FeatureVector> {
        let mut features = HashMap::new();

        // 基本统计
        features.insert("length".to_string(), css.len() as f32);
        features.insert("rule_count".to_string(), css.matches('{').count() as f32);

        // 选择器复杂度
        let class_selectors = css.matches('.').count() as f32;
        let id_selectors = css.matches('#').count() as f32;
        features.insert("class_selectors".to_string(), class_selectors);
        features.insert("id_selectors".to_string(), id_selectors);

        // 属性统计
        let properties = [
            "color",
            "background",
            "margin",
            "padding",
            "display",
            "position",
        ];
        for prop in &properties {
            let count = css.matches(prop).count() as f32;
            features.insert(format!("prop_{}", prop), count);
        }

        Ok(FeatureVector {
            feature_type: FeatureType::Css,
            features,
            normalized: false,
        })
    }

    /// 提取渲染特征
    fn extract_render_features(&self, _input: &str) -> Result<FeatureVector> {
        // 简化实现
        let features = HashMap::new();

        Ok(FeatureVector {
            feature_type: FeatureType::Render,
            features,
            normalized: false,
        })
    }

    /// 提取混合特征
    fn extract_mixed_features(&self, input: &str) -> Result<FeatureVector> {
        // 提取所有类型的特征并合并
        let code_features = self.extract_code_features(input)?;
        let dom_features = self.extract_dom_features(input)?;
        let css_features = self.extract_css_features(input)?;

        let mut merged = HashMap::new();
        merged.extend(code_features.features);
        merged.extend(dom_features.features);
        merged.extend(css_features.features);

        Ok(FeatureVector {
            feature_type: FeatureType::Mixed,
            features: merged,
            normalized: false,
        })
    }

    /// 计算最大括号深度
    fn calculate_max_bracket_depth(&self, code: &str) -> usize {
        let mut max_depth = 0;
        let mut current_depth: usize = 0;

        for ch in code.chars() {
            match ch {
                '{' | '(' | '[' => {
                    current_depth += 1;
                    max_depth = max_depth.max(current_depth);
                }
                '}' | ')' | ']' => {
                    current_depth = current_depth.saturating_sub(1);
                }
                _ => {}
            }
        }

        max_depth
    }

    /// 获取支持的特征类型
    pub fn supported_features(&self) -> Vec<FeatureType> {
        vec![
            FeatureType::Code,
            FeatureType::Dom,
            FeatureType::Css,
            FeatureType::Render,
            FeatureType::Mixed,
        ]
    }

    /// 清空缓存
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// 特征向量
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// 特征类型
    pub feature_type: FeatureType,
    /// 特征值
    pub features: HashMap<String, f32>,
    /// 是否已归一化
    pub normalized: bool,
}

impl FeatureVector {
    /// 创建新的特征向量
    pub fn new(feature_type: FeatureType) -> Self {
        Self {
            feature_type,
            features: HashMap::new(),
            normalized: false,
        }
    }

    /// 添加特征
    pub fn add(&mut self, name: impl Into<String>, value: f32) {
        self.features.insert(name.into(), value);
    }

    /// 获取特征值
    pub fn get(&self, name: &str) -> Option<f32> {
        self.features.get(name).copied()
    }

    /// 特征数量
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// 归一化特征（Min-Max归一化）
    pub fn normalize(&mut self) {
        if self.normalized || self.features.is_empty() {
            return;
        }

        let values: Vec<f32> = self.features.values().copied().collect();
        let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        if max - min > 1e-6 {
            for value in self.features.values_mut() {
                *value = (*value - min) / (max - min);
            }
        }

        self.normalized = true;
    }

    /// 转换为向量（按key排序）
    pub fn to_vec(&self) -> Vec<f32> {
        let mut pairs: Vec<_> = self.features.iter().collect();
        pairs.sort_by(|a, b| a.0.cmp(b.0));
        pairs.into_iter().map(|(_, v)| *v).collect()
    }

    /// 计算与另一个特征向量的欧氏距离
    pub fn euclidean_distance(&self, other: &FeatureVector) -> f32 {
        let keys: std::collections::HashSet<_> =
            self.features.keys().chain(other.features.keys()).collect();

        let mut sum_sq_diff = 0.0;
        for key in keys {
            let v1 = self.features.get(key).copied().unwrap_or(0.0);
            let v2 = other.features.get(key).copied().unwrap_or(0.0);
            sum_sq_diff += (v1 - v2).powi(2);
        }

        sum_sq_diff.sqrt()
    }
}

/// 特征类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureType {
    /// 代码特征
    Code,
    /// DOM特征
    Dom,
    /// CSS特征
    Css,
    /// 渲染特征
    Render,
    /// 混合特征
    Mixed,
}

/// 特征提取器配置
#[derive(Debug, Clone)]
pub struct FeatureExtractorConfig {
    /// 启用缓存
    pub enable_cache: bool,
    /// 最大缓存大小
    pub max_cache_size: usize,
    /// 自动归一化
    pub auto_normalize: bool,
}

impl Default for FeatureExtractorConfig {
    fn default() -> Self {
        Self {
            enable_cache: true,
            max_cache_size: 1000,
            auto_normalize: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extractor_creation() {
        let extractor = FeatureExtractor::new();
        let features = extractor.supported_features();
        assert!(!features.is_empty());
    }

    #[test]
    fn test_extract_code_features() {
        let extractor = FeatureExtractor::new();
        let code = "function test() { return 1 + 2; }";

        let features = extractor.extract(code, FeatureType::Code).unwrap();

        assert!(!features.is_empty());
        assert!(features.get("length").is_some());
        assert!(features.get("keyword_function").is_some());
    }

    #[test]
    fn test_extract_dom_features() {
        let extractor = FeatureExtractor::new();
        let html = "<div><span>Test</span></div>";

        let features = extractor.extract(html, FeatureType::Dom).unwrap();

        assert!(!features.is_empty());
        assert!(features.get("tag_open_count").is_some());
    }

    #[test]
    fn test_feature_vector_operations() {
        let mut vector = FeatureVector::new(FeatureType::Code);
        vector.add("feature1", 1.0);
        vector.add("feature2", 2.0);

        assert_eq!(vector.len(), 2);
        assert_eq!(vector.get("feature1"), Some(1.0));

        // 测试归一化
        vector.normalize();
        assert!(vector.normalized);
    }

    #[test]
    fn test_euclidean_distance() {
        let mut v1 = FeatureVector::new(FeatureType::Code);
        v1.add("a", 1.0);
        v1.add("b", 2.0);

        let mut v2 = FeatureVector::new(FeatureType::Code);
        v2.add("a", 4.0);
        v2.add("b", 6.0);

        let distance = v1.euclidean_distance(&v2);
        assert!(distance > 0.0);
    }
}
