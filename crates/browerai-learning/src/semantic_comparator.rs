use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// 语义对比结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticComparisonResult {
    /// 函数级别对比
    pub function_similarity: FunctionSimilarity,
    /// DOM 结构相似度 (0-1)
    pub dom_structure_similarity: f64,
    /// 事件处理相似度 (0-1)
    pub event_handling_similarity: f64,
    /// 样式相似度 (0-1)
    pub style_similarity: f64,
    /// 综合相似度 (0-1)
    pub overall_similarity: f64,
    /// 缺失的功能点
    pub missing_features: Vec<String>,
    /// 额外的功能点
    pub extra_features: Vec<String>,
}

impl SemanticComparisonResult {
    pub fn average_function_score(&self) -> f64 {
        self.function_similarity.average_score()
    }
}

/// 函数相似度
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionSimilarity {
    /// 每个关键函数的相似度评分
    pub function_scores: HashMap<String, f64>,
    /// 覆盖的函数 (生成代码包含的关键函数)
    pub covered_functions: Vec<String>,
    /// 遗漏的函数 (原始代码有但生成代码缺失)
    pub missing_functions: Vec<String>,
}

impl FunctionSimilarity {
    pub fn average_score(&self) -> f64 {
        if self.function_scores.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.function_scores.values().sum();
        (sum / self.function_scores.len() as f64).clamp(0.0, 1.0)
    }
}

/// 语义对比器
pub struct SemanticComparator;

impl SemanticComparator {
    /// 综合对比
    #[allow(clippy::too_many_arguments)]
    pub fn compare_all(
        original_html: &str,
        original_css: &str,
        original_js: &str,
        generated_html: &str,
        generated_css: &str,
        generated_js: &str,
        key_functions: &[String],
    ) -> Result<SemanticComparisonResult> {
        let dom_structure_similarity = Self::compare_dom_structure(original_html, generated_html)?;
        let event_handling_similarity =
            Self::compare_event_handlers(original_html, generated_html, original_js, generated_js)?;
        let function_similarity =
            Self::compare_js_functions(original_js, generated_js, key_functions)?;
        let style_similarity = Self::compare_css_rules(original_css, generated_css)?;

        let avg_fn = function_similarity.average_score();
        let overall_similarity = (dom_structure_similarity * 0.35
            + avg_fn * 0.35
            + style_similarity * 0.2
            + event_handling_similarity * 0.1)
            .clamp(0.0, 1.0);

        let missing_features = function_similarity.missing_functions.clone();
        let extra_features = Self::extra_functions(original_js, generated_js);

        Ok(SemanticComparisonResult {
            function_similarity,
            dom_structure_similarity,
            event_handling_similarity,
            style_similarity,
            overall_similarity,
            missing_features,
            extra_features,
        })
    }

    /// 对比 DOM 结构（基于标签集合的 Jaccard 相似度）
    pub fn compare_dom_structure(original_html: &str, generated_html: &str) -> Result<f64> {
        let original_tags = Self::extract_tags(original_html);
        let generated_tags = Self::extract_tags(generated_html);
        Ok(Self::jaccard(&original_tags, &generated_tags))
    }

    /// 对比事件处理器（内联 + addEventListener）
    pub fn compare_event_handlers(
        original_html: &str,
        generated_html: &str,
        original_js: &str,
        generated_js: &str,
    ) -> Result<f64> {
        let original_events = Self::extract_events(original_html, original_js);
        let generated_events = Self::extract_events(generated_html, generated_js);
        Ok(Self::jaccard(&original_events, &generated_events))
    }

    /// 对比 JavaScript 函数集合
    pub fn compare_js_functions(
        original_js: &str,
        generated_js: &str,
        key_functions: &[String],
    ) -> Result<FunctionSimilarity> {
        let original_funcs = Self::extract_functions(original_js);
        let generated_funcs = Self::extract_functions(generated_js);

        let mut function_scores = HashMap::new();
        let mut covered_functions = vec![];
        let mut missing_functions = vec![];

        let unique_keys: HashSet<String> = key_functions.iter().cloned().collect();

        for key in unique_keys {
            let in_original = original_funcs.contains(&key);
            let in_generated = generated_funcs.contains(&key);
            let score = match (in_original, in_generated) {
                (true, true) => 1.0,
                (true, false) => 0.0,
                (false, true) => 0.7, // 生成了未标记的函数，部分匹配
                (false, false) => 0.0,
            };

            function_scores.insert(key.clone(), score);
            if in_generated {
                covered_functions.push(key.clone());
            }
            if in_original && !in_generated {
                missing_functions.push(key);
            }
        }

        // 额外：对整体函数集合的重叠进行一次加权
        let overlap_bonus = Self::jaccard(&original_funcs, &generated_funcs);
        if !function_scores.is_empty() {
            for value in function_scores.values_mut() {
                *value = ((*value) * 0.7 + overlap_bonus * 0.3).clamp(0.0, 1.0);
            }
        }

        Ok(FunctionSimilarity {
            function_scores,
            covered_functions,
            missing_functions,
        })
    }

    /// 对比 CSS 规则（选择器 + 属性 Jaccard 相似度）
    pub fn compare_css_rules(original_css: &str, generated_css: &str) -> Result<f64> {
        let original_selectors = Self::extract_selectors(original_css);
        let generated_selectors = Self::extract_selectors(generated_css);
        let original_props = Self::extract_properties(original_css);
        let generated_props = Self::extract_properties(generated_css);

        let selector_sim = Self::jaccard(&original_selectors, &generated_selectors);
        let property_sim = Self::jaccard(&original_props, &generated_props);

        Ok((selector_sim * 0.6 + property_sim * 0.4).clamp(0.0, 1.0))
    }

    fn extra_functions(original_js: &str, generated_js: &str) -> Vec<String> {
        let original_funcs = Self::extract_functions(original_js);
        let generated_funcs = Self::extract_functions(generated_js);

        generated_funcs
            .difference(&original_funcs)
            .cloned()
            .collect()
    }

    fn extract_tags(html: &str) -> HashSet<String> {
        let re = regex::Regex::new(r"<\s*([a-zA-Z0-9]+)").unwrap();
        re.captures_iter(html)
            .filter_map(|cap| cap.get(1))
            .map(|m| m.as_str().to_lowercase())
            .collect()
    }

    fn extract_events(html: &str, js: &str) -> HashSet<String> {
        let mut events = HashSet::new();
        let inline_re = regex::Regex::new(r"on([a-zA-Z]+)\s*=").unwrap();
        for cap in inline_re.captures_iter(html) {
            if let Some(ev) = cap.get(1) {
                events.insert(ev.as_str().to_lowercase());
            }
        }

        let listener_re =
            regex::Regex::new(r#"addEventListener\s*\(\s*['\"]([a-zA-Z]+)['\"]"#).unwrap();
        for cap in listener_re.captures_iter(js) {
            if let Some(ev) = cap.get(1) {
                events.insert(ev.as_str().to_lowercase());
            }
        }

        events
    }

    fn extract_functions(js: &str) -> HashSet<String> {
        let mut funcs = HashSet::new();

        let fn_re = regex::Regex::new(r"(?:async\s+)?function\s+([A-Za-z0-9_]+)").unwrap();
        for cap in fn_re.captures_iter(js) {
            if let Some(name) = cap.get(1) {
                funcs.insert(name.as_str().to_string());
            }
        }

        let arrow_re = regex::Regex::new(
            r"(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?\(.*?\)\s*=>",
        )
        .unwrap();
        for cap in arrow_re.captures_iter(js) {
            if let Some(name) = cap.get(1) {
                funcs.insert(name.as_str().to_string());
            }
        }

        funcs
    }

    fn extract_selectors(css: &str) -> HashSet<String> {
        let rule_re = regex::Regex::new(r"([^{}]+)\{[^}]*\}").unwrap();
        rule_re
            .captures_iter(css)
            .filter_map(|cap| cap.get(1))
            .map(|m| m.as_str().trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn extract_properties(css: &str) -> HashSet<String> {
        let mut props = HashSet::new();
        let rule_re = regex::Regex::new(r"\{([^}]*)\}").unwrap();
        for cap in rule_re.captures_iter(css) {
            if let Some(body) = cap.get(1) {
                for prop in body.as_str().split(';') {
                    if let Some(colon) = prop.find(':') {
                        let name = prop[..colon].trim();
                        if !name.is_empty() {
                            props.insert(name.to_string());
                        }
                    }
                }
            }
        }
        props
    }

    fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        let intersection = a.intersection(b).count() as f64;
        let union = a.union(b).count() as f64;
        if union == 0.0 {
            0.0
        } else {
            (intersection / union).clamp(0.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dom_similarity() {
        let a = "<html><body><button>Hi</button></body></html>";
        let b = "<html><body><button>Hi</button><div></div></body></html>";
        let sim = SemanticComparator::compare_dom_structure(a, b).unwrap();
        assert!(sim > 0.5);
    }

    #[test]
    fn test_function_similarity() {
        let original = r#"
        function a() {}
        const b = () => {};
        "#;
        let generated = r#"
        function a() {}
        const c = () => {};
        "#;
        let key = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let sim = SemanticComparator::compare_js_functions(original, generated, &key).unwrap();
        assert!(sim.function_scores.get("a").unwrap() > &0.7);
        assert!(sim.missing_functions.contains(&"b".to_string()));
    }

    #[test]
    fn test_overall_similarity() {
        let original_html =
            "<html><body><button onclick=\"handleClick()\">Go</button></body></html>";
        let generated_html =
            "<html><body><button id=\"btn\" onclick=\"handleClick()\">Go</button></body></html>";
        let original_css = "button { color: red; }";
        let generated_css = "button { color: red; background: blue; }";
        let original_js = "function handleClick() { console.log('hi'); }";
        let generated_js = "function handleClick() { console.log('hi'); }";

        let key = vec!["handleClick".to_string()];
        let result = SemanticComparator::compare_all(
            original_html,
            original_css,
            original_js,
            generated_html,
            generated_css,
            generated_js,
            &key,
        )
        .unwrap();

        assert!(result.overall_similarity > 0.6);
        assert!(result.missing_features.is_empty());
    }
}
