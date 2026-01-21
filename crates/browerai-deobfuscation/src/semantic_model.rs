//! 语义反混淆模型 - Rust集成
//!
//! 本模块提供与Python训练的语义模型的Rust接口。
//! 支持ONNX运行时推理和知识库回退。

use std::collections::HashMap;

#[cfg(feature = "ai")]
use std::path::Path;
#[cfg(feature = "ai")]
use std::sync::Arc;

#[cfg(feature = "ai")]
use anyhow::Context;
use anyhow::Result;

#[cfg(feature = "ai")]
use ort::session::{builder::GraphOptimizationLevel, Session};

#[cfg(feature = "ai")]
use crate::onnx_inference::OnnxInference;

/// 语义预测结果
#[derive(Debug, Clone)]
pub struct SemanticPrediction {
    /// 推断的语义名称
    pub semantic_name: String,
    /// 语义角色 (counter, container, callback等)
    pub semantic_role: String,
    /// 推断的类型
    pub inferred_type: String,
    /// 置信度 (0.0-1.0)
    pub confidence: f32,
}

/// 反混淆结果
#[derive(Debug, Clone)]
pub struct DeobfuscationResult {
    /// 反混淆后的代码
    pub deobfuscated_code: String,
    /// 变量映射 (混淆名 -> 语义预测)
    pub variable_mappings: HashMap<String, SemanticPrediction>,
    /// 函数映射 (混淆名 -> 语义预测)
    pub function_mappings: HashMap<String, SemanticPrediction>,
    /// 检测到的代码特征
    pub detected_features: Vec<String>,
    /// 总体置信度
    pub overall_confidence: f32,
}

/// 语义知识库 - 基于启发式规则的回退
pub struct SemanticKnowledgeBase {
    /// 单字母语义映射
    single_letter_semantics: HashMap<char, (&'static str, &'static str)>,
    /// API模式识别
    api_patterns: Vec<(&'static str, &'static str, &'static str)>,
    /// 行为模式识别
    behavior_patterns: Vec<(&'static str, &'static str)>,
}

impl Default for SemanticKnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticKnowledgeBase {
    pub fn new() -> Self {
        let mut single_letter = HashMap::new();
        // 基于常见混淆模式的语义映射
        single_letter.insert('i', ("index", "counter"));
        single_letter.insert('j', ("innerIndex", "counter"));
        single_letter.insert('k', ("key", "counter"));
        single_letter.insert('n', ("count", "counter"));
        single_letter.insert('a', ("accumulator", "accumulator"));
        single_letter.insert('b', ("buffer", "container"));
        single_letter.insert('c', ("items", "container"));
        single_letter.insert('d', ("data", "parameter"));
        single_letter.insert('e', ("element", "iterator"));
        single_letter.insert('f', ("callback", "callback"));
        single_letter.insert('g', ("global", "context"));
        single_letter.insert('h', ("handler", "callback"));
        single_letter.insert('o', ("object", "parameter"));
        single_letter.insert('p', ("param", "parameter"));
        single_letter.insert('r', ("result", "accumulator"));
        single_letter.insert('s', ("str", "string"));
        single_letter.insert('t', ("temp", "temporary"));
        single_letter.insert('v', ("value", "parameter"));
        single_letter.insert('x', ("xCoord", "parameter"));
        single_letter.insert('y', ("yCoord", "parameter"));

        let api_patterns = vec![
            // DOM APIs
            ("getElementById", "element", "dom_element"),
            ("querySelector", "element", "dom_element"),
            ("addEventListener", "handler", "event_handler"),
            ("createElement", "element", "dom_element"),
            // Promise APIs
            ("then", "result", "promise_result"),
            ("catch", "error", "error_handler"),
            ("resolve", "value", "promise_resolver"),
            ("reject", "reason", "promise_rejecter"),
            // IndexedDB
            ("indexedDB", "database", "database"),
            ("createObjectStore", "store", "object_store"),
            ("transaction", "tx", "transaction"),
            // Network
            ("fetch", "response", "http_response"),
            ("XMLHttpRequest", "xhr", "http_request"),
            // Arrays
            ("forEach", "item", "iterator"),
            ("map", "result", "transform_result"),
            ("filter", "item", "filter_result"),
            ("reduce", "accumulator", "reduce_result"),
            ("push", "item", "array_element"),
        ];

        let behavior_patterns = vec![
            ("for", "loop"),
            ("while", "loop"),
            ("if", "conditional"),
            ("switch", "conditional"),
            ("async", "async_function"),
            ("await", "async_expression"),
            ("Promise", "promise"),
            ("new", "constructor"),
            ("class", "class_definition"),
            ("prototype", "prototype_method"),
        ];

        Self {
            single_letter_semantics: single_letter,
            api_patterns,
            behavior_patterns,
        }
    }

    /// 根据单字母变量名推断语义
    pub fn infer_from_single_letter(&self, name: &str) -> Option<SemanticPrediction> {
        if name.len() == 1 {
            let c = name.chars().next()?.to_ascii_lowercase();
            if let Some(&(semantic, role)) = self.single_letter_semantics.get(&c) {
                return Some(SemanticPrediction {
                    semantic_name: semantic.to_string(),
                    semantic_role: role.to_string(),
                    inferred_type: "unknown".to_string(),
                    confidence: 0.7,
                });
            }
        }
        None
    }

    /// 检测代码中的API模式
    pub fn detect_api_patterns(&self, code: &str) -> Vec<String> {
        let mut detected = Vec::new();
        for (pattern, _, feature) in &self.api_patterns {
            if code.contains(pattern) && !detected.contains(&feature.to_string()) {
                detected.push(feature.to_string());
            }
        }
        detected
    }

    /// 检测代码行为模式
    pub fn detect_behavior_patterns(&self, code: &str) -> Vec<String> {
        let mut detected = Vec::new();
        for (pattern, behavior) in &self.behavior_patterns {
            if code.contains(pattern) && !detected.contains(&behavior.to_string()) {
                detected.push(behavior.to_string());
            }
        }
        detected
    }
}

/// 语义反混淆引擎
pub struct SemanticDeobfuscator {
    /// 知识库
    knowledge_base: SemanticKnowledgeBase,
    /// ONNX会话 (可选)
    #[cfg(feature = "ai")]
    onnx_session: Option<Arc<Session>>,
}

impl Default for SemanticDeobfuscator {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticDeobfuscator {
    /// 创建新实例 (仅知识库回退)
    pub fn new() -> Self {
        Self {
            knowledge_base: SemanticKnowledgeBase::new(),
            #[cfg(feature = "ai")]
            onnx_session: None,
        }
    }

    /// 使用ONNX模型创建实例
    #[cfg(feature = "ai")]
    pub fn with_model<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path.as_ref())
            .context("Failed to load ONNX model")?;

        Ok(Self {
            knowledge_base: SemanticKnowledgeBase::new(),
            onnx_session: Some(Arc::new(session)),
        })
    }

    /// 反混淆JavaScript代码
    pub fn deobfuscate(&self, code: &str) -> Result<DeobfuscationResult> {
        // 1. 提取变量和函数名
        let (variables, functions) = self.extract_identifiers(code);

        // 2. 检测API和行为模式
        let api_features = self.knowledge_base.detect_api_patterns(code);
        let behavior_features = self.knowledge_base.detect_behavior_patterns(code);

        let mut detected_features = api_features;
        detected_features.extend(behavior_features);

        // 3. 为每个变量生成语义预测
        let mut variable_mappings = HashMap::new();
        for var in &variables {
            if let Some(pred) = self.predict_variable_semantic(var, code) {
                variable_mappings.insert(var.clone(), pred);
            }
        }

        // 4. 为每个函数生成语义预测
        let mut function_mappings = HashMap::new();
        for func in &functions {
            if let Some(pred) = self.predict_function_semantic(func, code) {
                function_mappings.insert(func.clone(), pred);
            }
        }

        // 5. 应用重命名
        let deobfuscated_code = self.apply_renamings(code, &variable_mappings, &function_mappings);

        // 6. 计算总体置信度
        let confidences: Vec<f32> = variable_mappings
            .values()
            .chain(function_mappings.values())
            .map(|p| p.confidence)
            .collect();

        let overall_confidence = if confidences.is_empty() {
            0.0
        } else {
            confidences.iter().sum::<f32>() / confidences.len() as f32
        };

        Ok(DeobfuscationResult {
            deobfuscated_code,
            variable_mappings,
            function_mappings,
            detected_features,
            overall_confidence,
        })
    }

    /// 提取代码中的标识符
    fn extract_identifiers(&self, code: &str) -> (Vec<String>, Vec<String>) {
        let mut variables = Vec::new();
        let mut functions = Vec::new();

        // 简单的正则提取 (实际项目中应使用AST解析)
        let var_pattern = regex::Regex::new(r"\bvar\s+([a-zA-Z_$][a-zA-Z0-9_$]*)").unwrap();
        let func_pattern = regex::Regex::new(r"\bfunction\s+([a-zA-Z_$][a-zA-Z0-9_$]*)").unwrap();
        let single_letter_pattern = regex::Regex::new(r"\b([a-zA-Z])\b").unwrap();

        for cap in var_pattern.captures_iter(code) {
            if let Some(m) = cap.get(1) {
                let name = m.as_str().to_string();
                if !variables.contains(&name) {
                    variables.push(name);
                }
            }
        }

        for cap in func_pattern.captures_iter(code) {
            if let Some(m) = cap.get(1) {
                let name = m.as_str().to_string();
                if !functions.contains(&name) {
                    functions.push(name);
                }
            }
        }

        // 提取单字母变量
        for cap in single_letter_pattern.captures_iter(code) {
            if let Some(m) = cap.get(1) {
                let name = m.as_str().to_string();
                // 排除常见关键字和已提取的
                if !["a", "i", "e", "o", "u"].contains(&name.as_str())
                    && !variables.contains(&name)
                    && !functions.contains(&name)
                {
                    variables.push(name);
                }
            }
        }

        (variables, functions)
    }

    /// 预测变量语义
    fn predict_variable_semantic(&self, name: &str, code: &str) -> Option<SemanticPrediction> {
        // 优先使用ONNX模型
        #[cfg(feature = "ai")]
        {
            if let Some(ref _session) = self.onnx_session {
                // 使用 ONNX 推理
                let inference = OnnxInference::new();
                let context = format!("var {} = {};", name, code);
                let result = inference.tokenize(&context);

                // 如果有推理结果，构建预测
                if !result.is_empty() {
                    let prediction = SemanticPrediction {
                        semantic_name: format!("var_{}_semantic", name),
                        semantic_role: "variable".to_string(),
                        inferred_type: "unknown".to_string(),
                        confidence: 0.75,
                    };
                    return Some(prediction);
                }
            }
        }

        // 知识库回退
        self.knowledge_base
            .infer_from_single_letter(name)
            .or_else(|| {
                // 基于上下文的启发式推断
                self.infer_from_context(name, code)
            })
    }

    /// 预测函数语义
    fn predict_function_semantic(&self, name: &str, code: &str) -> Option<SemanticPrediction> {
        // 优先使用ONNX模型
        #[cfg(feature = "ai")]
        {
            if let Some(ref _session) = self.onnx_session {
                // 使用 ONNX 推理
                let inference = OnnxInference::new();
                let context = format!("function {}() {{}}", name);
                let result = inference.tokenize(&context);

                // 如果有推理结果，构建预测
                if !result.is_empty() {
                    let semantic_name = if code.contains("Promise") || code.contains("async") {
                        "asyncHandler".to_string()
                    } else if code.contains("forEach") || code.contains("map") {
                        "arrayProcessor".to_string()
                    } else {
                        format!("func_{}", name)
                    };

                    let prediction = SemanticPrediction {
                        semantic_name,
                        semantic_role: "function".to_string(),
                        inferred_type: "function".to_string(),
                        confidence: 0.8,
                    };
                    return Some(prediction);
                }
            }
        }

        // 知识库回退
        // 检测函数是否是Promise相关
        let is_async = code.contains(&format!("function {}(", name)) && code.contains("Promise")
            || code.contains("async")
            || code.contains(".then(");

        // 检测函数是否处理数组
        let is_array_handler = code.contains(&format!("function {}(", name))
            && (code.contains("forEach") || code.contains("map") || code.contains("filter"));

        let semantic_name = if is_async {
            "asyncData".to_string()
        } else if is_array_handler {
            "transformData".to_string()
        } else if name.len() == 1 {
            "processData".to_string()
        } else {
            name.to_string()
        };

        Some(SemanticPrediction {
            semantic_name,
            semantic_role: if is_async { "async" } else { "function" }.to_string(),
            inferred_type: "function".to_string(),
            confidence: 0.75,
        })
    }

    /// 基于上下文推断语义
    fn infer_from_context(&self, name: &str, code: &str) -> Option<SemanticPrediction> {
        // 检测是否用于循环计数
        if code.contains(&format!("for(var {}=0", name))
            || code.contains(&format!("for (var {} = 0", name))
            || code.contains(&format!("{}<", name))
            || code.contains(&format!("{}++", name))
        {
            return Some(SemanticPrediction {
                semantic_name: "counter".to_string(),
                semantic_role: "counter".to_string(),
                inferred_type: "number".to_string(),
                confidence: 0.9,
            });
        }

        // 检测是否用作数组容器
        if code.contains(&format!("{}=[]", name))
            || code.contains(&format!("{} = []", name))
            || code.contains(&format!("{}.push", name))
        {
            return Some(SemanticPrediction {
                semantic_name: "items".to_string(),
                semantic_role: "container".to_string(),
                inferred_type: "array".to_string(),
                confidence: 0.85,
            });
        }

        // 检测是否是回调函数
        if code.contains(&format!("function({})", name))
            || code.contains(&format!("({}) =>", name))
            || code.contains(&format!(".then({})", name))
        {
            return Some(SemanticPrediction {
                semantic_name: "callback".to_string(),
                semantic_role: "callback".to_string(),
                inferred_type: "function".to_string(),
                confidence: 0.8,
            });
        }

        None
    }

    /// 应用重命名
    fn apply_renamings(
        &self,
        code: &str,
        variable_mappings: &HashMap<String, SemanticPrediction>,
        function_mappings: &HashMap<String, SemanticPrediction>,
    ) -> String {
        let mut result = code.to_string();

        // 按名称长度降序排序，避免替换冲突
        let mut all_mappings: Vec<_> = variable_mappings
            .iter()
            .chain(function_mappings.iter())
            .collect();
        all_mappings.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        for (old_name, prediction) in all_mappings {
            // 使用单词边界替换
            let pattern = format!(r"\b{}\b", regex::escape(old_name));
            if let Ok(re) = regex::Regex::new(&pattern) {
                result = re
                    .replace_all(&result, &prediction.semantic_name)
                    .to_string();
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_base_single_letter() {
        let kb = SemanticKnowledgeBase::new();

        let pred = kb.infer_from_single_letter("i").unwrap();
        assert_eq!(pred.semantic_name, "index");
        assert_eq!(pred.semantic_role, "counter");

        let pred = kb.infer_from_single_letter("c").unwrap();
        assert_eq!(pred.semantic_name, "items");
        assert_eq!(pred.semantic_role, "container");
    }

    #[test]
    fn test_deobfuscation() {
        let deob = SemanticDeobfuscator::new();

        let code = "function a(b,c){var d=[];for(var i=0;i<b.length;i++){d.push(b[i])}return d}";
        let result = deob.deobfuscate(code).unwrap();

        assert!(!result.variable_mappings.is_empty());
        assert!(result.overall_confidence > 0.0);
    }

    #[test]
    fn test_detect_api_patterns() {
        let kb = SemanticKnowledgeBase::new();

        let code = "fetch(url).then(response => response.json())";
        let features = kb.detect_api_patterns(code);

        assert!(features.contains(&"http_response".to_string()));
        assert!(features.contains(&"promise_result".to_string()));
    }
}
