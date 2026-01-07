/// 语义分析器 - 从提取的AST生成高级语义信息
///
/// 这个模块对提取的AST信息进行深层分析，识别：
/// - 函数之间的调用关系
/// - 类的继承和实现关系
/// - 事件处理器的绑定
/// - 框架检测
/// - 模块依赖
use super::types::*;
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// 语义分析器
pub struct SemanticAnalyzer {
    /// 已识别的框架列表
    _frameworks: Vec<FrameworkSignature>,

    /// 事件处理器模式
    _event_patterns: Vec<EventPattern>,
}

/// 框架签名 - 用于识别特定框架
struct FrameworkSignature {
    _name: &'static str,
    _keywords: Vec<&'static str>,
    _patterns: Vec<&'static str>,
}

/// 事件模式 - 用于识别事件处理器
struct EventPattern {
    _event_type: &'static str,
    _pattern: &'static str,
}

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticAnalyzer {
    /// 创建新的语义分析器
    pub fn new() -> Self {
        Self {
            _frameworks: Self::create_framework_signatures(),
            _event_patterns: Self::create_event_patterns(),
        }
    }

    /// 分析已提取的语义信息
    pub fn analyze(&self, semantic: &mut JsSemanticInfo) -> Result<AnalysisResult> {
        let mut result = AnalysisResult::default();

        // 1. 检测框架
        result.detected_frameworks = self.detect_frameworks(semantic);
        semantic.detected_frameworks = result.detected_frameworks.clone();

        // 2. 检测特殊特性
        result.special_features = self.detect_special_features(semantic);
        semantic.special_features = result.special_features.clone();

        // 3. 分析函数属性
        self.analyze_functions(semantic);

        // 4. 检测事件处理器
        result.event_handlers_found = semantic.event_handlers.len();

        Ok(result)
    }

    /// Phase 2 增强：使用 EnhancedAst 进行更精细的分析
    ///
    /// 利用位置信息、JSX 和 TypeScript 数据改进分析精度
    pub fn analyze_with_enhanced_ast(
        &self,
        semantic: &mut JsSemanticInfo,
        enhanced: &crate::swc_extractor::EnhancedAst,
    ) -> Result<AnalysisResult> {
        // 首先执行标准分析
        let mut result = self.analyze(semantic)?;

        // Phase 2 增强：利用 JSX 和 TypeScript 信息

        // 1. 改进 React 检测（如果有 JSX）
        if enhanced.has_jsx && !result.detected_frameworks.contains(&"React".to_string()) {
            // JSX 的存在强烈表明使用 React 或类似框架
            result.detected_frameworks.push("React".to_string());
            semantic.detected_frameworks.push("React".to_string());
        }

        // 2. 改进 TypeScript 检测
        if enhanced.has_typescript {
            result
                .special_features
                .push("typescript_support".to_string());
            semantic
                .special_features
                .push("typescript_support".to_string());
        }

        // 3. 利用位置信息改进精度
        // 将 JSX 元素信息关联到相应的组件函数
        for jsx_elem in &enhanced.jsx_elements {
            if jsx_elem.is_component {
                // 记录 JSX 组件使用
                log::debug!(
                    "Found JSX component: {} at line {}",
                    jsx_elem.name,
                    jsx_elem.location.line
                );
            }
        }

        // 4. 利用 TypeScript 类型信息
        for ts_type in &enhanced.typescript_types {
            log::debug!(
                "Found TypeScript {}: {} at line {}",
                ts_type.kind,
                ts_type.name,
                ts_type.location.line
            );

            // Interface 的存在表明可能在使用 TypeScript
            if ts_type.kind == "interface" {
                result
                    .special_features
                    .push("typescript_interfaces".to_string());
            }
        }

        // 5. 去重和排序
        result.detected_frameworks.sort();
        result.detected_frameworks.dedup();
        result.special_features.sort();
        result.special_features.dedup();

        Ok(result)
    }

    /// 分析函数细节（如调用关系、复杂度等）
    fn analyze_functions(&self, semantic: &mut JsSemanticInfo) {
        for func in &mut semantic.functions {
            // 基于名称推断可能的调用目标
            let name = func.name.as_deref().unwrap_or("");

            // 识别常见的回调函数
            if name.starts_with("on") || name.contains("callback") {
                semantic.event_handlers.push(JsEventHandler {
                    id: func.id.clone(),
                    event_type: self.infer_event_type(name),
                    handler_function_id: func.id.clone(),
                    target_selector: None,
                    is_delegated: false,
                    binding_method: EventBindingMethod::DirectProperty,
                });
            }
        }
    }

    /// 推断事件类型
    fn infer_event_type(&self, func_name: &str) -> String {
        let lower = func_name.to_lowercase();

        // 检查 camelCase 命名模式：onXxx
        if lower.starts_with("on") && lower.len() > 2 {
            let event = &lower[2..]; // 去掉 "on" 前缀
            return event.to_string();
        }

        // 检查 handleXxx 模式
        if lower.starts_with("handle") && lower.len() > 6 {
            let event = &lower[6..]; // 去掉 "handle" 前缀
            return event.to_string();
        }

        "unknown".to_string()
    }

    /// 检测框架
    fn detect_frameworks(&self, semantic: &JsSemanticInfo) -> Vec<String> {
        let mut detected = HashSet::new();

        // 检查全局变量
        for var in &semantic.global_vars {
            match var.as_str() {
                "React" | "ReactDOM" => detected.insert("React".to_string()),
                "Vue" => detected.insert("Vue".to_string()),
                "angular" | "ng" => detected.insert("Angular".to_string()),
                "$" | "jQuery" => detected.insert("jQuery".to_string()),
                "_" => detected.insert("Lodash".to_string()),
                _ => continue,
            };
        }

        // 检查函数模式
        for func in &semantic.functions {
            if let Some(name) = &func.name {
                // React hooks
                if name.starts_with("use") && name.len() > 3 {
                    detected.insert("React".to_string());
                }
                // Vue lifecycle
                if matches!(
                    name.as_str(),
                    "beforeCreate"
                        | "created"
                        | "beforeMount"
                        | "mounted"
                        | "beforeUpdate"
                        | "updated"
                        | "beforeDestroy"
                        | "destroyed"
                ) {
                    detected.insert("Vue".to_string());
                }
            }
        }

        detected.into_iter().collect()
    }

    /// 检测特殊特性
    fn detect_special_features(&self, semantic: &JsSemanticInfo) -> Vec<String> {
        let mut features = vec![];

        if semantic.uses_eval {
            features.push("dynamic_eval".to_string());
        }

        if semantic.uses_dynamic_require {
            features.push("dynamic_require".to_string());
        }

        // 检查异步与生成器
        let has_async = semantic.functions.iter().any(|f| f.is_async);
        let has_generator = semantic.functions.iter().any(|f| f.is_generator);

        if has_async {
            features.push("async_functions".to_string());
        }
        if has_generator {
            features.push("generators".to_string());
        }

        // 检查高复杂度
        if semantic
            .functions
            .iter()
            .any(|f| f.cyclomatic_complexity > 10)
        {
            features.push("high_complexity".to_string());
        }

        // 检查大量函数
        if semantic.functions.len() > 100 {
            features.push("many_functions".to_string());
        }

        features.sort();
        features.dedup();
        features
    }

    /// 创建框架签名
    fn create_framework_signatures() -> Vec<FrameworkSignature> {
        vec![
            FrameworkSignature {
                _name: "React",
                _keywords: vec!["React", "Component", "useState", "useEffect", "jsx"],
                _patterns: vec!["React.createElement", "ReactDOM.render"],
            },
            FrameworkSignature {
                _name: "Vue",
                _keywords: vec!["Vue", "Vuex", "computed", "watch"],
                _patterns: vec!["Vue.extend", "new Vue", "v-if", "v-for"],
            },
            FrameworkSignature {
                _name: "Angular",
                _keywords: vec!["Angular", "Component", "Service", "Module"],
                _patterns: vec!["@Component", "@Injectable", "ngIf"],
            },
            FrameworkSignature {
                _name: "jQuery",
                _keywords: vec!["jQuery", "$"],
                _patterns: vec!["$.ajax", "$(", "jQuery("],
            },
        ]
    }

    /// 创建事件模式
    fn create_event_patterns() -> Vec<EventPattern> {
        vec![
            EventPattern {
                _event_type: "click",
                _pattern: r#"addEventListener\s*\(\s*["']click["']"#,
            },
            EventPattern {
                _event_type: "change",
                _pattern: r#"addEventListener\s*\(\s*["']change["']"#,
            },
            EventPattern {
                _event_type: "submit",
                _pattern: r#"addEventListener\s*\(\s*["']submit["']"#,
            },
            EventPattern {
                _event_type: "load",
                _pattern: r#"addEventListener\s*\(\s*["']load["']"#,
            },
        ]
    }
}

/// 分析结果汇总
#[derive(Debug, Clone, Default)]
pub struct AnalysisResult {
    pub detected_frameworks: Vec<String>,
    pub special_features: Vec<String>,
    pub event_handlers_found: usize,
    pub analysis_duration_ms: u64,
}

/// 函数调用分析结果
#[derive(Debug, Clone)]
pub struct FunctionCallAnalysis {
    /// 函数ID到其调用的函数列表的映射
    pub call_graph: HashMap<String, Vec<String>>,

    /// 循环调用检测
    pub circular_calls: Vec<Vec<String>>,

    /// 入口点函数
    pub entry_points: Vec<String>,
}

impl FunctionCallAnalysis {
    /// 检测循环调用
    pub fn detect_cycles(&self) -> Vec<Vec<String>> {
        let mut cycles = vec![];
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for start_node in self.call_graph.keys() {
            if !visited.contains(start_node) {
                Self::dfs(
                    start_node,
                    &self.call_graph,
                    &mut visited,
                    &mut rec_stack,
                    &mut vec![],
                    &mut cycles,
                );
            }
        }

        cycles
    }

    /// DFS算法用于循环检测
    fn dfs(
        node: &str,
        graph: &HashMap<String, Vec<String>>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
        path: &mut Vec<String>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());
        path.push(node.to_string());

        if let Some(neighbors) = graph.get(node) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    Self::dfs(neighbor, graph, visited, rec_stack, path, cycles);
                } else if rec_stack.contains(neighbor) {
                    // 找到循环
                    if let Some(idx) = path.iter().position(|n| n == neighbor) {
                        let cycle = path[idx..].to_vec();
                        cycles.push(cycle);
                    }
                }
            }
        }

        path.pop();
        rec_stack.remove(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = SemanticAnalyzer::new();
        assert!(!analyzer._frameworks.is_empty());
        assert!(!analyzer._event_patterns.is_empty());
    }

    #[test]
    fn test_framework_detection_react() {
        let analyzer = SemanticAnalyzer::new();
        let mut semantic = JsSemanticInfo::default();
        semantic.global_vars.push("React".to_string());

        let result = analyzer.analyze(&mut semantic).unwrap();
        assert!(result.detected_frameworks.contains(&"React".to_string()));
    }

    #[test]
    fn test_framework_detection_vue() {
        let analyzer = SemanticAnalyzer::new();
        let mut semantic = JsSemanticInfo::default();
        semantic.global_vars.push("Vue".to_string());

        let result = analyzer.analyze(&mut semantic).unwrap();
        assert!(result.detected_frameworks.contains(&"Vue".to_string()));
    }

    #[test]
    fn test_framework_detection_angular() {
        let analyzer = SemanticAnalyzer::new();
        let mut semantic = JsSemanticInfo::default();
        semantic.global_vars.push("angular".to_string());

        let result = analyzer.analyze(&mut semantic).unwrap();
        assert!(result.detected_frameworks.contains(&"Angular".to_string()));
    }

    #[test]
    fn test_special_features_async() {
        let analyzer = SemanticAnalyzer::new();
        let mut semantic = JsSemanticInfo::default();
        semantic.functions.push(JsFunctionInfo {
            id: "func_0".to_string(),
            name: Some("asyncFunc".to_string()),
            scope_level: 0,
            parameters: vec![],
            return_type_hint: None,
            statement_count: 1,
            cyclomatic_complexity: 1,
            is_async: true,
            is_generator: false,
            captured_vars: vec![],
            local_vars: vec![],
            called_functions: vec![],
            start_line: 0,
            end_line: 0,
        });

        let result = analyzer.analyze(&mut semantic).unwrap();
        assert!(result
            .special_features
            .contains(&"async_functions".to_string()));
    }

    #[test]
    fn test_special_features_generator() {
        let analyzer = SemanticAnalyzer::new();
        let mut semantic = JsSemanticInfo::default();
        semantic.functions.push(JsFunctionInfo {
            id: "func_0".to_string(),
            name: Some("genFunc".to_string()),
            scope_level: 0,
            parameters: vec![],
            return_type_hint: None,
            statement_count: 1,
            cyclomatic_complexity: 1,
            is_async: false,
            is_generator: true,
            captured_vars: vec![],
            local_vars: vec![],
            called_functions: vec![],
            start_line: 0,
            end_line: 0,
        });

        let result = analyzer.analyze(&mut semantic).unwrap();
        assert!(result.special_features.contains(&"generators".to_string()));
    }

    #[test]
    fn test_event_handler_detection() {
        let analyzer = SemanticAnalyzer::new();
        let mut semantic = JsSemanticInfo::default();
        semantic.functions.push(JsFunctionInfo {
            id: "func_0".to_string(),
            name: Some("onClick".to_string()),
            scope_level: 0,
            parameters: vec![],
            return_type_hint: None,
            statement_count: 1,
            cyclomatic_complexity: 1,
            is_async: false,
            is_generator: false,
            captured_vars: vec![],
            local_vars: vec![],
            called_functions: vec![],
            start_line: 0,
            end_line: 0,
        });

        let result = analyzer.analyze(&mut semantic).unwrap();
        // 事件处理器应该被检测出来
        assert!(!semantic.event_handlers.is_empty());
    }

    #[test]
    fn test_high_complexity_detection() {
        let analyzer = SemanticAnalyzer::new();
        let mut semantic = JsSemanticInfo::default();
        semantic.functions.push(JsFunctionInfo {
            id: "func_0".to_string(),
            name: Some("complexFunc".to_string()),
            scope_level: 0,
            parameters: vec![],
            return_type_hint: None,
            statement_count: 1,
            cyclomatic_complexity: 15, // > 10，应该被标记
            is_async: false,
            is_generator: false,
            captured_vars: vec![],
            local_vars: vec![],
            called_functions: vec![],
            start_line: 0,
            end_line: 0,
        });

        let result = analyzer.analyze(&mut semantic).unwrap();
        assert!(result
            .special_features
            .contains(&"high_complexity".to_string()));
    }

    #[test]
    fn test_many_functions_detection() {
        let analyzer = SemanticAnalyzer::new();
        let mut semantic = JsSemanticInfo::default();

        // 添加 101 个函数
        for i in 0..101 {
            semantic.functions.push(JsFunctionInfo {
                id: format!("func_{}", i),
                name: Some(format!("func{}", i)),
                scope_level: 0,
                parameters: vec![],
                return_type_hint: None,
                statement_count: 1,
                cyclomatic_complexity: 1,
                is_async: false,
                is_generator: false,
                captured_vars: vec![],
                local_vars: vec![],
                called_functions: vec![],
                start_line: 0,
                end_line: 0,
            });
        }

        let result = analyzer.analyze(&mut semantic).unwrap();
        assert!(result
            .special_features
            .contains(&"many_functions".to_string()));
    }

    #[test]
    fn test_event_type_inference() {
        let analyzer = SemanticAnalyzer::new();

        assert_eq!(analyzer.infer_event_type("onClick"), "click");
        assert_eq!(analyzer.infer_event_type("onChange"), "change");
        assert_eq!(analyzer.infer_event_type("onSubmit"), "submit");
        assert_eq!(analyzer.infer_event_type("onLoad"), "load");
        assert_eq!(analyzer.infer_event_type("handleFocus"), "focus");
    }
}
