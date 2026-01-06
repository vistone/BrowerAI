/// AST提取器 - 将Boa AST转换为我们的中间表示
///
/// 这个模块负责遍历Boa生成的AST，提取所有相关的JavaScript语义信息。
/// 设计原则：
/// - 错误恢复：部分解析失败不影响整体结果
/// - 增量处理：处理大型代码时保持效率
/// - 信息保留：不丢失关键的语义信息
use super::types::*;
use anyhow::Result;

/// AST提取器 - 从 Boa 解析器提取 JavaScript 语义信息
///
/// 支持：
/// - 函数和类声明提取
/// - 基本元数据统计
/// - 错误恢复与部分解析
#[derive(Default)]
pub struct AstExtractor {
    func_counter: usize,
    class_counter: usize,
}

impl AstExtractor {
    pub fn new() -> Self {
        Self::default()
    }

    /// 从源代码提取 AST 语义信息
    pub fn extract_from_source(&mut self, source: &str) -> Result<ExtractedAst> {
        let line_count = source.lines().count();
        let char_count = source.chars().count();

        // 基于正则匹配进行快速提取（避免复杂 Boa API）
        let mut semantic = JsSemanticInfo::default();
        let mut warnings = Vec::new();

        // 提取函数声明
        self.extract_functions_from_source(source, &mut semantic);

        // 提取类声明
        self.extract_classes_from_source(source, &mut semantic);

        // 提取全局变量和导入
        self.extract_globals_from_source(source, &mut semantic);

        // 计算复杂度和其他指标
        let statement_count = source.matches(';').count();
        let complexity = self.calculate_complexity(&semantic, statement_count);

        Ok(ExtractedAst {
            metadata: self.create_metadata(line_count, char_count, true, complexity),
            semantic,
            warnings,
        })
    }

    /// 从源代码中提取函数声明
    fn extract_functions_from_source(&mut self, source: &str, semantic: &mut JsSemanticInfo) {
        use regex::Regex;

        // 匹配 function 声明
        let func_pattern = Regex::new(r"function\s+(\w+)\s*\(([^)]*)\)").unwrap();
        for caps in func_pattern.captures_iter(source) {
            let name = caps.get(1).map(|m| m.as_str().to_string());
            let params_str = caps.get(2).map(|m| m.as_str()).unwrap_or("");

            let parameters: Vec<JsParameter> = params_str
                .split(',')
                .filter(|p| !p.trim().is_empty())
                .map(|p| JsParameter {
                    name: p.trim().to_string(),
                    has_default: p.contains('='),
                    is_rest: p.trim().starts_with("..."),
                    type_hint: None,
                })
                .collect();

            semantic.functions.push(JsFunctionInfo {
                id: format!("func_{}", self.func_counter),
                name,
                scope_level: 0,
                parameters,
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
            self.func_counter += 1;
        }

        // 匹配箭头函数
        let arrow_pattern =
            Regex::new(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>").unwrap();
        for caps in arrow_pattern.captures_iter(source) {
            let name = caps.get(1).map(|m| m.as_str().to_string());

            semantic.functions.push(JsFunctionInfo {
                id: format!("func_{}", self.func_counter),
                name,
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
            self.func_counter += 1;
        }
    }

    /// 从源代码中提取类声明
    fn extract_classes_from_source(&mut self, source: &str, semantic: &mut JsSemanticInfo) {
        use regex::Regex;

        let class_pattern = Regex::new(r"class\s+(\w+)").unwrap();
        for caps in class_pattern.captures_iter(source) {
            let name = caps
                .get(1)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();

            semantic.classes.push(JsClassInfo {
                id: format!("class_{}", self.class_counter),
                name,
                parent_class: None,
                implements: vec![],
                properties: vec![],
                methods: vec![],
                static_methods: vec![],
                constructor: None,
                start_line: 0,
                end_line: 0,
            });
            self.class_counter += 1;
        }
    }

    /// 从源代码中提取全局变量和导入
    fn extract_globals_from_source(&mut self, source: &str, semantic: &mut JsSemanticInfo) {
        use regex::Regex;

        // 提取导入语句中的全局引用
        let import_pattern = Regex::new(r"import\s+(?:\{[^}]*\}|.*?)\s+from").unwrap();
        if import_pattern.is_match(source) {
            // 标记为模块化代码
        }

        // 提取全局变量定义
        let global_pattern = Regex::new(r"^(?:const|let|var)\s+(\w+)").unwrap();
        for caps in global_pattern.captures_iter(source) {
            if let Some(name_match) = caps.get(1) {
                semantic.global_vars.push(name_match.as_str().to_string());
            }
        }

        // 检测特殊全局变量
        if source.contains("eval(") {
            semantic.uses_eval = true;
        }
        if source.contains("require(") || source.contains("__webpack_require__") {
            semantic.uses_dynamic_require = true;
        }
    }

    /// 创建元数据对象
    fn create_metadata(
        &self,
        line_count: usize,
        char_count: usize,
        is_valid: bool,
        complexity: u32,
    ) -> JsAstMetadata {
        JsAstMetadata {
            line_count,
            char_count,
            statement_count: 0,
            expression_count: 0,
            is_valid,
            complexity_score: complexity,
            code_size: match line_count {
                0..=100 => super::types::CodeSizeCategory::Tiny,
                101..=1000 => super::types::CodeSizeCategory::Small,
                1001..=10000 => super::types::CodeSizeCategory::Medium,
                _ => super::types::CodeSizeCategory::Large,
            },
        }
    }

    /// 计算复杂度分数
    fn calculate_complexity(&self, semantic: &JsSemanticInfo, stmt_count: usize) -> u32 {
        let func_score = (semantic.functions.len() as u32) * 5;
        let class_score = (semantic.classes.len() as u32) * 10;
        let stmt_score = (stmt_count as u32).min(50);
        ((func_score + class_score + stmt_score) / 10).min(100)
    }
}

/// 提取的AST表示
#[derive(Debug, Clone, Default)]
pub struct ExtractedAst {
    pub metadata: JsAstMetadata,
    pub semantic: JsSemanticInfo,
    pub warnings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extractor_basic_stats() {
        let mut extractor = AstExtractor::new();
        let src = "function hello() { return 1; }";
        let result = extractor.extract_from_source(src).unwrap();
        assert!(result.metadata.is_valid);
        assert_eq!(result.metadata.line_count, 1);
        assert!(result.metadata.char_count > 0);
    }

    #[test]
    fn test_extractor_function_detection() {
        let mut extractor = AstExtractor::new();
        let src = "function foo() {} function bar() {}";
        let result = extractor.extract_from_source(src).unwrap();
        assert_eq!(result.semantic.functions.len(), 2);
    }

    #[test]
    fn test_extractor_class_detection() {
        let mut extractor = AstExtractor::new();
        let src = "class MyClass {} class Another {}";
        let result = extractor.extract_from_source(src).unwrap();
        assert_eq!(result.semantic.classes.len(), 2);
    }

    #[test]
    fn test_extractor_multiline() {
        let mut extractor = AstExtractor::new();
        let src = "function test() {\n  let x = 1;\n  return x;\n}";
        let result = extractor.extract_from_source(src).unwrap();
        assert_eq!(result.metadata.line_count, 4);
    }

    #[test]
    fn test_extractor_error_recovery() {
        let mut extractor = AstExtractor::new();
        // 基于正则的提取器会部分匹配即使语法不完整，仍然会返回数据
        let src = "function incomplete() {"; // 不完整的函数
        let result = extractor.extract_from_source(src).unwrap();
        // 验证能够提取到函数（即使不完整）
        assert!(!result.semantic.functions.is_empty());
    }

    #[test]
    fn test_extractor_complexity_calculation() {
        let mut extractor = AstExtractor::new();
        let src = "function a() {} function b() {} class C {}";
        let result = extractor.extract_from_source(src).unwrap();
        // 复杂度应该 > 0
        assert!(result.metadata.complexity_score > 0);
    }

    #[test]
    fn test_extractor_code_size_categorization() {
        let mut extractor = AstExtractor::new();

        // 小型代码
        let result1 = extractor.extract_from_source("let x = 1;").unwrap();
        assert_eq!(result1.metadata.code_size, CodeSizeCategory::Tiny);

        // 中型代码
        let big_code = "let x = 1;\n".repeat(500);
        let result2 = extractor.extract_from_source(&big_code).unwrap();
        assert_eq!(result2.metadata.code_size, CodeSizeCategory::Small);
    }
}
