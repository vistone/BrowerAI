//! 符号执行引擎 (Symbolic Execution Engine)
//!
//! 通过符号执行追踪变量值，进行更深层的代码分析
//! 支持：变量追踪、条件分析、值推导、常量传播

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 符号值类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SymbolicValue {
    Constant(String),           // 常量值
    Variable(String),            // 变量名
    BinaryOp {
        left: Box<SymbolicValue>,
        op: String,
        right: Box<SymbolicValue>,
    },
    Call {
        function: String,
        args: Vec<SymbolicValue>,
    },
    Array(Vec<SymbolicValue>),
    Object(HashMap<String, SymbolicValue>),
    Unknown,
}

/// 执行上下文
#[derive(Debug, Clone, Default)]
pub struct ExecutionContext {
    /// 变量值映射
    variables: HashMap<String, SymbolicValue>,
    /// 字符串表 (常用于追踪字符串解码)
    string_table: HashMap<String, String>,
    /// 数组表 (追踪数组初始化和访问)
    array_table: HashMap<String, Vec<String>>,
    /// 函数映射
    functions: HashMap<String, FunctionInfo>,
}

#[derive(Debug, Clone)]
struct FunctionInfo {
    name: String,
    params: Vec<String>,
    body: String,
}

/// 符号执行引擎
pub struct SymbolicExecutor {
    context: ExecutionContext,
    max_depth: usize,
    current_depth: usize,
}

impl SymbolicExecutor {
    /// 创建新的符号执行器
    pub fn new() -> Self {
        Self {
            context: ExecutionContext::default(),
            max_depth: 100,
            current_depth: 0,
        }
    }

    /// 分析代码并返回推导结果
    pub fn analyze(&mut self, code: &str) -> Result<SymbolicAnalysisResult> {
        let mut result = SymbolicAnalysisResult::default();

        // 解析变量赋值
        self.extract_assignments(code, &mut result)?;

        // 解析字符串操作
        self.extract_string_operations(code, &mut result)?;

        // 解析数组操作
        self.extract_array_operations(code, &mut result)?;

        // 解析函数调用
        self.extract_function_calls(code, &mut result)?;

        // 执行常量传播
        self.constant_propagation(&mut result)?;

        // 追踪变量流
        self.trace_data_flow(code, &mut result)?;

        Ok(result)
    }

    /// 提取变量赋值
    fn extract_assignments(&mut self, code: &str, result: &mut SymbolicAnalysisResult) -> Result<()> {
        // 匹配 var/let/const x = value 模式
        let patterns = vec![
            r#"(?:var|let|const)\s+(\w+)\s*=\s*['"]([^'"]+)['"]\s*;"#,  // 字符串赋值
            r#"(?:var|let|const)\s+(\w+)\s*=\s*(\d+)\s*;"#,              // 数字赋值
            r#"(?:var|let|const)\s+(\w+)\s*=\s*(\[.*?\])\s*;"#,          // 数组赋值
            r#"(\w+)\s*=\s*['"]([^'"]+)['"]\s*;"#,                       // 重新赋值
        ];

        for pattern in patterns {
            let re = Regex::new(pattern)?;
            for caps in re.captures_iter(code) {
                if let (Some(var_name), Some(value)) = (caps.get(1), caps.get(2)) {
                    let var = var_name.as_str().to_string();
                    let val = value.as_str().to_string();

                    self.context
                        .variables
                        .insert(var.clone(), SymbolicValue::Constant(val.clone()));

                    result.assignments.push(Assignment {
                        variable: var,
                        value: val,
                        line: 0, // 可以改进为追踪行号
                    });
                }
            }
        }

        Ok(())
    }

    /// 提取字符串操作
    fn extract_string_operations(
        &mut self,
        code: &str,
        result: &mut SymbolicAnalysisResult,
    ) -> Result<()> {
        // String.fromCharCode
        let re = Regex::new(r"String\.fromCharCode\s*\(([^)]+)\)")?;
        for caps in re.captures_iter(code) {
            if let Some(codes) = caps.get(1) {
                let decoded = self.decode_char_codes(codes.as_str());
                result.decoded_strings.push(decoded);
            }
        }

        // unescape
        let re = Regex::new(r#"unescape\s*\(\s*["']([^"']+)["']\s*\)"#)?;
        for caps in re.captures_iter(code) {
            if let Some(escaped) = caps.get(1) {
                let decoded = self.unescape_string(escaped.as_str());
                result.decoded_strings.push(decoded);
            }
        }

        // atob
        let re = Regex::new(r#"atob\s*\(\s*["']([^"']+)["']\s*\)"#)?;
        for caps in re.captures_iter(code) {
            if let Some(b64) = caps.get(1) {
                if let Ok(decoded) = self.decode_base64(b64.as_str()) {
                    result.decoded_strings.push(decoded);
                }
            }
        }

        Ok(())
    }

    /// 提取数组操作
    fn extract_array_operations(
        &mut self,
        code: &str,
        result: &mut SymbolicAnalysisResult,
    ) -> Result<()> {
        // 匹配 var arr = [...]
        let re = Regex::new(r#"(?:var|let|const)\s+(\w+)\s*=\s*\[(.*?)\]"#)?;

        for caps in re.captures_iter(code) {
            if let (Some(arr_name), Some(elements)) = (caps.get(1), caps.get(2)) {
                let name = arr_name.as_str().to_string();
                let elems: Vec<String> = elements
                    .as_str()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();

                self.context.array_table.insert(name.clone(), elems.clone());

                result.array_operations.push(ArrayOperation {
                    array_name: name,
                    elements: elems,
                });
            }
        }

        Ok(())
    }

    /// 提取函数调用
    fn extract_function_calls(
        &mut self,
        code: &str,
        result: &mut SymbolicAnalysisResult,
    ) -> Result<()> {
        // 匹配函数定义 function name(params) { body }
        let re = Regex::new(r"function\s+(\w+)\s*\(([^)]*)\)\s*\{([^}]+)\}")?;

        for caps in re.captures_iter(code) {
            if let (Some(name), Some(params), Some(body)) =
                (caps.get(1), caps.get(2), caps.get(3))
            {
                let param_list: Vec<String> = params
                    .as_str()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();

                result.function_calls.push(FunctionCall {
                    name: name.as_str().to_string(),
                    parameters: param_list,
                    return_type: "unknown".to_string(),
                });
            }
        }

        Ok(())
    }

    /// 常量传播
    fn constant_propagation(&self, result: &mut SymbolicAnalysisResult) -> Result<()> {
        // 识别可以传播的常量
        for assignment in &result.assignments {
            // 如果值是简单的字符串或数字，标记为常量
            if assignment.value.chars().all(|c| c.is_numeric() || c == '.' || c == '-') {
                result.constants.push(assignment.variable.clone());
            }
        }

        Ok(())
    }

    /// 追踪数据流
    fn trace_data_flow(&self, code: &str, result: &mut SymbolicAnalysisResult) -> Result<()> {
        // 追踪变量使用和定义
        for (var, value) in &self.context.variables {
            let use_count = code.matches(var).count();
            result.data_flows.push(DataFlow {
                variable: var.clone(),
                definition_type: "assignment".to_string(),
                usage_count: use_count,
                value: format!("{:?}", value),
            });
        }

        Ok(())
    }

    // 辅助函数
    fn decode_char_codes(&self, codes: &str) -> String {
        codes
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .filter_map(char::from_u32)
            .collect()
    }

    fn unescape_string(&self, s: &str) -> String {
        let mut result = String::new();
        let mut chars = s.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '%' {
                let hex: String = chars.by_ref().take(2).collect();
                if hex.len() == 2 {
                    if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                        result.push(byte as char);
                        continue;
                    }
                }
            }
            result.push(ch);
        }

        result
    }

    fn decode_base64(&self, b64: &str) -> Result<String> {
        use base64::Engine as _;
        let decoded = base64::engine::general_purpose::STANDARD.decode(b64)?;
        Ok(String::from_utf8(decoded)?)
    }
}

impl Default for SymbolicExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// 分析结果
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolicAnalysisResult {
    pub assignments: Vec<Assignment>,
    pub decoded_strings: Vec<String>,
    pub array_operations: Vec<ArrayOperation>,
    pub function_calls: Vec<FunctionCall>,
    pub constants: Vec<String>,
    pub data_flows: Vec<DataFlow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    pub variable: String,
    pub value: String,
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayOperation {
    pub array_name: String,
    pub elements: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub parameters: Vec<String>,
    pub return_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlow {
    pub variable: String,
    pub definition_type: String,
    pub usage_count: usize,
    pub value: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_extraction() {
        let code = r#"
            var x = "hello";
            let y = 42;
            const z = "world";
        "#;

        let mut executor = SymbolicExecutor::new();
        let result = executor.analyze(code).unwrap();

        assert!(result.assignments.iter().any(|a| a.variable == "x"));
        assert!(result.assignments.iter().any(|a| a.variable == "y"));
        assert!(result.assignments.iter().any(|a| a.variable == "z"));
    }

    #[test]
    fn test_string_decoding() {
        let code = r#"
            var msg = String.fromCharCode(72, 101, 108, 108, 111);
            var unesc = unescape("%48%65%6C%6C%6F");
        "#;

        let mut executor = SymbolicExecutor::new();
        let result = executor.analyze(code).unwrap();

        assert!(!result.decoded_strings.is_empty());
        assert!(result.decoded_strings.iter().any(|s| s.contains("Hello")));
    }

    #[test]
    fn test_array_extraction() {
        let code = r#"
            var arr = ["a", "b", "c"];
            var nums = [1, 2, 3];
        "#;

        let mut executor = SymbolicExecutor::new();
        let result = executor.analyze(code).unwrap();

        assert_eq!(result.array_operations.len(), 2);
    }

    #[test]
    fn test_function_extraction() {
        let code = r#"
            function decode(str, key) {
                return str + key;
            }
        "#;

        let mut executor = SymbolicExecutor::new();
        let result = executor.analyze(code).unwrap();

        assert!(result.function_calls.iter().any(|f| f.name == "decode"));
    }

    #[test]
    fn test_constant_propagation() {
        let code = r#"
            var PI = 3.14159;
            var count = 42;
        "#;

        let mut executor = SymbolicExecutor::new();
        let result = executor.analyze(code).unwrap();

        assert!(!result.constants.is_empty());
    }
}
