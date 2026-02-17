//! 类型推断系统 (Type Inference System)
//!
//! 从代码使用推断变量类型，改进代码理解
//! 支持：基础类型推断、返回类型推断、污点类型、union 类型

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// JavaScript 类型
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JSType {
    String,
    Number,
    Boolean,
    Undefined,
    Null,
    Object,
    Array,
    Function,
    Symbol,
    BigInt,
    Unknown,
    // Union 类型
    Union(Vec<Box<JSType>>),
    // 约束类型
    Constrained {
        base_type: Box<JSType>,
        constraints: Vec<String>, // 例如：["length > 0", "numeric"]
    },
}

impl std::fmt::Display for JSType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            JSType::String => write!(f, "string"),
            JSType::Number => write!(f, "number"),
            JSType::Boolean => write!(f, "boolean"),
            JSType::Undefined => write!(f, "undefined"),
            JSType::Null => write!(f, "null"),
            JSType::Object => write!(f, "object"),
            JSType::Array => write!(f, "array"),
            JSType::Function => write!(f, "function"),
            JSType::Symbol => write!(f, "symbol"),
            JSType::BigInt => write!(f, "bigint"),
            JSType::Unknown => write!(f, "unknown"),
            JSType::Union(types) => {
                let type_strs: Vec<String> = types.iter().map(|t| t.to_string()).collect();
                write!(f, "{}", type_strs.join(" | "))
            }
            JSType::Constrained {
                base_type,
                constraints,
            } => {
                write!(f, "{} ({})", base_type, constraints.join(", "))
            }
        }
    }
}

/// 变量类型信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    pub variable: String,
    pub inferred_type: JSType,
    pub confidence: f32,       // 0.0 到 1.0
    pub evidence: Vec<String>, // 推断证据
}

/// 函数类型签名
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSignature {
    pub name: String,
    pub parameters: Vec<(String, JSType)>,
    pub return_type: JSType,
}

/// 类型推断器
pub struct TypeInferencer {
    types: HashMap<String, JSType>,
    function_signatures: HashMap<String, FunctionSignature>,
}

impl TypeInferencer {
    /// 创建新的类型推断器
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            function_signatures: Self::create_builtin_signatures(),
        }
    }

    /// 推断代码的类型
    pub fn infer(&mut self, code: &str) -> Result<TypeInferenceResult> {
        let mut result = TypeInferenceResult::default();

        // 第一阶段：推断变量初始化类型
        self.infer_initialization_types(code, &mut result)?;

        // 第二阶段：推断操作类型
        self.infer_operation_types(code, &mut result)?;

        // 第三阶段：推断函数类型
        self.infer_function_types(code, &mut result)?;

        // 第四阶段：推断返回类型
        self.infer_return_types(code, &mut result)?;

        // 第五阶段：类型优化
        self.optimize_types(&mut result)?;

        Ok(result)
    }

    /// 推断初始化类型
    fn infer_initialization_types(
        &mut self,
        code: &str,
        result: &mut TypeInferenceResult,
    ) -> Result<()> {
        // 字符串初始化
        let re = Regex::new(r#"(?:var|let|const)\s+(\w+)\s*=\s*["']([^"']+)["']"#)?;
        for caps in re.captures_iter(code) {
            if let Some(var) = caps.get(1) {
                let var_name = var.as_str();
                self.types.insert(var_name.to_string(), JSType::String);

                result.type_info.push(TypeInfo {
                    variable: var_name.to_string(),
                    inferred_type: JSType::String,
                    confidence: 1.0,
                    evidence: vec!["string literal".to_string()],
                });
            }
        }

        // 数字初始化
        let re = Regex::new(r"(?:var|let|const)\s+(\w+)\s*=\s*(\d+(?:\.\d+)?)")?;
        for caps in re.captures_iter(code) {
            if let Some(var) = caps.get(1) {
                let var_name = var.as_str();
                self.types.insert(var_name.to_string(), JSType::Number);

                result.type_info.push(TypeInfo {
                    variable: var_name.to_string(),
                    inferred_type: JSType::Number,
                    confidence: 1.0,
                    evidence: vec!["numeric literal".to_string()],
                });
            }
        }

        // 布尔初始化
        let re = Regex::new(r"(?:var|let|const)\s+(\w+)\s*=\s*(true|false)")?;
        for caps in re.captures_iter(code) {
            if let Some(var) = caps.get(1) {
                let var_name = var.as_str();
                self.types.insert(var_name.to_string(), JSType::Boolean);

                result.type_info.push(TypeInfo {
                    variable: var_name.to_string(),
                    inferred_type: JSType::Boolean,
                    confidence: 1.0,
                    evidence: vec!["boolean literal".to_string()],
                });
            }
        }

        // 数组初始化
        let re = Regex::new(r"(?:var|let|const)\s+(\w+)\s*=\s*\[.*?\]")?;
        for caps in re.captures_iter(code) {
            if let Some(var) = caps.get(1) {
                let var_name = var.as_str();
                self.types.insert(var_name.to_string(), JSType::Array);

                result.type_info.push(TypeInfo {
                    variable: var_name.to_string(),
                    inferred_type: JSType::Array,
                    confidence: 1.0,
                    evidence: vec!["array literal".to_string()],
                });
            }
        }

        // 对象初始化
        let re = Regex::new(r"(?:var|let|const)\s+(\w+)\s*=\s*\{.*?\}")?;
        for caps in re.captures_iter(code) {
            if let Some(var) = caps.get(1) {
                let var_name = var.as_str();
                self.types.insert(var_name.to_string(), JSType::Object);

                result.type_info.push(TypeInfo {
                    variable: var_name.to_string(),
                    inferred_type: JSType::Object,
                    confidence: 1.0,
                    evidence: vec!["object literal".to_string()],
                });
            }
        }

        Ok(())
    }

    /// 推断操作类型
    fn infer_operation_types(
        &mut self,
        code: &str,
        result: &mut TypeInferenceResult,
    ) -> Result<()> {
        // 字符串操作 + 结果是字符串
        let re = Regex::new(r#"(\w+)\s*\+\s*(["'].+?["'])"#)?;
        for caps in re.captures_iter(code) {
            if let Some(var) = caps.get(1) {
                let var_name = var.as_str();
                let current_type = self.types.get(var_name).cloned().unwrap_or(JSType::Unknown);

                // 更新为 string 或 union
                match current_type {
                    JSType::Unknown => {
                        self.types.insert(var_name.to_string(), JSType::String);
                    }
                    JSType::Number | JSType::Boolean => {
                        self.types.insert(
                            var_name.to_string(),
                            JSType::Union(vec![Box::new(JSType::String), Box::new(current_type)]),
                        );
                    }
                    _ => {}
                }
            }
        }

        // 比较操作 -> boolean
        let re = Regex::new(r#"(\w+)\s*(?:===|!==|==|!=|<|>|<=|>=)\s*(.+?)(?:;|\)|])"#)?;
        for caps in re.captures_iter(code) {
            if let Some(var) = caps.get(1) {
                // 比较结果的类型总是布尔值
                // 这会影响使用比较结果的变量
            }
        }

        Ok(())
    }

    /// 推断函数类型
    fn infer_function_types(&mut self, code: &str, result: &mut TypeInferenceResult) -> Result<()> {
        // 匹配函数定义
        let re = Regex::new(r"function\s+(\w+)\s*\(([^)]*)\)\s*\{([^}]+)\}")?;

        for caps in re.captures_iter(code) {
            if let (Some(name), Some(params)) = (caps.get(1), caps.get(2)) {
                let func_name = name.as_str();
                let param_list: Vec<String> = params
                    .as_str()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();

                // 创建参数类型列表（初始都是 unknown）
                let param_types: Vec<(String, JSType)> = param_list
                    .iter()
                    .map(|p| (p.clone(), JSType::Unknown))
                    .collect();

                let signature = FunctionSignature {
                    name: func_name.to_string(),
                    parameters: param_types,
                    return_type: JSType::Unknown,
                };

                self.function_signatures
                    .insert(func_name.to_string(), signature);
                result.function_signatures.push(func_name.to_string());
            }
        }

        Ok(())
    }

    /// 推断返回类型
    fn infer_return_types(&mut self, code: &str, result: &mut TypeInferenceResult) -> Result<()> {
        // 匹配 return 语句
        let re = Regex::new(r"return\s+([^;]+);")?;

        for caps in re.captures_iter(code) {
            if let Some(return_val) = caps.get(1) {
                let val = return_val.as_str();

                // 推断返回类型
                let return_type = if val.contains('"') || val.contains('\'') {
                    JSType::String
                } else if val.chars().all(|c| c.is_numeric() || c == '.' || c == '-') {
                    JSType::Number
                } else if val == "true" || val == "false" {
                    JSType::Boolean
                } else {
                    JSType::Unknown
                };

                result.return_types.push((val.to_string(), return_type));
            }
        }

        Ok(())
    }

    /// 类型优化
    fn optimize_types(&self, result: &mut TypeInferenceResult) -> Result<()> {
        // 移除重复的类型信息
        // 合并相同变量的多个类型推断
        let mut seen = std::collections::HashSet::new();
        result.type_info.retain(|t| seen.insert(t.variable.clone()));

        Ok(())
    }

    /// 创建内置函数签名
    fn create_builtin_signatures() -> HashMap<String, FunctionSignature> {
        let mut sigs = HashMap::new();

        // String 函数
        sigs.insert(
            "String.fromCharCode".to_string(),
            FunctionSignature {
                name: "String.fromCharCode".to_string(),
                parameters: vec![("...codes".to_string(), JSType::Number)],
                return_type: JSType::String,
            },
        );

        // Array 函数
        sigs.insert(
            "Array.isArray".to_string(),
            FunctionSignature {
                name: "Array.isArray".to_string(),
                parameters: vec![("value".to_string(), JSType::Unknown)],
                return_type: JSType::Boolean,
            },
        );

        // Object 函数
        sigs.insert(
            "Object.keys".to_string(),
            FunctionSignature {
                name: "Object.keys".to_string(),
                parameters: vec![("obj".to_string(), JSType::Object)],
                return_type: JSType::Array,
            },
        );

        sigs
    }
}

impl Default for TypeInferencer {
    fn default() -> Self {
        Self::new()
    }
}

/// 推断结果
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeInferenceResult {
    pub type_info: Vec<TypeInfo>,
    pub function_signatures: Vec<String>,
    pub return_types: Vec<(String, JSType)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_type_inference() {
        let code = r#"var str = "hello";"#;
        let mut inferencer = TypeInferencer::new();
        let result = inferencer.infer(code).unwrap();

        assert!(result
            .type_info
            .iter()
            .any(|t| t.variable == "str" && t.inferred_type == JSType::String));
    }

    #[test]
    fn test_number_type_inference() {
        let code = r#"var num = 42;"#;
        let mut inferencer = TypeInferencer::new();
        let result = inferencer.infer(code).unwrap();

        assert!(result
            .type_info
            .iter()
            .any(|t| t.variable == "num" && t.inferred_type == JSType::Number));
    }

    #[test]
    fn test_array_type_inference() {
        let code = r#"var arr = [1, 2, 3];"#;
        let mut inferencer = TypeInferencer::new();
        let result = inferencer.infer(code).unwrap();

        assert!(result
            .type_info
            .iter()
            .any(|t| t.variable == "arr" && t.inferred_type == JSType::Array));
    }

    #[test]
    fn test_function_signature_extraction() {
        let code = r#"
            function decode(str, key) {
                return str + key;
            }
        "#;
        let mut inferencer = TypeInferencer::new();
        let result = inferencer.infer(code).unwrap();

        assert!(result.function_signatures.contains(&"decode".to_string()));
    }

    #[test]
    fn test_return_type_inference() {
        let code = r#"
            function getValue() {
                return "hello";
            }
        "#;
        let mut inferencer = TypeInferencer::new();
        let result = inferencer.infer(code).unwrap();

        assert!(!result.return_types.is_empty());
    }
}
