/// 推理阶段：变量语义理解
///
/// 理解变量的含义和类型：
/// - 通过代码上下文推断变量类型
/// - 识别变量的业务含义（如：购物车商品、用户名等）
/// - 追踪变量的转换过程
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::v8_tracer::ExecutionTrace;
use crate::workflow_extractor::Workflow;

/// 变量语义
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VariableSemantics {
    /// 变量名
    pub variable_name: String,

    /// 推断的数据类型（string, number, object, array 等）
    pub data_type: DataType,

    /// 业务含义
    pub business_meaning: Option<String>,

    /// 变量的作用域（local, global, closure）
    pub scope: VariableScope,

    /// 该变量涉及的函数
    pub referenced_in_functions: Vec<String>,

    /// 可信度（0-1）
    pub confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DataType {
    String,
    Number,
    Boolean,
    Object,
    Array,
    Function,
    Null,
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum VariableScope {
    Local,
    Global,
    Closure,
    Parameter,
}

/// 变量间的依赖关系
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VariableDependency {
    pub source_var: String,
    pub target_var: String,
    pub dependency_type: DependencyType,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DependencyType {
    Assignment,     // target = source
    Transformation, // target = f(source)
    Composition,    // target = {source, ...}
    Usage,          // source is used to compute target
}

/// 推理结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceResult {
    pub variables: Vec<VariableSemantics>,
    pub dependencies: Vec<VariableDependency>,
    pub data_structures: Vec<DataStructureInference>,
    pub accuracy: f64,
}

/// 数据结构推断
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataStructureInference {
    pub name: String,
    pub inferred_type: String, // "class", "interface", "dict", etc
    pub fields: Vec<FieldInfo>,
    pub confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldInfo {
    pub field_name: String,
    pub field_type: String,
    pub is_required: bool,
}

/// 变量语义分析器
pub struct VariableSemanticsAnalyzer;

impl VariableSemanticsAnalyzer {
    /// 分析变量语义
    pub fn analyze_variables(
        traces: &ExecutionTrace,
        workflows: &[Workflow],
    ) -> Result<InferenceResult> {
        log::info!("📊 分析变量语义...");

        let mut variables = Self::extract_variables(traces, workflows)?;
        let dependencies = Self::infer_dependencies(traces, &variables)?;
        let data_structures = Self::infer_data_structures(traces, &variables)?;

        // 优化变量列表，删除冗余
        variables = Self::deduplicate_variables(variables);

        // 计算准确度
        let accuracy = Self::calculate_accuracy(&variables, &dependencies);

        Ok(InferenceResult {
            variables,
            dependencies,
            data_structures,
            accuracy,
        })
    }

    /// 提取变量信息
    fn extract_variables(
        traces: &ExecutionTrace,
        _workflows: &[Workflow],
    ) -> Result<Vec<VariableSemantics>> {
        let mut variables = HashMap::new();

        // 从状态变化中提取变量
        for state_change in &traces.state_changes {
            let var_name = &state_change.variable_name;

            // 推断数据类型
            let data_type = Self::infer_type(&state_change.new_value_type);

            // 推断业务含义
            let business_meaning =
                Self::infer_business_meaning(var_name, &state_change.new_value_type);

            let entry = variables
                .entry(var_name.clone())
                .or_insert_with(|| VariableSemantics {
                    variable_name: var_name.clone(),
                    data_type,
                    business_meaning,
                    scope: VariableScope::Global,
                    referenced_in_functions: vec![],
                    confidence: 0.0,
                });

            entry.confidence = (entry.confidence + 0.9).min(1.0);
        }

        // 从函数参数推断变量
        for call in &traces.function_calls {
            for (idx, arg) in call.arguments.iter().enumerate() {
                let var_name = format!("{}[arg{}]", call.function_name, idx);
                let data_type = Self::infer_type(arg);

                if !variables.contains_key(&var_name) {
                    variables.insert(
                        var_name.clone(),
                        VariableSemantics {
                            variable_name: var_name,
                            data_type,
                            business_meaning: None,
                            scope: VariableScope::Parameter,
                            referenced_in_functions: vec![call.function_name.clone()],
                            confidence: 0.7,
                        },
                    );
                }
            }
        }

        Ok(variables.into_values().collect())
    }

    /// 推断数据类型
    fn infer_type(value: &str) -> DataType {
        if value == "null" || value.is_empty() {
            return DataType::Null;
        }

        if value == "true" || value == "false" {
            return DataType::Boolean;
        }

        if value.parse::<f64>().is_ok() {
            return DataType::Number;
        }

        if value.starts_with('[') && value.ends_with(']') {
            return DataType::Array;
        }

        if value.starts_with('{') && value.ends_with('}') {
            return DataType::Object;
        }

        if value.contains("function") {
            return DataType::Function;
        }

        DataType::String
    }

    /// 推断业务含义
    fn infer_business_meaning(var_name: &str, value: &str) -> Option<String> {
        let var_lower = var_name.to_lowercase();

        let meanings = vec![
            ("cart", "购物车"),
            ("product", "商品"),
            ("price", "价格"),
            ("quantity", "数量"),
            ("total", "总额"),
            ("user", "用户"),
            ("id", "唯一标识"),
            ("name", "名称"),
            ("email", "邮箱"),
            ("status", "状态"),
            ("item", "项目"),
            ("list", "列表"),
            ("query", "查询"),
            ("result", "结果"),
            ("data", "数据"),
            ("error", "错误"),
            ("loading", "加载中"),
            ("visible", "可见"),
        ];

        for (keyword, meaning) in meanings {
            if var_lower.contains(keyword) {
                return Some(format!("{}(含关键词'{}')", meaning, keyword));
            }
        }

        // 通过值的形式推断
        if value.starts_with('[') {
            return Some("可能是数组/列表".to_string());
        }

        if value.starts_with('{') {
            return Some("可能是对象/结构".to_string());
        }

        None
    }

    /// 推断变量依赖关系
    fn infer_dependencies(
        _traces: &ExecutionTrace,
        variables: &[VariableSemantics],
    ) -> Result<Vec<VariableDependency>> {
        let mut dependencies = vec![];

        let var_names: HashSet<_> = variables.iter().map(|v| v.variable_name.clone()).collect();

        for var in variables {
            // 检查变量名称中是否包含其他变量的模式
            for other_var in &var_names {
                if other_var != &var.variable_name && var.variable_name.contains(other_var) {
                    dependencies.push(VariableDependency {
                        source_var: other_var.clone(),
                        target_var: var.variable_name.clone(),
                        dependency_type: DependencyType::Usage,
                    });
                }
            }
        }

        Ok(dependencies)
    }

    /// 推断数据结构
    fn infer_data_structures(
        _traces: &ExecutionTrace,
        variables: &[VariableSemantics],
    ) -> Result<Vec<DataStructureInference>> {
        let mut structures = vec![];

        for var in variables {
            if var.data_type == DataType::Object {
                // 这可能是一个类或接口
                structures.push(DataStructureInference {
                    name: var.variable_name.clone(),
                    inferred_type: "class".to_string(),
                    fields: vec![],
                    confidence: var.confidence * 0.8,
                });
            }
        }

        Ok(structures)
    }

    /// 删除冗余变量
    fn deduplicate_variables(mut variables: Vec<VariableSemantics>) -> Vec<VariableSemantics> {
        variables.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut seen = HashSet::new();
        variables.retain(|v| {
            let key = v.variable_name.clone();
            seen.insert(key)
        });

        variables
    }

    /// 计算准确度
    fn calculate_accuracy(
        variables: &[VariableSemantics],
        dependencies: &[VariableDependency],
    ) -> f64 {
        if variables.is_empty() {
            return 0.5;
        }

        let var_confidence: f64 =
            variables.iter().map(|v| v.confidence).sum::<f64>() / variables.len() as f64;
        let dep_factor = if dependencies.is_empty() { 0.5 } else { 0.8 };

        (var_confidence + dep_factor) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_inference() {
        assert_eq!(
            VariableSemanticsAnalyzer::infer_type("123"),
            DataType::Number
        );
        assert_eq!(VariableSemanticsAnalyzer::infer_type("[]"), DataType::Array);
        assert_eq!(
            VariableSemanticsAnalyzer::infer_type("{}"),
            DataType::Object
        );
        assert_eq!(
            VariableSemanticsAnalyzer::infer_type("hello"),
            DataType::String
        );
    }

    #[test]
    fn test_business_meaning() {
        let result = VariableSemanticsAnalyzer::infer_business_meaning("cart_items", "[]");
        assert!(result.is_some());
        assert!(result.unwrap().contains("购物车"));
    }
}
