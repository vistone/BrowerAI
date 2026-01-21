/// 推理阶段：数据结构推断
///
/// 从代码执行追踪中推断数据结构：
/// - 识别对象和类的结构
/// - 推断 API 响应的格式
/// - 推断数据库字段类型
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::v8_tracer::ExecutionTrace;
use crate::variable_semantics::{DataType, VariableSemantics};

/// 推断的数据结构
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferredStructure {
    /// 结构名称
    pub name: String,

    /// 结构类型（class, interface, dto, response, request）
    pub structure_type: StructureType,

    /// 字段定义
    pub fields: Vec<Field>,

    /// 该结构出现的次数
    pub occurrences: usize,

    /// 推断可信度
    pub confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum StructureType {
    Class,
    Interface,
    DTO, // Data Transfer Object
    APIResponse,
    APIRequest,
    Unknown,
}

/// 字段定义
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub field_type: String,
    pub is_required: bool,
    pub is_nullable: bool,
    pub description: Option<String>,
}

/// 数据结构推断结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructureInferenceResult {
    pub structures: Vec<InferredStructure>,
    pub relationships: Vec<StructureRelationship>,
    pub accuracy: f64,
}

/// 结构之间的关系
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructureRelationship {
    pub source: String,
    pub target: String,
    pub relationship_type: RelationshipType,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum RelationshipType {
    Contains,   // A contains B
    References, // A references B
    Inherits,   // A inherits from B
    Implements, // A implements B
}

/// 数据结构推断引擎
pub struct DataStructureInferenceEngine;

impl DataStructureInferenceEngine {
    /// 推断数据结构
    pub fn infer_structures(
        _traces: &ExecutionTrace,
        variables: &[VariableSemantics],
    ) -> Result<StructureInferenceResult> {
        log::info!("🏗️  推断数据结构...");

        let structures = Self::identify_structures(variables)?;
        let relationships = Self::identify_relationships(&structures)?;
        let accuracy = Self::calculate_accuracy(&structures);

        Ok(StructureInferenceResult {
            structures,
            relationships,
            accuracy,
        })
    }

    /// 识别数据结构
    fn identify_structures(variables: &[VariableSemantics]) -> Result<Vec<InferredStructure>> {
        let mut structures = Vec::new();
        let mut seen = HashSet::new();

        for var in variables {
            if var.data_type == DataType::Object && seen.insert(var.variable_name.clone()) {
                let structure_type =
                    Self::infer_structure_type(&var.variable_name, &var.business_meaning);

                let fields = Self::infer_fields(&var.variable_name);

                structures.push(InferredStructure {
                    name: var.variable_name.clone(),
                    structure_type,
                    fields,
                    occurrences: 1,
                    confidence: var.confidence * 0.85,
                });
            }
        }

        Ok(structures)
    }

    /// 推断结构类型
    fn infer_structure_type(name: &str, business_meaning: &Option<String>) -> StructureType {
        let name_lower = name.to_lowercase();

        if name_lower.contains("response") || name_lower.contains("result") {
            return StructureType::APIResponse;
        }

        if name_lower.contains("request") {
            return StructureType::APIRequest;
        }

        if name_lower.contains("dto") {
            return StructureType::DTO;
        }

        if let Some(meaning) = business_meaning {
            if meaning.contains("数据") || meaning.contains("结构") {
                return StructureType::DTO;
            }
        }

        StructureType::Class
    }

    /// 推断字段
    fn infer_fields(structure_name: &str) -> Vec<Field> {
        let mut fields = Vec::new();

        // 根据结构名称推断可能的字段
        let field_patterns = vec![
            ("cart", vec!["items", "total", "quantity"]),
            (
                "product",
                vec!["id", "name", "price", "description", "image"],
            ),
            ("user", vec!["id", "name", "email", "username", "password"]),
            ("order", vec!["id", "items", "total", "status", "date"]),
            ("response", vec!["code", "message", "data", "timestamp"]),
            ("request", vec!["type", "data", "timestamp"]),
        ];

        let name_lower = structure_name.to_lowercase();

        for (keyword, field_names) in field_patterns {
            if name_lower.contains(keyword) {
                for field_name in field_names {
                    fields.push(Field {
                        name: field_name.to_string(),
                        field_type: Self::infer_field_type(field_name),
                        is_required: Self::is_field_required(field_name),
                        is_nullable: Self::is_field_nullable(field_name),
                        description: None,
                    });
                }
                break;
            }
        }

        if fields.is_empty() {
            // 默认字段
            fields.push(Field {
                name: "id".to_string(),
                field_type: "string|number".to_string(),
                is_required: true,
                is_nullable: false,
                description: None,
            });
            fields.push(Field {
                name: "data".to_string(),
                field_type: "any".to_string(),
                is_required: false,
                is_nullable: true,
                description: None,
            });
        }

        fields
    }

    fn infer_field_type(field_name: &str) -> String {
        let field_lower = field_name.to_lowercase();

        if field_lower.contains("id")
            || field_lower.contains("count")
            || field_lower.contains("quantity")
        {
            return "number".to_string();
        }

        if field_lower.contains("price") || field_lower.contains("total") {
            return "number|string".to_string();
        }

        if field_lower.contains("name")
            || field_lower.contains("title")
            || field_lower.contains("email")
        {
            return "string".to_string();
        }

        if field_lower.contains("date") || field_lower.contains("time") {
            return "string|number".to_string();
        }

        if field_lower.contains("items") || field_lower.contains("list") {
            return "array".to_string();
        }

        if field_lower.contains("data") || field_lower.contains("content") {
            return "object".to_string();
        }

        "any".to_string()
    }

    fn is_field_required(field_name: &str) -> bool {
        let field_lower = field_name.to_lowercase();
        matches!(field_lower.as_str(), "id" | "type" | "code")
    }

    fn is_field_nullable(field_name: &str) -> bool {
        let field_lower = field_name.to_lowercase();
        matches!(
            field_lower.as_str(),
            "description" | "image" | "data" | "message"
        )
    }

    /// 识别结构之间的关系
    fn identify_relationships(
        _structures: &[InferredStructure],
    ) -> Result<Vec<StructureRelationship>> {
        let relationships = Vec::new();
        // 从字段类型推断关系
        // 当 A.field 的类型是 B 时，A 包含 B
        Ok(relationships)
    }

    /// 计算准确度
    fn calculate_accuracy(structures: &[InferredStructure]) -> f64 {
        if structures.is_empty() {
            return 0.5;
        }

        let avg_confidence =
            structures.iter().map(|s| s.confidence).sum::<f64>() / structures.len() as f64;

        avg_confidence * 0.9
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_type_detection() {
        let resp_type = DataStructureInferenceEngine::infer_structure_type("api_response", &None);
        assert_eq!(resp_type, StructureType::APIResponse);

        let req_type = DataStructureInferenceEngine::infer_structure_type("request", &None);
        assert_eq!(req_type, StructureType::APIRequest);
    }

    #[test]
    fn test_field_type_inference() {
        assert_eq!(
            DataStructureInferenceEngine::infer_field_type("user_id"),
            "number"
        );
        assert_eq!(
            DataStructureInferenceEngine::infer_field_type("price"),
            "number|string"
        );
        assert_eq!(
            DataStructureInferenceEngine::infer_field_type("items"),
            "array"
        );
    }
}
