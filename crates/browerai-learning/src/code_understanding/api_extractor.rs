//! API 提取器 - 识别导出的公共接口

use crate::code_understanding::{ApiInfo, ParamInfo};
use anyhow::Result;
use regex::Regex;

/// API 提取器
pub struct ApiExtractor {
    function_sig_pattern: Regex,
    param_pattern: Regex,
    comment_pattern: Regex,
}

impl ApiExtractor {
    pub fn new() -> Self {
        Self {
            function_sig_pattern: Regex::new(
                r#"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)|(\w+)\s*:\s*(?:\(([^)]*)\)\s*=>|function)"#,
            )
            .unwrap(),
            param_pattern: Regex::new(r#"(\w+)\s*(?::|,|=)"#).unwrap(),
            comment_pattern: Regex::new(r#"/\*\*(.*?)\*\/|//\s*(.*)"#).unwrap(),
        }
    }

    /// 从代码中提取 API
    pub fn extract(&self, code: &str) -> Result<Vec<ApiInfo>> {
        let mut apis = Vec::new();

        // 查找所有导出的函数
        let export_func_pattern =
            Regex::new(r#"export\s+(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*(?:\{|:|=>)"#)
                .unwrap();

        for cap in export_func_pattern.captures_iter(code) {
            if let Some(func_name) = cap.get(1) {
                let name = func_name.as_str().to_string();
                let params_str = cap.get(2).map(|m| m.as_str()).unwrap_or("");

                let params = self.parse_params(params_str);
                let signature = format!("{}({})", name, params_str);
                let description = self.infer_description(&name);

                apis.push(ApiInfo {
                    name: name.clone(),
                    signature,
                    description,
                    params,
                    return_type: "any".to_string(),
                    examples: Vec::new(),
                });
            }
        }

        // 如果没有找到导出函数，查找所有顶级函数
        if apis.is_empty() {
            apis = self.extract_all_functions(code)?;
        }

        Ok(apis)
    }

    fn parse_params(&self, params_str: &str) -> Vec<ParamInfo> {
        let mut params = Vec::new();

        for param in params_str.split(',') {
            let param = param.trim();
            if !param.is_empty() {
                // 简单解析参数名
                let parts: Vec<&str> = param.split(':').collect();
                let name = parts[0].trim().to_string();
                let typ = if parts.len() > 1 {
                    parts[1].trim().to_string()
                } else {
                    "any".to_string()
                };

                params.push(ParamInfo {
                    name,
                    typ,
                    description: String::new(),
                });
            }
        }

        params
    }

    fn extract_all_functions(&self, code: &str) -> Result<Vec<ApiInfo>> {
        let mut apis = Vec::new();
        let func_pattern = Regex::new(r#"(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)"#).unwrap();

        for cap in func_pattern.captures_iter(code) {
            if let Some(func_name) = cap.get(1) {
                let name = func_name.as_str().to_string();
                let params_str = cap.get(2).map(|m| m.as_str()).unwrap_or("");

                let params = self.parse_params(params_str);
                let signature = format!("{}({})", name, params_str);
                let description = self.infer_description(&name);

                apis.push(ApiInfo {
                    name: name.clone(),
                    signature,
                    description,
                    params,
                    return_type: "any".to_string(),
                    examples: Vec::new(),
                });
            }
        }

        Ok(apis)
    }

    fn infer_description(&self, func_name: &str) -> String {
        let name_lower = func_name.to_lowercase();

        if name_lower.starts_with("is_") || name_lower.starts_with("is") {
            "检查或验证某个条件".to_string()
        } else if name_lower.starts_with("get") {
            "获取或返回某个值".to_string()
        } else if name_lower.starts_with("set") {
            "设置或更新某个值".to_string()
        } else if name_lower.starts_with("create") {
            "创建新的实例或对象".to_string()
        } else if name_lower.starts_with("parse") {
            "解析或转换数据".to_string()
        } else if name_lower.starts_with("format") {
            "格式化或美化数据".to_string()
        } else if name_lower.starts_with("validate") {
            "验证数据的有效性".to_string()
        } else if name_lower.starts_with("handle") {
            "处理特定事件或操作".to_string()
        } else if name_lower.starts_with("compute") || name_lower.starts_with("calculate") {
            "计算或推导结果".to_string()
        } else {
            format!("执行 {} 操作", func_name)
        }
    }
}

impl Default for ApiExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_api() -> Result<()> {
        let code = r#"
            export function add(a, b) {
                return a + b;
            }
            
            export function format(date) {
                return date.toString();
            }
        "#;

        let extractor = ApiExtractor::new();
        let apis = extractor.extract(code)?;
        assert_eq!(apis.len(), 2);
        Ok(())
    }

    #[test]
    fn test_parse_params() {
        let extractor = ApiExtractor::new();
        let params = extractor.parse_params("name: string, age: number");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name, "name");
    }
}
