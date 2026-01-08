//! 模块分析器 - 提取模块边界、导出、依赖关系

use crate::code_understanding::ModuleInfo;
use anyhow::Result;
use regex::Regex;

/// 模块分析器
pub struct ModuleAnalyzer {
    export_pattern: Regex,
    import_pattern: Regex,
    require_pattern: Regex,
    function_pattern: Regex,
    variable_pattern: Regex,
}

impl ModuleAnalyzer {
    pub fn new() -> Self {
        Self {
            export_pattern: Regex::new(
                r#"export\s+(?:default\s+)?(?:async\s+)?(?:function|const|let|var|class)\s+(\w+)"#,
            )
            .unwrap(),
            import_pattern: Regex::new(
                r#"import\s+(?:\{[^}]*\}|[\w\s,]+)\s+from\s+["']([^"']+)["']"#,
            )
            .unwrap(),
            require_pattern: Regex::new(r#"require\s*\(\s*["']([^"']+)["']\s*\)"#).unwrap(),
            function_pattern: Regex::new(
                r#"(?:async\s+)?(?:function|\w+\s*=\s*(?:async\s+)?)\w+\s*\("#,
            )
            .unwrap(),
            variable_pattern: Regex::new(r#"(?:const|let|var)\s+(\w+)\s*="#).unwrap(),
        }
    }

    /// 从代码中提取模块信息
    pub fn extract_modules(&self, code: &str) -> Result<Vec<ModuleInfo>> {
        let lines: Vec<&str> = code.lines().collect();
        let mut modules = Vec::new();

        // 策略1: 检查 ES6 模块/CommonJS 导出
        let exports = self.extract_exports(code);
        if !exports.is_empty() {
            for export_name in &exports {
                let module = self.create_module_from_export(code, export_name, &lines);
                modules.push(module);
            }
        }

        // 策略2: 检查明显的模块标记（如类、大函数等）
        if modules.is_empty() {
            modules = self.extract_implicit_modules(code, &lines);
        }

        // 如果仍然为空，将整个代码视为单个模块
        if modules.is_empty() {
            modules.push(ModuleInfo {
                name: "main".to_string(),
                responsibility: "主模块".to_string(),
                exports: vec![],
                dependencies: self.extract_dependencies(code),
                functions: self.extract_function_names(code),
                variables: self.extract_variable_names(code),
                size: code.lines().count(),
            });
        }

        Ok(modules)
    }

    fn extract_exports(&self, code: &str) -> Vec<String> {
        self.export_pattern
            .captures_iter(code)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect()
    }

    fn extract_dependencies(&self, code: &str) -> Vec<String> {
        let mut deps = Vec::new();

        // ES6 imports
        for cap in self.import_pattern.captures_iter(code) {
            if let Some(m) = cap.get(1) {
                deps.push(m.as_str().to_string());
            }
        }

        // CommonJS requires
        for cap in self.require_pattern.captures_iter(code) {
            if let Some(m) = cap.get(1) {
                deps.push(m.as_str().to_string());
            }
        }

        deps.sort();
        deps.dedup();
        deps
    }

    fn extract_function_names(&self, code: &str) -> Vec<String> {
        let func_pattern =
            Regex::new(r#"(?:async\s+)?function\s+(\w+)|(\w+)\s*=\s*(?:async\s+)?function"#)
                .unwrap();

        let mut functions = Vec::new();
        for cap in func_pattern.captures_iter(code) {
            if let Some(m) = cap.get(1) {
                functions.push(m.as_str().to_string());
            } else if let Some(m) = cap.get(2) {
                functions.push(m.as_str().to_string());
            }
        }

        functions.sort();
        functions.dedup();
        functions
    }

    fn extract_variable_names(&self, code: &str) -> Vec<String> {
        let mut vars = Vec::new();
        for cap in self.variable_pattern.captures_iter(code) {
            if let Some(m) = cap.get(1) {
                vars.push(m.as_str().to_string());
            }
        }

        vars.sort();
        vars.dedup();
        vars
    }

    fn create_module_from_export(
        &self,
        code: &str,
        export_name: &str,
        _lines: &[&str],
    ) -> ModuleInfo {
        let responsibility = self.infer_responsibility(export_name);
        let dependencies = self.extract_dependencies(code);

        ModuleInfo {
            name: export_name.to_string(),
            responsibility,
            exports: vec![export_name.to_string()],
            dependencies,
            functions: self.extract_function_names(code),
            variables: self.extract_variable_names(code),
            size: code.lines().count(),
        }
    }

    fn extract_implicit_modules(&self, code: &str, _lines: &[&str]) -> Vec<ModuleInfo> {
        let mut modules = Vec::new();

        // 按类分组
        let class_pattern = Regex::new(r#"class\s+(\w+)"#).unwrap();
        for cap in class_pattern.captures_iter(code) {
            if let Some(m) = cap.get(1) {
                let class_name = m.as_str().to_string();
                modules.push(ModuleInfo {
                    name: class_name.clone(),
                    responsibility: format!("{}类及其相关逻辑", class_name),
                    exports: vec![class_name],
                    dependencies: self.extract_dependencies(code),
                    functions: vec![],
                    variables: vec![],
                    size: code.lines().count(),
                });
            }
        }

        modules
    }

    fn infer_responsibility(&self, name: &str) -> String {
        let name_lower = name.to_lowercase();

        if name_lower.contains("controller") {
            "请求控制和业务逻辑处理".to_string()
        } else if name_lower.contains("service") {
            "业务服务和数据处理".to_string()
        } else if name_lower.contains("model") {
            "数据模型和结构定义".to_string()
        } else if name_lower.contains("view") {
            "视图和 UI 相关逻辑".to_string()
        } else if name_lower.contains("util") || name_lower.contains("helper") {
            "工具函数和辅助方法".to_string()
        } else if name_lower.contains("config") {
            "配置管理和初始化".to_string()
        } else if name_lower.contains("parser") || name_lower.contains("format") {
            "数据解析和格式转换".to_string()
        } else {
            format!("提供{}功能", name)
        }
    }
}

impl Default for ModuleAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_modules() -> Result<()> {
        let code = r#"
            export function add(a, b) { return a + b; }
            export function subtract(a, b) { return a - b; }
        "#;

        let analyzer = ModuleAnalyzer::new();
        let modules = analyzer.extract_modules(code)?;
        assert!(!modules.is_empty());
        Ok(())
    }

    #[test]
    fn test_extract_dependencies() {
        let code = r#"
            import { foo } from './foo.js';
            const bar = require('./bar.js');
        "#;

        let analyzer = ModuleAnalyzer::new();
        let deps = analyzer.extract_dependencies(code);
        assert_eq!(deps.len(), 2);
    }
}
