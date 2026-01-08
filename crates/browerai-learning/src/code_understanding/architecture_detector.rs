//! 架构检测器 - 识别代码的架构模式和组织方式

use crate::code_understanding::{ArchitectureInfo, ArchitecturePattern};
use anyhow::Result;

/// 架构检测器
pub struct ArchitectureDetector;

impl ArchitectureDetector {
    pub fn new() -> Self {
        Self
    }

    /// 检测代码的架构模式
    pub fn detect(&self, code: &str) -> Result<ArchitectureInfo> {
        let pattern = self.detect_pattern(code);
        let characteristics = self.extract_characteristics(code);
        let description = self.describe_pattern(&pattern, &characteristics);

        Ok(ArchitectureInfo {
            pattern,
            characteristics,
            description,
        })
    }

    fn detect_pattern(&self, code: &str) -> ArchitecturePattern {
        // 检查导入导出
        let has_es6_modules = code.contains("export ") || code.contains("import ");
        let has_commonjs = code.contains("module.exports") || code.contains("require(");
        let _has_amd = code.contains("define(") || code.contains("require([");

        // 检查类和函数
        let has_classes = code.contains("class ");
        let _has_constructors = code.matches("constructor(").count() > 0;

        // 检查 MVC/MVVM 相关模式
        let has_controller = code.contains("Controller") || code.contains("control");
        let has_model = code.contains("Model") || code.contains("model");
        let has_view = code.contains("View") || code.contains("view") || code.contains("render");
        let has_vm = code.contains("ViewModel") || code.contains("viewModel");

        // 检查插件模式
        let has_plugin = code.contains("Plugin")
            || code.contains("plugin")
            || code.contains("use(")
            || code.contains(".install");

        // 检查库/工具集特征
        let has_utility_functions = code.matches("function ").count() > 20;
        let is_single_file = !has_es6_modules && !has_commonjs;

        // 确定模式
        if has_classes && has_vm {
            ArchitecturePattern::MVVM
        } else if has_classes && has_controller && has_view && has_model {
            ArchitecturePattern::MVC
        } else if has_plugin {
            ArchitecturePattern::Plugin
        } else if has_es6_modules || has_commonjs {
            if has_utility_functions {
                ArchitecturePattern::Library
            } else {
                ArchitecturePattern::Modular
            }
        } else if is_single_file && has_utility_functions {
            ArchitecturePattern::Library
        } else if is_single_file {
            ArchitecturePattern::Monolithic
        } else {
            ArchitecturePattern::Hybrid
        }
    }

    fn extract_characteristics(&self, code: &str) -> Vec<String> {
        let mut chars = Vec::new();

        // 模块化特征
        if code.contains("export ") || code.contains("import ") {
            chars.push("ES6 Modules".to_string());
        }
        if code.contains("module.exports") || code.contains("require(") {
            chars.push("CommonJS".to_string());
        }

        // OOP 特征
        if code.contains("class ") {
            chars.push("Class-based".to_string());
        }
        if code.contains("prototype") {
            chars.push("Prototype-based".to_string());
        }

        // 函数式编程特征
        if code.matches(" => ").count() > 10 {
            chars.push("Functional Programming".to_string());
        }
        if code.contains("reduce(") || code.contains("map(") || code.contains("filter(") {
            chars.push("FP Utilities".to_string());
        }

        // 异步特征
        if code.contains("async ") && code.contains("await ") {
            chars.push("Async/Await".to_string());
        }
        if code.contains("Promise") {
            chars.push("Promise-based".to_string());
        }
        if code.contains(".then(") {
            chars.push("Promise Chains".to_string());
        }

        // 事件驱动
        if code.contains(".on(") || code.contains(".off(") || code.contains(".emit(") {
            chars.push("Event-driven".to_string());
        }

        // 状态管理
        if code.contains("state") && code.contains("setState") {
            chars.push("State Management".to_string());
        }

        // 配置驱动
        if code.contains("config") || code.contains("Config") {
            chars.push("Configuration-driven".to_string());
        }

        chars
    }

    fn describe_pattern(&self, pattern: &ArchitecturePattern, chars: &[String]) -> String {
        let pattern_desc = match pattern {
            ArchitecturePattern::Monolithic => "单文件单体架构 - 所有代码在一个文件中",
            ArchitecturePattern::Modular => "模块化架构 - 代码分为多个独立模块",
            ArchitecturePattern::MVC => "MVC 架构 - 将应用分为 Model、View、Controller",
            ArchitecturePattern::MVVM => "MVVM 架构 - 使用 ViewModel 解耦 View 和 Model",
            ArchitecturePattern::Plugin => "插件架构 - 支持动态加载和扩展",
            ArchitecturePattern::Library => "库/工具集 - 提供可复用的函数和工具",
            ArchitecturePattern::Hybrid => "混合架构 - 结合多种模式",
            ArchitecturePattern::Unknown => "未知架构",
        };

        format!("{}。特征：{}", pattern_desc, chars.join(", "))
    }
}

impl Default for ArchitectureDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_library() -> Result<()> {
        let code = r#"
            export function add(a, b) { return a + b; }
            export function subtract(a, b) { return a - b; }
            export const VERSION = "1.0.0";
        "#;

        let detector = ArchitectureDetector::new();
        let result = detector.detect(code)?;
        assert_eq!(result.pattern, ArchitecturePattern::Library);
        Ok(())
    }

    #[test]
    fn test_detect_monolithic() -> Result<()> {
        let code = r#"
            function main() {
                var x = 10;
                console.log(x);
            }
            main();
        "#;

        let detector = ArchitectureDetector::new();
        let result = detector.detect(code)?;
        assert_eq!(result.pattern, ArchitecturePattern::Monolithic);
        Ok(())
    }
}
