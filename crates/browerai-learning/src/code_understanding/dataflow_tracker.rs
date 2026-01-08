//! 数据流追踪器 - 分析数据如何在系统间流动

use crate::code_understanding::{DataflowInfo, DataflowType, ModuleInfo};
use anyhow::Result;
use regex::Regex;

/// 数据流追踪器
pub struct DataflowTracker {
    call_pattern: Regex,
    event_pattern: Regex,
    state_pattern: Regex,
}

impl DataflowTracker {
    pub fn new() -> Self {
        Self {
            call_pattern: Regex::new(r#"(\w+)\s*\(\s*([^)]*)\s*\)"#).unwrap(),
            event_pattern: Regex::new(
                r#"\.(?:on|off|emit|dispatch|subscribe)\s*\(\s*["']?(\w+)["']?"#,
            )
            .unwrap(),
            state_pattern: Regex::new(r#"setState\s*\(\s*\{([^}]+)\}"#).unwrap(),
        }
    }

    /// 分析代码中的数据流
    pub fn analyze(&self, code: &str, modules: &[ModuleInfo]) -> Result<Vec<DataflowInfo>> {
        let mut flows = Vec::new();

        // 分析函数调用流
        flows.extend(self.analyze_function_calls(code, modules));

        // 分析事件流
        flows.extend(self.analyze_event_flows(code, modules));

        // 分析状态流
        flows.extend(self.analyze_state_flows(code, modules));

        // 分析数据传递
        flows.extend(self.analyze_data_passing(code, modules));

        Ok(flows)
    }

    fn analyze_function_calls(&self, code: &str, _modules: &[ModuleInfo]) -> Vec<DataflowInfo> {
        let mut flows = Vec::new();

        // 识别主要函数调用
        for cap in self.call_pattern.captures_iter(code) {
            if let Some(func_name) = cap.get(1) {
                let name = func_name.as_str();

                // 过滤内置和库函数
                if !self.is_builtin_function(name) {
                    if let Some(args) = cap.get(2) {
                        flows.push(DataflowInfo {
                            source: "caller".to_string(),
                            target: name.to_string(),
                            description: format!("调用函数 {} 传递参数: {}", name, args.as_str()),
                            flow_type: DataflowType::FunctionCall,
                        });
                    }
                }
            }
        }

        flows
    }

    fn analyze_event_flows(&self, code: &str, _modules: &[ModuleInfo]) -> Vec<DataflowInfo> {
        let mut flows = Vec::new();

        // 识别事件处理
        for cap in self.event_pattern.captures_iter(code) {
            if let Some(event_name) = cap.get(1) {
                flows.push(DataflowInfo {
                    source: "event".to_string(),
                    target: event_name.as_str().to_string(),
                    description: format!("事件 '{}' 的处理和传播", event_name.as_str()),
                    flow_type: DataflowType::EventPropagation,
                });
            }
        }

        flows
    }

    fn analyze_state_flows(&self, code: &str, _modules: &[ModuleInfo]) -> Vec<DataflowInfo> {
        let mut flows = Vec::new();

        // 识别状态管理
        if code.contains("setState") {
            for cap in self.state_pattern.captures_iter(code) {
                if let Some(state_changes) = cap.get(1) {
                    flows.push(DataflowInfo {
                        source: "state".to_string(),
                        target: "component".to_string(),
                        description: format!("状态更新: {}", state_changes.as_str()),
                        flow_type: DataflowType::StateManagement,
                    });
                }
            }
        }

        // 识别 Redux 风格的状态管理
        if code.contains("dispatch") {
            flows.push(DataflowInfo {
                source: "action".to_string(),
                target: "reducer".to_string(),
                description: "分发 action 到 reducer 处理".to_string(),
                flow_type: DataflowType::StateManagement,
            });
        }

        flows
    }

    fn analyze_data_passing(&self, _code: &str, modules: &[ModuleInfo]) -> Vec<DataflowInfo> {
        let mut flows = Vec::new();

        // 分析模块间的数据传递
        for module in modules {
            for dep in &module.dependencies {
                flows.push(DataflowInfo {
                    source: module.name.clone(),
                    target: dep.clone(),
                    description: format!("{} 依赖 {}", module.name, dep),
                    flow_type: DataflowType::DataPassing,
                });
            }

            // 分析导出的数据
            for export in &module.exports {
                flows.push(DataflowInfo {
                    source: module.name.clone(),
                    target: "external".to_string(),
                    description: format!("导出 {}", export),
                    flow_type: DataflowType::DataPassing,
                });
            }
        }

        flows
    }

    fn is_builtin_function(&self, name: &str) -> bool {
        matches!(
            name,
            "console"
                | "document"
                | "window"
                | "setTimeout"
                | "setInterval"
                | "fetch"
                | "JSON"
                | "Math"
                | "Array"
                | "String"
                | "Object"
                | "Promise"
                | "Map"
                | "Set"
                | "WeakMap"
                | "WeakSet"
                | "Symbol"
                | "Proxy"
                | "Reflect"
        )
    }
}

impl Default for DataflowTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_dataflow() -> Result<()> {
        let code = r#"
            import { foo } from './foo.js';
            
            function process(data) {
                const result = foo(data);
                return result;
            }
        "#;

        let tracker = DataflowTracker::new();
        let flows = tracker.analyze(code, &[])?;
        assert!(!flows.is_empty());
        Ok(())
    }
}
