/// 工作流提取引擎
///
/// 从 V8 执行追踪中识别高层工作流程
/// 例如：
/// - "用户点击Add to Cart" → 调用 addToCart() → 更新购物车显示
/// - "用户填表并提交" → 验证表单 → 发送请求 → 显示确认
use crate::v8_tracer::{ExecutionTrace, OperationChain, UserEvent};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 识别的工作流程
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Workflow {
    /// 工作流名称，自动推断（AddToCart, Checkout 等）
    pub name: String,

    /// 触发这个工作流的用户交互
    pub trigger: UserEvent,

    /// 工作流包含的所有操作
    pub operations: OperationChain,

    /// 关键的函数调用
    pub key_functions: Vec<String>,

    /// 工作流的复杂度评分（0-10）
    pub complexity_score: f64,

    /// 这个工作流的重要性评分（0-10）
    /// 基于：调用频率、修改 DOM 数量、涉及网络请求
    pub importance_score: f64,
}

/// 工作流识别的结果集
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkflowExtractionResult {
    /// 发现的所有工作流
    pub workflows: Vec<Workflow>,

    /// 覆盖的用户交互数
    pub total_user_interactions: usize,

    /// 覆盖的函数调用数
    pub total_function_calls: usize,

    /// 工作流识别的覆盖率（0-1）
    pub coverage_ratio: f64,
}

/// 工作流识别引擎
pub struct WorkflowExtractor;

impl WorkflowExtractor {
    /// 从执行追踪中提取所有工作流
    pub fn extract_workflows(traces: &ExecutionTrace) -> Result<WorkflowExtractionResult> {
        log::info!("开始工作流识别...");

        let mut workflows = vec![];
        let mut covered_operations_count = 0;

        // 对每个用户交互事件，提取关联的工作流
        for user_event in &traces.user_events {
            log::debug!(
                "  识别工作流: {} @ {:?}",
                user_event.event_type,
                user_event.target_element
            );

            // 获取该用户事件触发的操作链
            let operation_chain = traces.get_operation_chain(user_event);

            // 提取该操作链的关键函数
            let key_functions = Self::extract_key_functions(&operation_chain);

            // 计算复杂度
            let complexity_score = Self::calculate_complexity(&operation_chain);

            // 计算重要性
            let importance_score = Self::calculate_importance(&operation_chain, &key_functions);

            // 推断工作流名称
            let workflow_name = Self::infer_workflow_name(user_event, &key_functions);

            covered_operations_count += operation_chain.total_operations();

            let workflow = Workflow {
                name: workflow_name,
                trigger: user_event.clone(),
                operations: operation_chain,
                key_functions,
                complexity_score,
                importance_score,
            };

            workflows.push(workflow);
        }

        // 去重：如果两个工作流的函数调用链完全相同，合并它们
        workflows = Self::deduplicate_workflows(workflows);

        let total_ops = traces.function_calls.len() + traces.dom_operations.len();
        let coverage = if total_ops > 0 {
            (covered_operations_count as f64) / (total_ops as f64)
        } else {
            1.0
        };

        log::info!(
            "✓ 工作流识别完成: {} 个工作流, 覆盖率 {:.1}%",
            workflows.len(),
            coverage * 100.0
        );

        Ok(WorkflowExtractionResult {
            workflows,
            total_user_interactions: traces.user_events.len(),
            total_function_calls: traces.function_calls.len(),
            coverage_ratio: coverage,
        })
    }

    /// 从操作链中提取关键函数
    ///
    /// 关键函数的定义：
    /// - 在调用链中出现多次的函数
    /// - 调用深度较浅的函数（直接调用）
    /// - 造成 DOM 改动的函数
    fn extract_key_functions(chain: &OperationChain) -> Vec<String> {
        let mut function_freq: HashMap<String, usize> = HashMap::new();
        let mut function_depth: HashMap<String, usize> = HashMap::new();

        // 统计函数频率和深度
        for call in &chain.function_calls {
            *function_freq.entry(call.function_name.clone()).or_insert(0) += 1;

            function_depth
                .entry(call.function_name.clone())
                .and_modify(|d| {
                    if call.call_depth < *d {
                        *d = call.call_depth;
                    }
                })
                .or_insert(call.call_depth);
        }

        // 评分：频率 + (10 - 深度)
        let mut scored: Vec<_> = function_freq
            .iter()
            .map(|(name, freq)| {
                let depth = function_depth[name];
                let score = (*freq as f64) + (10.0 - depth as f64) * 2.0;
                (name.clone(), score)
            })
            .collect();

        // 按分数排序，取前 5 个
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored
            .iter()
            .take(5)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// 计算工作流的复杂度（0-10）
    fn calculate_complexity(chain: &OperationChain) -> f64 {
        let factors: f64 = [
            chain.max_call_depth() as f64 / 2.0,           // 调用深度
            (chain.function_calls.len() as f64).min(10.0), // 函数数量（最多贡献10分）
            (chain.dom_operations.len() as f64).min(5.0),  // DOM操作数（最多贡献5分）
        ]
        .iter()
        .sum();

        (factors / 10.0).min(10.0)
    }

    /// 计算工作流的重要性（0-10）
    ///
    /// 基于：
    /// - 函数数量（更多函数 = 更重要）
    /// - DOM 操作（更多改动 = 更重要）
    /// - 关键函数数量
    fn calculate_importance(chain: &OperationChain, key_functions: &[String]) -> f64 {
        let func_factor = (chain.function_calls.len() as f64 / 5.0).min(5.0);
        let dom_factor = (chain.dom_operations.len() as f64 / 2.0).min(3.0);
        let key_factor = (key_functions.len() as f64 / 3.0).min(2.0);

        (func_factor + dom_factor + key_factor).min(10.0)
    }

    /// 推断工作流的名称
    ///
    /// 通过分析：
    /// - 触发事件的元素（按钮文字等）
    /// - 关键函数的名称
    /// - 涉及的 DOM 元素
    fn infer_workflow_name(user_event: &UserEvent, key_functions: &[String]) -> String {
        // 策略 1：从元素 ID 或 class 推断
        if let Some(selector) = &user_event.selector {
            let selector_lower = selector.to_lowercase();
            if selector_lower.contains("add") && selector_lower.contains("cart") {
                return "AddToCart".to_string();
            }
            if selector_lower.contains("add-to-cart") || selector_lower.contains("addtocart") {
                return "AddToCart".to_string();
            }
            if selector_lower.contains("checkout") {
                return "Checkout".to_string();
            }
            if selector_lower.contains("login") {
                return "Login".to_string();
            }
            if selector_lower.contains("search") {
                return "Search".to_string();
            }
        }

        // 策略 2：从关键函数名推断
        if !key_functions.is_empty() {
            let first_func = &key_functions[0];

            if first_func.contains("cart") {
                return "CartOperation".to_string();
            }
            if first_func.contains("checkout") || first_func.contains("pay") {
                return "Checkout".to_string();
            }
            if first_func.contains("login") || first_func.contains("auth") {
                return "Authentication".to_string();
            }
            if first_func.contains("search") || first_func.contains("query") {
                return "Search".to_string();
            }
            if first_func.contains("filter") {
                return "Filter".to_string();
            }
        }

        // 默认：按事件类型命名
        match user_event.event_type.as_str() {
            "click" => "Click".to_string(),
            "submit" => "Submit".to_string(),
            "change" => "Change".to_string(),
            "input" => "Input".to_string(),
            _ => "Interaction".to_string(),
        }
    }

    /// 去重：合并相同的工作流
    fn deduplicate_workflows(workflows: Vec<Workflow>) -> Vec<Workflow> {
        // 简单实现：如果两个工作流的关键函数完全相同，认为是重复的
        // （更复杂的实现可以使用相似度检测）

        let mut unique_workflows = vec![];
        let mut seen_function_sets: Vec<Vec<String>> = vec![];

        for workflow in workflows {
            let mut func_set = workflow.key_functions.clone();
            func_set.sort();

            // 检查是否已经见过这个函数集合
            if !seen_function_sets.contains(&func_set) {
                seen_function_sets.push(func_set);
                unique_workflows.push(workflow);
            }
        }

        unique_workflows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v8_tracer::{CallRecord, DOMOperation};

    #[test]
    fn test_workflow_extraction() {
        let mut trace = ExecutionTrace::new();

        // 模拟一个点击事件，带有明确的选择器
        trace.user_events.push(UserEvent {
            event_type: "click".to_string(),
            target_element: "button".to_string(),
            selector: Some("add-to-cart".to_string()), // 这会触发工作流名称推断
            timestamp_ms: 100,
            triggered_operations: 0,
        });

        // 模拟后续的函数调用
        trace.function_calls.push(CallRecord {
            function_name: "addToCart".to_string(),
            arguments: vec!["productId".to_string()],
            return_type: "Promise".to_string(),
            timestamp_ms: 150,
            context_object: None,
            call_depth: 1,
        });

        // 模拟 DOM 操作
        trace.dom_operations.push(DOMOperation {
            operation_type: "setInnerHTML".to_string(),
            target_tag: "div".to_string(),
            target_id: Some("cart-count".to_string()),
            target_class: None,
            details: "更新购物车数量".to_string(),
            timestamp_ms: 200,
        });

        let result = WorkflowExtractor::extract_workflows(&trace).unwrap();
        assert_eq!(result.workflows.len(), 1);
        // 应该识别为 AddToCart，因为选择器包含 "add-to-cart"
        assert_eq!(result.workflows[0].name, "AddToCart");
    }

    #[test]
    fn test_key_function_extraction() {
        let chain = OperationChain {
            trigger: UserEvent {
                event_type: "click".to_string(),
                target_element: "button".to_string(),
                selector: None,
                timestamp_ms: 0,
                triggered_operations: 0,
            },
            function_calls: vec![
                CallRecord {
                    function_name: "handleClick".to_string(),
                    arguments: vec![],
                    return_type: "void".to_string(),
                    timestamp_ms: 0,
                    context_object: None,
                    call_depth: 1,
                },
                CallRecord {
                    function_name: "addToCart".to_string(),
                    arguments: vec![],
                    return_type: "Promise".to_string(),
                    timestamp_ms: 10,
                    context_object: None,
                    call_depth: 2,
                },
                CallRecord {
                    function_name: "updateUI".to_string(),
                    arguments: vec![],
                    return_type: "void".to_string(),
                    timestamp_ms: 20,
                    context_object: None,
                    call_depth: 2,
                },
            ],
            dom_operations: vec![],
        };

        let key_funcs = WorkflowExtractor::extract_key_functions(&chain);
        assert!(!key_funcs.is_empty());
        assert!(key_funcs.contains(&"addToCart".to_string()));
    }
}
