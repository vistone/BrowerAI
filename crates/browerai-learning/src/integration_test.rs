/// 完整集成测试模块
/// 验证学习-推理-生成的完整流程
#[allow(unused_imports)]
use crate::{
    BrowserAIClient, CallRecord, ClientConfig, ClientState, CompleteInferencePipeline,
    DOMOperation, DataStructureInferenceEngine, ExecutionTrace, ImprovedCodeGenerator,
    LearningQuality, UserEvent, VariableSemanticsAnalyzer, WorkflowExtractor,
};

/// 创建测试用的执行追踪数据
#[allow(dead_code)]
fn create_test_trace() -> ExecutionTrace {
    let mut trace = ExecutionTrace::new();

    // 添加函数调用
    trace.function_calls.push(CallRecord {
        function_name: "handleSearch".to_string(),
        arguments: vec!["query".to_string()],
        return_type: "void".to_string(),
        timestamp_ms: 100,
        context_object: None,
        call_depth: 0,
    });

    trace.function_calls.push(CallRecord {
        function_name: "fetchResults".to_string(),
        arguments: vec!["url".to_string()],
        return_type: "Promise<array>".to_string(),
        timestamp_ms: 150,
        context_object: None,
        call_depth: 1,
    });

    trace.function_calls.push(CallRecord {
        function_name: "renderResults".to_string(),
        arguments: vec!["data".to_string()],
        return_type: "void".to_string(),
        timestamp_ms: 200,
        context_object: None,
        call_depth: 1,
    });

    // 添加 DOM 操作
    trace.dom_operations.push(DOMOperation {
        operation_type: "appendChild".to_string(),
        target_tag: "div".to_string(),
        target_id: Some("results".to_string()),
        target_class: None,
        details: "Added result container".to_string(),
        timestamp_ms: 180,
    });

    // 添加用户事件
    trace.user_events.push(UserEvent {
        event_type: "click".to_string(),
        target_element: "button".to_string(),
        selector: Some(".search-btn".to_string()),
        timestamp_ms: 50,
        triggered_operations: 3,
    });

    trace.total_duration_ms = 300;
    trace.page_ready_ms = Some(50);

    trace
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_learning_to_generation_flow() {
        log::info!("开始测试完整流程...");

        let trace = create_test_trace();

        // Step 1: 提取工作流
        let workflows =
            WorkflowExtractor::extract_workflows(&trace).expect("Failed to extract workflows");

        assert!(
            !workflows.workflows.is_empty(),
            "Should extract at least one workflow"
        );
        log::info!("✓ 提取了 {} 个工作流", workflows.workflows.len());

        // Step 2: 评估学习质量
        let quality =
            LearningQuality::evaluate(&trace, &workflows).expect("Failed to evaluate quality");

        assert!(
            quality.overall_score > 0.0,
            "Quality score should be positive"
        );
        log::info!("✓ 学习质量评分: {:.1}%", quality.overall_score * 100.0);

        // Step 3: 分析变量语义
        let variable_result =
            VariableSemanticsAnalyzer::analyze_variables(&trace, &workflows.workflows)
                .expect("Failed to analyze variables");

        log::info!("✓ 分析了 {} 个变量", variable_result.variables.len());

        // Step 4: 推断数据结构
        let structure_result =
            DataStructureInferenceEngine::infer_structures(&trace, &variable_result.variables)
                .expect("Failed to infer structures");

        log::info!("✓ 推断了 {} 个数据结构", structure_result.structures.len());

        // Step 5: 执行完整推理
        let inference_result = CompleteInferencePipeline::infer(&trace, &workflows)
            .expect("Failed to run inference pipeline");

        assert!(
            inference_result.overall_inference_score > 0.0,
            "Overall inference score should be positive"
        );
        log::info!(
            "✓ 推理评分: {:.1}%",
            inference_result.overall_inference_score * 100.0
        );

        // Step 6: 生成代码
        let modules = ImprovedCodeGenerator::generate_code(&inference_result)
            .expect("Failed to generate code");

        assert!(!modules.is_empty(), "Should generate at least one module");
        log::info!("✓ 生成了 {} 个代码模块", modules.len());

        // 验证生成的代码
        for module in &modules {
            assert!(
                !module.code.is_empty(),
                "Generated code should not be empty"
            );
            log::info!(
                "  - {}: {} 行代码",
                module.module_name,
                module.code.lines().count()
            );
        }
    }

    #[test]
    fn test_workflow_extraction_produces_valid_workflows() {
        let trace = create_test_trace();
        let workflows =
            WorkflowExtractor::extract_workflows(&trace).expect("Failed to extract workflows");

        for workflow in &workflows.workflows {
            assert!(
                !workflow.name.is_empty(),
                "Workflow name should not be empty"
            );
            assert!(
                workflow.complexity_score >= 0.0 && workflow.complexity_score <= 10.0,
                "Complexity score should be between 0-10"
            );
            assert!(
                workflow.importance_score >= 0.0 && workflow.importance_score <= 10.0,
                "Importance score should be between 0-10"
            );
        }
    }

    #[test]
    fn test_inference_pipeline_combines_all_stages() {
        let trace = create_test_trace();
        let workflows =
            WorkflowExtractor::extract_workflows(&trace).expect("Failed to extract workflows");

        let inference_result =
            CompleteInferencePipeline::infer(&trace, &workflows).expect("Failed to infer");

        // 验证所有推理阶段都被执行
        assert!(inference_result.learning_quality.overall_score > 0.0);
        assert!(inference_result.variable_inference.accuracy >= 0.0);
        assert!(inference_result.structure_inference.accuracy >= 0.0);
        assert!(
            inference_result.overall_inference_score >= 0.0
                && inference_result.overall_inference_score <= 1.0
        );
    }

    #[test]
    fn test_code_generation_with_client() {
        let config = ClientConfig::default();
        let client = BrowserAIClient::new(config);

        assert_eq!(client.state(), ClientState::Idle);
    }

    #[test]
    fn test_complete_cycle_with_all_modules() {
        println!("🔄 完整周期测试开始...");

        let trace = create_test_trace();

        // 执行完整周期
        let workflows = WorkflowExtractor::extract_workflows(&trace).unwrap();
        let _quality = LearningQuality::evaluate(&trace, &workflows).unwrap();
        let variables =
            VariableSemanticsAnalyzer::analyze_variables(&trace, &workflows.workflows).unwrap();
        let _structures =
            DataStructureInferenceEngine::infer_structures(&trace, &variables.variables).unwrap();
        let inference = CompleteInferencePipeline::infer(&trace, &workflows).unwrap();
        let modules = ImprovedCodeGenerator::generate_code(&inference).unwrap();

        // 验证所有模块都产生了输出
        assert!(!modules.is_empty());

        println!("✅ 完整周期测试成功!");
    }
}
