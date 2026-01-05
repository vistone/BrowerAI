use anyhow::Result;
use std::path::PathBuf;

use crate::ai::model_manager::{ModelManager, ModelType};
use crate::ai::performance_monitor::PerformanceMonitor;
use crate::ai::AiRuntime;

/// AI 报告生成器，用于导出模型健康、性能指标和学习状态
pub struct AiReporter {
    runtime: AiRuntime,
    monitor: PerformanceMonitor,
}

impl AiReporter {
    pub fn new(runtime: AiRuntime, monitor: PerformanceMonitor) -> Self {
        Self { runtime, monitor }
    }

    /// 生成完整的 AI 状态报告
    pub fn generate_full_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("╔════════════════════════════════════════════════════════════════╗\n");
        report.push_str("║          BrowerAI - AI 系统状态报告                           ║\n");
        report.push_str("╚════════════════════════════════════════════════════════════════╝\n\n");

        // 1. 性能监控报告
        report.push_str("【性能监控】\n");
        report.push_str(&self.monitor.generate_report());
        report.push_str("\n");

        // 2. 模型健康状态
        report.push_str("【模型健康状态】\n");
        if self.runtime.has_models() {
            report.push_str(&self.generate_model_health_report());
        } else {
            report.push_str("  ⚠️  未加载模型目录\n");
        }
        report.push_str("\n");

        // 3. 推荐操作
        report.push_str("【推荐操作】\n");
        report.push_str(&self.generate_recommendations());

        report
    }

    /// 生成模型健康状态报告
    fn generate_model_health_report(&self) -> String {
        let mut report = String::new();
        
        let model_types = vec![
            ModelType::HtmlParser,
            ModelType::CssParser,
            ModelType::JsParser,
            ModelType::LayoutOptimizer,
            ModelType::RenderingOptimizer,
            ModelType::CodeUnderstanding,
            ModelType::JsDeobfuscator,
        ];

        for model_type in model_types {
            let type_name = format!("{:?}", model_type);
            if let Some((config, path)) = self.runtime.best_model(model_type) {
                let health_icon = match config.health {
                    crate::ai::model_manager::ModelHealth::Ready => "✅",
                    crate::ai::model_manager::ModelHealth::MissingFile => "❌",
                    crate::ai::model_manager::ModelHealth::Unknown => "❓",
                };
                report.push_str(&format!(
                    "  {} {:20} | {} | v{} | 优先级: {} | 路径: {}\n",
                    health_icon,
                    type_name,
                    config.name,
                    config.version,
                    config.priority,
                    path.display()
                ));
            } else {
                report.push_str(&format!("  ⚠️  {:20} | 无可用模型\n", type_name));
            }
        }

        report
    }

    /// 生成推荐操作
    fn generate_recommendations(&self) -> String {
        let mut recommendations = Vec::new();

        // 检查性能指标
        let all_stats = self.monitor.get_all_stats();
        for stats in &all_stats {
            if stats.total_inferences > 100 && stats.success_rate() < 80.0 {
                recommendations.push(format!(
                    "  ⚠️  模型 '{}' 成功率低 ({:.1}%)，建议重新训练或切换模型",
                    stats.model_name,
                    stats.success_rate()
                ));
            }
        }

        // 检查模型可用性
        if !self.runtime.has_models() {
            recommendations.push("  💡 运行 'cd training && python scripts/prepare_data.py' 准备训练数据".to_string());
            recommendations.push("  💡 运行训练脚本生成模型，参考 training/QUICKSTART.md".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("  ✅ 系统运行正常，无需特殊操作".to_string());
        }

        recommendations.join("\n")
    }

    /// 导出性能指标到 JSON
    pub fn export_metrics_json(&self) -> Result<String> {
        let all_stats = self.monitor.get_all_stats();
        
        let json_data: Vec<serde_json::Value> = all_stats
            .iter()
            .map(|stats| {
                serde_json::json!({
                    "model_name": stats.model_name,
                    "total_inferences": stats.total_inferences,
                    "successful_inferences": stats.successful_inferences,
                    "failed_inferences": stats.failed_inferences,
                    "success_rate": stats.success_rate(),
                    "avg_inference_time_ms": stats.avg_inference_time.as_secs_f64() * 1000.0,
                    "min_inference_time_ms": stats.min_inference_time.as_secs_f64() * 1000.0,
                    "max_inference_time_ms": stats.max_inference_time.as_secs_f64() * 1000.0,
                    "throughput": stats.throughput(),
                    "total_input_bytes": stats.total_input_bytes,
                    "total_output_bytes": stats.total_output_bytes,
                })
            })
            .collect();

        Ok(serde_json::to_string_pretty(&json_data)?)
    }
}
