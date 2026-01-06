use anyhow::Result;
use std::path::PathBuf;

use crate::ai::model_manager::{ModelManager, ModelType};
use crate::ai::performance_monitor::PerformanceMonitor;
use crate::ai::AiRuntime;

/// AI report generator for exporting model health, performance metrics, and learning status
pub struct AiReporter {
    runtime: AiRuntime,
    monitor: PerformanceMonitor,
}

impl AiReporter {
    pub fn new(runtime: AiRuntime, monitor: PerformanceMonitor) -> Self {
        Self { runtime, monitor }
    }

    /// Generate complete AI status report
    pub fn generate_full_report(&self) -> String {
        let mut report = String::new();

        report.push_str("╔════════════════════════════════════════════════════════════════╗\n");
        report.push_str("║          BrowerAI - AI System Status Report                  ║\n");
        report.push_str("╚════════════════════════════════════════════════════════════════╝\n\n");

        // 1. Performance monitoring report
        report.push_str("【Performance Monitoring】\n");
        report.push_str(&self.monitor.generate_report());
        report.push_str("\n");

        // 2. Model health status
        report.push_str("【Model Health Status】\n");
        if self.runtime.has_models() {
            report.push_str(&self.generate_model_health_report());
        } else {
            report.push_str("  ⚠️  No model directory loaded\n");
        }
        report.push_str("\n");

        // 3. Recommended actions
        report.push_str("【Recommended Actions】\n");
        report.push_str(&self.generate_recommendations());

        report
    }

    /// Generate model health status report
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
                let health_icon = match &config.health {
                    crate::ai::model_manager::ModelHealth::Ready => "✅",
                    crate::ai::model_manager::ModelHealth::MissingFile => "❌",
                    crate::ai::model_manager::ModelHealth::LoadFailed(_) => "⚠️",
                    crate::ai::model_manager::ModelHealth::ValidationFailed(_) => "⚠️",
                    crate::ai::model_manager::ModelHealth::InferenceFailing => "❗",
                    crate::ai::model_manager::ModelHealth::Unknown => "❓",
                };
                report.push_str(&format!(
                    "  {} {:20} | {} | v{} | Priority: {} | Path: {}\n",
                    health_icon,
                    type_name,
                    config.name,
                    config.version,
                    config.priority,
                    path.display()
                ));
            } else {
                report.push_str(&format!("  ⚠️  {:20} | No available model\n", type_name));
            }
        }

        report
    }

    /// Generate recommendations
    fn generate_recommendations(&self) -> String {
        let mut recommendations = Vec::new();

        // Check performance metrics
        let all_stats = self.monitor.get_all_stats();
        for stats in &all_stats {
            if stats.total_inferences > 100 && stats.success_rate() < 80.0 {
                recommendations.push(format!(
                    "  ⚠️  Model '{}' has low success rate ({:.1}%), consider retraining or switching models",
                    stats.model_name,
                    stats.success_rate()
                ));
            }
        }

        // Check model availability
        if !self.runtime.has_models() {
            recommendations.push(
                "  💡 Run 'cd training && python scripts/prepare_data.py' to prepare training data"
                    .to_string(),
            );
            recommendations.push(
                "  💡 Run training scripts to generate models, see training/QUICKSTART.md"
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations
                .push("  ✅ System running normally, no special actions needed".to_string());
        }

        recommendations.join("\n")
    }

    /// Export performance metrics to JSON
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
