use anyhow::Result;
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

use crate::learning::feedback::{Feedback, FeedbackType};

/// 在线反馈管道，收集浏览器运行中的各类事件用于后续学习
#[derive(Clone)]
pub struct FeedbackPipeline {
    events: Arc<RwLock<VecDeque<FeedbackEvent>>>,
    max_capacity: usize,
}

/// 反馈事件类型
#[derive(Debug, Clone)]
pub enum FeedbackEvent {
    /// HTML 解析事件
    HtmlParsing {
        success: bool,
        complexity: f32,
        ai_used: bool,
        error: Option<String>,
        /// 实际HTML内容（用于混淆分析）
        content: Option<String>,
        /// HTML大小
        size: usize,
    },
    /// CSS 解析事件
    CssParsing {
        success: bool,
        rule_count: usize,
        ai_used: bool,
        error: Option<String>,
        /// 实际CSS内容
        content: Option<String>,
    },
    /// JavaScript 解析事件
    JsParsing {
        success: bool,
        statement_count: usize,
        compatibility_warnings: Vec<String>,
        ai_used: bool,
        error: Option<String>,
        /// 实际JS内容（用于混淆检测）
        content: Option<String>,
    },
    /// JavaScript 执行兼容性违规
    JsCompatibilityViolation {
        feature: String,
        detail: String,
        enforced: bool,
    },
    /// 渲染性能事件
    RenderingPerformance {
        node_count: usize,
        duration_ms: f64,
        ai_optimized: bool,
    },
    /// 布局性能事件
    LayoutPerformance {
        element_count: usize,
        duration_ms: f64,
        ai_optimized: bool,
    },
    /// AI 模型推理事件
    ModelInference {
        model_name: String,
        success: bool,
        duration_ms: f64,
        error: Option<String>,
    },
}

impl FeedbackPipeline {
    /// 创建新的反馈管道
    pub fn new(max_capacity: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(VecDeque::with_capacity(max_capacity))),
            max_capacity,
        }
    }

    /// 添加事件到管道
    pub fn push_event(&self, event: FeedbackEvent) {
        let mut events = self.events.write().unwrap();
        
        if events.len() >= self.max_capacity {
            events.pop_front(); // 删除最老的事件
        }
        
        events.push_back(event);
    }

    /// 记录 HTML 解析事件
    pub fn record_html_parsing(&self, success: bool, complexity: f32, ai_used: bool, error: Option<String>, content: Option<String>, size: usize) {
        self.push_event(FeedbackEvent::HtmlParsing {
            success,
            complexity,
            ai_used,
            error,
            content,
            size,
        });
    }

    /// 记录 CSS 解析事件
    pub fn record_css_parsing(&self, success: bool, rule_count: usize, ai_used: bool, error: Option<String>, content: Option<String>) {
        self.push_event(FeedbackEvent::CssParsing {
            success,
            rule_count,
            ai_used,
            error,
            content,
        });
    }

    /// 记录 JavaScript 解析事件
    pub fn record_js_parsing(
        &self,
        success: bool,
        statement_count: usize,
        compatibility_warnings: Vec<String>,
        ai_used: bool,
        error: Option<String>,
        content: Option<String>,
    ) {
        self.push_event(FeedbackEvent::JsParsing {
            success,
            statement_count,
            compatibility_warnings,
            ai_used,
            error,
            content,
        });
    }

    /// 记录 JS 兼容性违规
    pub fn record_js_compatibility_violation(&self, feature: String, detail: String, enforced: bool) {
        self.push_event(FeedbackEvent::JsCompatibilityViolation {
            feature,
            detail,
            enforced,
        });
    }

    /// 记录渲染性能事件
    pub fn record_rendering_performance(&self, node_count: usize, duration_ms: f64, ai_optimized: bool) {
        self.push_event(FeedbackEvent::RenderingPerformance {
            node_count,
            duration_ms,
            ai_optimized,
        });
    }

    /// 记录布局性能事件
    pub fn record_layout_performance(&self, element_count: usize, duration_ms: f64, ai_optimized: bool) {
        self.push_event(FeedbackEvent::LayoutPerformance {
            element_count,
            duration_ms,
            ai_optimized,
        });
    }

    /// 记录模型推理事件
    pub fn record_model_inference(&self, model_name: String, success: bool, duration_ms: f64, error: Option<String>) {
        self.push_event(FeedbackEvent::ModelInference {
            model_name,
            success,
            duration_ms,
            error,
        });
    }

    /// 获取所有事件
    pub fn get_events(&self) -> Vec<FeedbackEvent> {
        self.events.read().unwrap().iter().cloned().collect()
    }

    /// 清空事件队列
    pub fn clear(&self) {
        self.events.write().unwrap().clear();
    }

    /// 获取事件数量
    pub fn len(&self) -> usize {
        self.events.read().unwrap().len()
    }

    /// 转换为训练样本格式（可导出到 JSON）
    pub fn export_training_samples(&self) -> Result<String> {
        let events = self.get_events();
        let samples: Vec<serde_json::Value> = events
            .iter()
            .map(|event| self.event_to_json(event))
            .collect();

        Ok(serde_json::to_string_pretty(&samples)?)
    }

    /// 将事件转换为 JSON
    fn event_to_json(&self, event: &FeedbackEvent) -> serde_json::Value {
        match event {
            FeedbackEvent::HtmlParsing { success, complexity, ai_used, error, content, size } => {
                serde_json::json!({
                    "type": "html_parsing",
                    "success": success,
                    "complexity": complexity,
                    "ai_used": ai_used,
                    "error": error,
                    "content": content,
                    "size": size,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            FeedbackEvent::CssParsing { success, rule_count, ai_used, error, content } => {
                serde_json::json!({
                    "type": "css_parsing",
                    "success": success,
                    "rule_count": rule_count,
                    "ai_used": ai_used,
                    "error": error,
                    "content": content,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            FeedbackEvent::JsParsing { success, statement_count, compatibility_warnings, ai_used, error, content } => {
                serde_json::json!({
                    "type": "js_parsing",
                    "success": success,
                    "statement_count": statement_count,
                    "compatibility_warnings": compatibility_warnings,
                    "ai_used": ai_used,
                    "error": error,
                    "content": content,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            FeedbackEvent::JsCompatibilityViolation { feature, detail, enforced } => {
                serde_json::json!({
                    "type": "js_compatibility_violation",
                    "feature": feature,
                    "detail": detail,
                    "enforced": enforced,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            FeedbackEvent::RenderingPerformance { node_count, duration_ms, ai_optimized } => {
                serde_json::json!({
                    "type": "rendering_performance",
                    "node_count": node_count,
                    "duration_ms": duration_ms,
                    "ai_optimized": ai_optimized,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            FeedbackEvent::LayoutPerformance { element_count, duration_ms, ai_optimized } => {
                serde_json::json!({
                    "type": "layout_performance",
                    "element_count": element_count,
                    "duration_ms": duration_ms,
                    "ai_optimized": ai_optimized,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            FeedbackEvent::ModelInference { model_name, success, duration_ms, error } => {
                serde_json::json!({
                    "type": "model_inference",
                    "model_name": model_name,
                    "success": success,
                    "duration_ms": duration_ms,
                    "error": error,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
        }
    }

    /// 生成统计摘要
    pub fn generate_summary(&self) -> String {
        let events = self.get_events();
        let mut summary = String::new();

        summary.push_str("【反馈管道统计】\n");
        summary.push_str(&format!("  总事件数: {}\n", events.len()));

        // 统计各类事件
        let html_count = events.iter().filter(|e| matches!(e, FeedbackEvent::HtmlParsing { .. })).count();
        let css_count = events.iter().filter(|e| matches!(e, FeedbackEvent::CssParsing { .. })).count();
        let js_count = events.iter().filter(|e| matches!(e, FeedbackEvent::JsParsing { .. })).count();
        let violation_count = events.iter().filter(|e| matches!(e, FeedbackEvent::JsCompatibilityViolation { .. })).count();

        summary.push_str(&format!("  HTML 解析事件: {}\n", html_count));
        summary.push_str(&format!("  CSS 解析事件: {}\n", css_count));
        summary.push_str(&format!("  JS 解析事件: {}\n", js_count));
        summary.push_str(&format!("  兼容性违规: {}\n", violation_count));

        summary
    }
}

impl Default for FeedbackPipeline {
    fn default() -> Self {
        Self::new(10000) // 默认保存 10000 条事件
    }
}
