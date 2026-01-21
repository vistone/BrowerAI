//! DevTools 样式切换面板 UI 模块
//!
//! 提供候选列表、应用、回退与审计日志展示的最小 UI 原型
//! 支持 CLI 输出（plain text）与结构化数据（JSON）两种模式

use crate::style_switcher::StyleSwitcherBackend;
use anyhow::Result;
use std::fmt;

/// 面板 UI 组件
pub struct StyleSwitcherPanel {
    backend: Box<dyn StyleSwitcherBackend>,
}

/// 候选项的可视化展示
#[derive(Debug, Clone)]
pub struct CandidateDisplay {
    pub variant_id: String,
    pub compatibility: String, // ████░░░░ 80%
    pub accessibility: String,
    pub performance: String,
}

impl fmt::Display for CandidateDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "  [{}]\n    兼容性: {}\n    可访问性: {}\n    性能: {}\n",
            self.variant_id, self.compatibility, self.accessibility, self.performance
        )
    }
}

impl StyleSwitcherPanel {
    /// 创建面板，传入后端实现
    pub fn new(backend: Box<dyn StyleSwitcherBackend>) -> Self {
        Self { backend }
    }

    /// 渲染候选列表（CLI 风格）
    pub fn render_candidates_cli(&self) -> Result<String> {
        let candidates = self.backend.list_candidates()?;
        let mut output = String::from("=== BrowerAI 样式/布局候选 ===\n\n");

        if candidates.is_empty() {
            output.push_str("（无可用候选）\n");
            return Ok(output);
        }

        for (idx, cand) in candidates.iter().enumerate() {
            let compat_bar = Self::score_to_bar(cand.compatibility_score);
            let access_bar = Self::score_to_bar(cand.accessibility_score);
            let perf_bar = Self::score_to_bar(cand.performance_score);

            output.push_str(&format!("{}. {}\n", idx + 1, cand.variant_id));
            output.push_str(&format!(
                "   兼容性:    {} {:.0}%\n",
                compat_bar,
                cand.compatibility_score * 100.0
            ));
            output.push_str(&format!(
                "   可访问性:  {} {:.0}%\n",
                access_bar,
                cand.accessibility_score * 100.0
            ));
            output.push_str(&format!(
                "   性能:      {} {:.0}%\n",
                perf_bar,
                cand.performance_score * 100.0
            ));
            output.push('\n');
        }

        output.push_str("选择候选：panel.apply_candidate(\"<variant_id>\")？\n");
        Ok(output)
    }

    /// 渲染审计日志（CLI 风格）
    pub fn render_audit_log_cli(&self) -> Result<String> {
        let audit = self.backend.audit_log()?;
        let mut output = String::from("=== 审计日志 ===\n\n");

        if audit.is_empty() {
            output.push_str("（无日志）\n");
            return Ok(output);
        }

        for (idx, entry) in audit.iter().enumerate() {
            output.push_str(&format!(
                "{}. [{}] 应用候选: {}\n",
                idx + 1,
                entry.action,
                entry.variant_id
            ));
        }

        Ok(output)
    }

    /// 应用候选并返回状态
    pub fn apply_and_report(&mut self, variant_id: &str) -> Result<String> {
        self.backend.apply_candidate(variant_id)?;
        Ok(format!("✓ 已应用候选: {}", variant_id))
    }

    /// 导出候选摘要为 JSON（用于外部系统集成）
    pub fn export_candidates_json(&self) -> Result<String> {
        let candidates = self.backend.list_candidates()?;
        let json = serde_json::to_string_pretty(&candidates)?;
        Ok(json)
    }

    /// 导出审计日志为 JSON（用于合规与审查）
    pub fn export_audit_json(&self) -> Result<String> {
        let audit = self.backend.audit_log()?;
        let json = serde_json::to_string_pretty(&audit)?;
        Ok(json)
    }

    /// 分数转为可视化条形图（用于 CLI 展示）
    fn score_to_bar(score: f32) -> String {
        let filled = (score * 8.0).round() as usize;
        let empty = 8 - filled;
        let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
        bar
    }

    /// 获取面板摘要（一页纸报告）
    pub fn render_summary(&self) -> Result<String> {
        let candidates = self.backend.list_candidates()?;
        let audit = self.backend.audit_log()?;

        let mut output = String::from("=== BrowerAI 面板摘要 ===\n\n");
        output.push_str(&format!("可用候选数: {}\n", candidates.len()));
        output.push_str(&format!("总操作数: {}\n\n", audit.len()));

        if !candidates.is_empty() {
            output.push_str("最佳候选 (综合评分):\n");
            let best = candidates
                .iter()
                .max_by(|a, b| {
                    let a_score =
                        a.compatibility_score + a.accessibility_score + a.performance_score;
                    let b_score =
                        b.compatibility_score + b.accessibility_score + b.performance_score;
                    a_score.partial_cmp(&b_score).unwrap()
                })
                .unwrap();
            output.push_str(&format!("  {}\n", best.variant_id));
            output.push_str(&format!(
                "  综合评分: {:.1}/3.0\n\n",
                best.compatibility_score + best.accessibility_score + best.performance_score
            ));
        }

        if !audit.is_empty() {
            output.push_str("最近操作:\n");
            if let Some(last) = audit.last() {
                output.push_str(&format!("  {} - {}\n", last.action, last.variant_id));
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style_switcher::{CandidateSummary, MemoryBackend};

    #[test]
    fn test_panel_render_candidates() {
        let candidates = vec![
            CandidateSummary {
                variant_id: "Minimal".to_string(),
                compatibility_score: 0.95,
                accessibility_score: 0.9,
                performance_score: 0.98,
            },
            CandidateSummary {
                variant_id: "CardBased".to_string(),
                compatibility_score: 0.85,
                accessibility_score: 0.88,
                performance_score: 0.82,
            },
        ];

        let backend = Box::new(MemoryBackend::new(candidates));
        let panel = StyleSwitcherPanel::new(backend);
        let output = panel.render_candidates_cli().unwrap();

        assert!(output.contains("Minimal"));
        assert!(output.contains("CardBased"));
        assert!(output.contains("兼容性"));
        assert!(output.contains("可访问性"));
    }

    #[test]
    fn test_panel_apply_and_audit() {
        let candidates = vec![CandidateSummary {
            variant_id: "Minimal".to_string(),
            compatibility_score: 0.9,
            accessibility_score: 0.85,
            performance_score: 0.92,
        }];

        let backend = Box::new(MemoryBackend::new(candidates));
        let mut panel = StyleSwitcherPanel::new(backend);

        let result = panel.apply_and_report("Minimal").unwrap();
        assert!(result.contains("Minimal"));

        let audit = panel.render_audit_log_cli().unwrap();
        assert!(audit.contains("Minimal"));
    }

    #[test]
    fn test_panel_summary() {
        let candidates = vec![CandidateSummary {
            variant_id: "Minimal".to_string(),
            compatibility_score: 0.9,
            accessibility_score: 0.85,
            performance_score: 0.92,
        }];

        let backend = Box::new(MemoryBackend::new(candidates));
        let panel = StyleSwitcherPanel::new(backend);

        let summary = panel.render_summary().unwrap();
        assert!(summary.contains("可用候选数"));
        assert!(summary.contains("Minimal"));
    }

    #[test]
    fn test_score_to_bar() {
        assert_eq!(StyleSwitcherPanel::score_to_bar(0.0), "░░░░░░░░");
        assert_eq!(StyleSwitcherPanel::score_to_bar(1.0), "████████");
        assert_eq!(StyleSwitcherPanel::score_to_bar(0.5), "████░░░░");
    }
}
