//! 智能渲染器 - 渲染阶段
//!
//! 将生成的体验渲染给用户

use super::generation::GeneratedExperience;
use anyhow::Result;

/// 智能渲染器
pub struct IntelligentRenderer {
    /// 当前选择的体验
    current_experience: GeneratedExperience,

    /// 所有可用体验
    available_experiences: Vec<GeneratedExperience>,

    /// 审计日志
    audit_log: Vec<AuditEntry>,
}

/// 渲染结果
pub struct RenderResult {
    /// 最终HTML
    pub final_html: String,

    /// 最终CSS
    pub final_css: String,

    /// 最终JS（原始 + 桥接）
    pub final_js: String,

    /// 渲染统计
    pub stats: RenderStats,
}

/// 渲染统计
#[derive(Debug, Clone)]
pub struct RenderStats {
    pub html_size: usize,
    pub css_size: usize,
    pub js_size: usize,
    pub functions_bridged: usize,
}

/// 候选摘要（用于 DevTools 面板）
#[derive(Debug, Clone)]
pub struct CandidateSummary {
    pub variant_id: String,
    pub compatibility_score: f32,
    pub accessibility_score: f32,
    pub performance_score: f32,
}

/// 审计条目
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub action: String,
    pub variant_id: String,
}

impl IntelligentRenderer {
    /// 创建渲染器
    pub fn new(
        current_experience: GeneratedExperience,
        available_experiences: Vec<GeneratedExperience>,
    ) -> Self {
        Self {
            current_experience,
            available_experiences,
            audit_log: Vec::new(),
        }
    }

    /// 智能渲染
    pub fn render(&self) -> Result<RenderResult> {
        // 1. 组装最终页面
        let final_html = self.assemble_page()?;

        // 2. 收集统计信息
        let stats = self.collect_stats()?;

        Ok(RenderResult {
            final_html: final_html.clone(),
            final_css: self.current_experience.css.clone(),
            final_js: self.current_experience.bridge_js.clone(),
            stats,
        })
    }

    /// 运行时切换体验
    pub fn switch_experience(&mut self, variant_id: &str) -> Result<()> {
        let experience = self
            .available_experiences
            .iter()
            .find(|e| e.variant_id == variant_id)
            .ok_or_else(|| anyhow::anyhow!("Experience not found"))?;

        self.current_experience = (*experience).clone();
        self.audit_log.push(AuditEntry {
            action: "switch".to_string(),
            variant_id: variant_id.to_string(),
        });
        Ok(())
    }

    /// 获取可用体验列表
    pub fn get_available_experiences(&self) -> Vec<String> {
        self.available_experiences
            .iter()
            .map(|e| e.variant_id.clone())
            .collect()
    }

    /// 获取候选摘要（简单评分示例）
    pub fn list_candidates(&self) -> Vec<CandidateSummary> {
        self.available_experiences
            .iter()
            .map(|e| {
                let stats = RenderStats {
                    html_size: e.html.len(),
                    css_size: e.css.len(),
                    js_size: e.bridge_js.len(),
                    functions_bridged: e.function_validation.function_map.len(),
                };
                // 简单的打分逻辑：功能映射越多 → 兼容性分更高；
                // CSS/JS 越小 → 性能分更高；可访问性分暂定固定值（示例）。
                let compatibility = (stats.functions_bridged as f32).max(1.0);
                let performance =
                    1_000_000.0_f32.min((stats.html_size + stats.css_size + stats.js_size) as f32);
                CandidateSummary {
                    variant_id: e.variant_id.clone(),
                    compatibility_score: compatibility / 10.0,
                    accessibility_score: 0.8, // TODO: 接入可访问性检查
                    performance_score: (1_000_000.0_f32 / performance).min(1.0),
                }
            })
            .collect()
    }

    /// 获取审计日志
    pub fn audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }

    fn assemble_page(&self) -> Result<String> {
        let mut page = self.current_experience.html.clone();

        // 注入CSS
        let css_tag = format!(
            "<style>\n{}\n</style>\n</head>",
            self.current_experience.css
        );
        page = page.replace("</head>", &css_tag);

        // 注入JS
        let js_tag = format!(
            "<script>\n{}\n</script>\n</body>",
            self.current_experience.bridge_js
        );
        page = page.replace("</body>", &js_tag);

        // 注入体验切换器
        let switcher = self.generate_experience_switcher();
        page = page.replace("</body>", &format!("{}\n</body>", switcher));

        Ok(page)
    }

    fn generate_experience_switcher(&self) -> String {
        let mut switcher = String::from(
            "<div id='browerai-switcher' style='position:fixed;bottom:20px;right:20px;background:#fff;padding:15px;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,0.1);z-index:9999;'>\n"
        );

        switcher.push_str("  <h4 style='margin:0 0 10px 0;font-size:14px;'>体验切换</h4>\n");

        for exp in &self.available_experiences {
            let is_current = exp.variant_id == self.current_experience.variant_id;
            let style = if is_current {
                "background:#3B82F6;color:white;border:none;padding:8px 12px;margin:4px 0;cursor:pointer;border-radius:4px;display:block;width:100%;"
            } else {
                "background:#F3F4F6;color:#111827;border:none;padding:8px 12px;margin:4px 0;cursor:pointer;border-radius:4px;display:block;width:100%;"
            };

            switcher.push_str(&format!(
                "  <button onclick=\"location.reload()\" style=\"{}\">{}</button>\n",
                style, exp.variant_id
            ));
        }

        switcher.push_str("</div>\n");

        switcher
    }

    fn collect_stats(&self) -> Result<RenderStats> {
        Ok(RenderStats {
            html_size: self.current_experience.html.len(),
            css_size: self.current_experience.css.len(),
            js_size: self.current_experience.bridge_js.len(),
            functions_bridged: self
                .current_experience
                .function_validation
                .function_map
                .len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::IntelligentGeneration;
    use crate::reasoning::IntelligentReasoning;
    use crate::site_understanding::SiteUnderstanding;

    #[test]
    fn test_render_process() {
        let html = "<html><body><h1>Test</h1></body></html>".to_string();
        let css = "".to_string();
        let js = "".to_string();

        let understanding = SiteUnderstanding::learn_from_content(html, css, js).unwrap();
        let reasoning = IntelligentReasoning::new(understanding);
        let reasoning_result = reasoning.reason().unwrap();

        let generation = IntelligentGeneration::new(reasoning_result);
        let experiences = generation.generate().unwrap();

        assert!(!experiences.is_empty());

        let renderer = IntelligentRenderer::new(experiences[0].clone(), experiences.clone());

        let result = renderer.render().unwrap();

        assert!(!result.final_html.is_empty());
        assert!(!result.final_css.is_empty());
        assert!(!result.final_js.is_empty());
        assert!(result.stats.html_size > 0);
    }

    #[test]
    fn test_experience_switching() {
        let html = "<html><body><h1>Test</h1></body></html>".to_string();
        let css = "".to_string();
        let js = "".to_string();

        let understanding = SiteUnderstanding::learn_from_content(html, css, js).unwrap();
        let reasoning = IntelligentReasoning::new(understanding);
        let reasoning_result = reasoning.reason().unwrap();

        let generation = IntelligentGeneration::new(reasoning_result);
        let experiences = generation.generate().unwrap();

        let mut renderer = IntelligentRenderer::new(experiences[0].clone(), experiences.clone());

        let available = renderer.get_available_experiences();
        assert!(available.len() >= 3);

        // 切换到不同的体验
        if available.len() > 1 {
            renderer.switch_experience(&available[1]).unwrap();
            assert_eq!(renderer.audit_log().last().unwrap().action, "switch");
        }

        // 候选摘要应可用
        let candidates = renderer.list_candidates();
        assert!(!candidates.is_empty());
        assert!(candidates[0].compatibility_score >= 0.0);
    }
}
