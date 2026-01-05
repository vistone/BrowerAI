//! 智能渲染器 - 渲染阶段
//! 
//! 将生成的体验渲染给用户

use anyhow::Result;
use super::generation::GeneratedExperience;

/// 智能渲染器
pub struct IntelligentRenderer {
    /// 当前选择的体验
    current_experience: GeneratedExperience,
    
    /// 所有可用体验
    available_experiences: Vec<GeneratedExperience>,
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

impl IntelligentRenderer {
    /// 创建渲染器
    pub fn new(
        current_experience: GeneratedExperience,
        available_experiences: Vec<GeneratedExperience>,
    ) -> Self {
        Self {
            current_experience,
            available_experiences,
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
        let experience = self.available_experiences
            .iter()
            .find(|e| e.variant_id == variant_id)
            .ok_or_else(|| anyhow::anyhow!("Experience not found"))?;
        
        self.current_experience = (*experience).clone();
        Ok(())
    }
    
    /// 获取可用体验列表
    pub fn get_available_experiences(&self) -> Vec<String> {
        self.available_experiences
            .iter()
            .map(|e| e.variant_id.clone())
            .collect()
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
            functions_bridged: self.current_experience
                .function_validation
                .function_map
                .len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intelligent_rendering::site_understanding::SiteUnderstanding;
    use crate::intelligent_rendering::reasoning::IntelligentReasoning;
    use crate::intelligent_rendering::generation::IntelligentGeneration;
    
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
        
        let renderer = IntelligentRenderer::new(
            experiences[0].clone(),
            experiences.clone(),
        );
        
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
        
        let mut renderer = IntelligentRenderer::new(
            experiences[0].clone(),
            experiences.clone(),
        );
        
        let available = renderer.get_available_experiences();
        assert!(available.len() >= 3);
        
        // 切换到不同的体验
        if available.len() > 1 {
            renderer.switch_experience(&available[1]).unwrap();
        }
    }
}
