//! 智能生成模块 - 生成阶段
//!
//! 生成保持功能的多样化体验

use super::reasoning::{CoreFunction, ExperienceVariant, ReasoningResult};
use super::validation::FunctionValidation;
use crate::intelligent_rendering::FunctionMapping;
use anyhow::Result;
use std::collections::HashMap;

/// 智能生成
pub struct IntelligentGeneration {
    reasoning: ReasoningResult,
}

/// 生成的体验
#[derive(Clone)]
pub struct GeneratedExperience {
    /// 变体ID
    pub variant_id: String,

    /// 生成的HTML（保持功能）
    pub html: String,

    /// 生成的CSS（新体验）
    pub css: String,

    /// 桥接JS（功能映射）
    pub bridge_js: String,

    /// 功能验证
    pub function_validation: FunctionValidation,
}

impl IntelligentGeneration {
    /// 创建生成实例
    pub fn new(reasoning: ReasoningResult) -> Self {
        Self { reasoning }
    }

    /// 生成保持功能的新体验
    pub fn generate(&self) -> Result<Vec<GeneratedExperience>> {
        let mut experiences = Vec::new();

        for variant in &self.reasoning.experience_variants {
            // 1. 生成新的HTML结构
            let html = self.generate_html_for_variant(variant)?;

            // 2. 生成新的CSS样式
            let css = self.generate_css_for_variant(variant)?;

            // 3. 生成功能桥接JS
            let bridge_js = self.generate_function_bridge(variant)?;

            // 4. 验证功能完整性
            let validation = self.validate_functions(&html, &bridge_js)?;

            if validation.all_functions_present {
                experiences.push(GeneratedExperience {
                    variant_id: variant.name.clone(),
                    html,
                    css,
                    bridge_js,
                    function_validation: validation,
                });
            }
        }

        Ok(experiences)
    }

    fn generate_html_for_variant(&self, variant: &ExperienceVariant) -> Result<String> {
        let mut html = String::from("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!("  <title>{} Experience</title>\n", variant.name));
        html.push_str("  <meta charset='utf-8'>\n");
        html.push_str("</head>\n<body>\n");

        // 根据布局方案生成结构
        match variant.layout_scheme {
            crate::intelligent_rendering::LayoutScheme::Minimal => {
                html.push_str("  <div class='minimal-container'>\n");
                html.push_str("    <main class='minimal-content'>\n");
            }
            crate::intelligent_rendering::LayoutScheme::CardBased => {
                html.push_str("  <div class='card-grid'>\n");
                html.push_str("    <div class='card'>\n");
            }
            _ => {
                html.push_str("  <div class='container'>\n");
                html.push_str("    <main>\n");
            }
        }

        // 为每个核心功能生成元素
        for (original_name, new_id) in &variant.function_mapping {
            html.push_str(&format!(
                "      <div id='{}' data-original-function='{}'>\n",
                new_id, original_name
            ));
            html.push_str(&format!("        <p>Function: {}</p>\n", original_name));
            html.push_str("      </div>\n");
        }

        // 关闭标签
        match variant.layout_scheme {
            crate::intelligent_rendering::LayoutScheme::Minimal => {
                html.push_str("    </main>\n  </div>\n");
            }
            crate::intelligent_rendering::LayoutScheme::CardBased => {
                html.push_str("    </div>\n  </div>\n");
            }
            _ => {
                html.push_str("    </main>\n  </div>\n");
            }
        }

        html.push_str("</body>\n</html>");

        Ok(html)
    }

    fn generate_css_for_variant(&self, variant: &ExperienceVariant) -> Result<String> {
        let mut css = String::new();
        let style = &variant.visual_style;

        // 基础样式
        css.push_str("body {\n");
        css.push_str(&format!(
            "  font-family: {};\n",
            style.typography.font_family
        ));
        css.push_str(&format!("  font-size: {}px;\n", style.typography.base_size));
        css.push_str(&format!(
            "  background: {};\n",
            style.color_scheme.background
        ));
        css.push_str(&format!("  color: {};\n", style.color_scheme.text));
        css.push_str("  margin: 0;\n");
        css.push_str("  padding: 0;\n");
        css.push_str("}\n\n");

        // 容器样式
        match variant.layout_scheme {
            crate::intelligent_rendering::LayoutScheme::Minimal => {
                css.push_str(".minimal-container {\n");
                css.push_str("  max-width: 800px;\n");
                css.push_str("  margin: 0 auto;\n");
                css.push_str("  padding: 2rem;\n");
                css.push_str("}\n\n");
            }
            crate::intelligent_rendering::LayoutScheme::CardBased => {
                css.push_str(".card-grid {\n");
                css.push_str("  display: grid;\n");
                css.push_str("  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));\n");
                css.push_str("  gap: 1.5rem;\n");
                css.push_str("  padding: 2rem;\n");
                css.push_str("}\n\n");
                css.push_str(".card {\n");
                css.push_str("  background: white;\n");
                css.push_str("  border-radius: 8px;\n");
                css.push_str("  padding: 1.5rem;\n");
                css.push_str("  box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n");
                css.push_str("}\n\n");
            }
            _ => {
                css.push_str(".container {\n");
                css.push_str("  max-width: 1200px;\n");
                css.push_str("  margin: 0 auto;\n");
                css.push_str("  padding: 1rem;\n");
                css.push_str("}\n\n");
            }
        }

        Ok(css)
    }

    fn generate_function_bridge(&self, variant: &ExperienceVariant) -> Result<String> {
        let mut bridge_code = String::from("// BrowerAI 功能桥接层 - 确保原始功能完全保持\n\n");

        bridge_code.push_str("const BrowerAI = {\n");
        bridge_code.push_str("  functionBridge: {},\n");
        bridge_code.push_str("  originalHandlers: {},\n\n");

        bridge_code.push_str("  init: function() {\n");
        bridge_code.push_str("    console.log('BrowerAI: Initializing function bridges');\n");

        // 为每个核心功能生成桥接
        for (original_name, new_id) in &variant.function_mapping {
            bridge_code.push_str(&format!("    // 桥接 {} 功能\n", original_name));

            bridge_code.push_str(&format!(
                "    const elem_{} = document.getElementById('{}');\n",
                original_name.replace("-", "_"),
                new_id
            ));

            bridge_code.push_str(&format!(
                "    if (elem_{}) {{\n",
                original_name.replace("-", "_")
            ));

            bridge_code.push_str(&format!(
                "      elem_{}.addEventListener('click', (e) => {{\n",
                original_name.replace("-", "_")
            ));

            bridge_code.push_str(&format!(
                "        console.log('Function {} triggered');\n",
                original_name
            ));

            bridge_code.push_str("        // 调用原始功能逻辑\n");
            bridge_code.push_str(&format!(
                "        if (this.originalHandlers['{}']) {{\n",
                original_name
            ));
            bridge_code.push_str(&format!(
                "          this.originalHandlers['{}'].call(this, e);\n",
                original_name
            ));
            bridge_code.push_str("        }\n");
            bridge_code.push_str("      });\n");
            bridge_code.push_str("    }\n\n");
        }

        bridge_code.push_str("    console.log('BrowerAI: All function bridges initialized');\n");
        bridge_code.push_str("  }\n");
        bridge_code.push_str("};\n\n");

        bridge_code.push_str("// 页面加载后初始化\n");
        bridge_code.push_str("if (document.readyState === 'loading') {\n");
        bridge_code
            .push_str("  document.addEventListener('DOMContentLoaded', () => BrowerAI.init());\n");
        bridge_code.push_str("} else {\n");
        bridge_code.push_str("  BrowerAI.init();\n");
        bridge_code.push_str("}\n");

        Ok(bridge_code)
    }

    fn validate_functions(&self, html: &str, bridge_js: &str) -> Result<FunctionValidation> {
        let mut function_map = HashMap::new();
        let mut all_present = true;

        // 简化的验证：检查每个核心功能是否在HTML和JS中存在
        for core_func in &self.reasoning.core_functions {
            let in_html = html.contains(&format!("data-original-function='{}'", core_func.name));
            let in_js = bridge_js.contains(&format!("桥接 {} 功能", core_func.name));

            let is_mapped = in_html && in_js;
            all_present = all_present && is_mapped;

            function_map.insert(
                core_func.name.clone(),
                FunctionMapping {
                    original_id: core_func.name.clone(),
                    new_id: format!("new-{}", core_func.name),
                    is_mapped,
                },
            );
        }

        Ok(FunctionValidation {
            all_functions_present: all_present,
            function_map,
            interaction_tests: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intelligent_rendering::reasoning::IntelligentReasoning;
    use crate::intelligent_rendering::site_understanding::SiteUnderstanding;

    #[test]
    fn test_generation_process() {
        let html = "<html><body><input type='search'/></body></html>".to_string();
        let css = "".to_string();
        let js = "".to_string();

        let understanding = SiteUnderstanding::learn_from_content(html, css, js).unwrap();
        let reasoning = IntelligentReasoning::new(understanding);
        let reasoning_result = reasoning.reason().unwrap();

        let generation = IntelligentGeneration::new(reasoning_result);
        let experiences = generation.generate().unwrap();

        assert!(!experiences.is_empty());

        for exp in &experiences {
            assert!(!exp.html.is_empty());
            assert!(!exp.css.is_empty());
            assert!(!exp.bridge_js.is_empty());
            assert!(exp.function_validation.all_functions_present);
        }
    }
}
