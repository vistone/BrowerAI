//! 智能生成模块 - 生成阶段
//!
//! 生成保持功能的多样化体验 - 使用训练好的模型库生成样式

use super::model_api_client::{FallbackStyleGenerator, ModelApiClient, ModelApiConfig};
use super::reasoning::{ExperienceVariant, ReasoningResult};
use super::validation::FunctionValidation;
use crate::{FunctionMapping, LayoutScheme};
use anyhow::Result;
use std::collections::HashMap;

/// 智能生成
pub struct IntelligentGeneration {
    reasoning: ReasoningResult,
    /// 原始网站HTML内容
    original_html: String,
    /// 原始网站CSS内容
    original_css: String,
}

/// 生成的体验
#[derive(Clone)]
pub struct GeneratedExperience {
    /// 变体ID
    pub variant_id: String,

    /// 生成的HTML（保持原始内容，添加新样式）
    pub html: String,

    /// 生成的CSS（新体验样式 + 原始样式覆盖）
    pub css: String,

    /// 桥接JS（功能映射）
    pub bridge_js: String,

    /// 功能验证
    pub function_validation: FunctionValidation,
}

impl IntelligentGeneration {
    /// 创建生成实例（带原始内容）
    pub fn with_content(
        reasoning: ReasoningResult,
        original_html: String,
        original_css: String,
    ) -> Self {
        Self {
            reasoning,
            original_html,
            original_css,
        }
    }

    /// 创建生成实例（仅推理结果，向后兼容）
    pub fn new(reasoning: ReasoningResult) -> Self {
        Self {
            reasoning,
            original_html: String::new(),
            original_css: String::new(),
        }
    }

    /// 生成保持功能的新体验
    ///
    /// 使用训练好的模型库生成样式，如果模型 API 不可用则使用回退方案
    pub async fn generate(&self) -> Result<Vec<GeneratedExperience>> {
        let mut experiences = Vec::new();

        // 尝试连接模型 API
        let api_config = ModelApiConfig::default();
        let api_client = ModelApiClient::new(api_config)?;
        let api_available = api_client.health_check().await.unwrap_or(false);

        for (idx, variant) in self.reasoning.experience_variants.iter().enumerate() {
            // 1. 使用模型生成样式（或回退方案）
            let generated_style = if api_available {
                match api_client
                    .generate_style_from_content(
                        &variant.name,
                        &self.original_html,
                        &self.original_css,
                        "", // scripts
                        &format!("variant_{}", idx),
                    )
                    .await
                {
                    Ok(style) => {
                        log::info!(
                            "✅ Model API generated style with confidence: {:.2}",
                            style.confidence
                        );
                        style
                    }
                    Err(e) => {
                        log::warn!("⚠️ Model API failed ({}), using fallback", e);
                        FallbackStyleGenerator::generate(idx)
                    }
                }
            } else {
                log::info!("ℹ️ Model API not available, using fallback style generator");
                FallbackStyleGenerator::generate(idx)
            };

            // 保留原始样式并叠加新体验样式，确保功能相关视觉规则不丢失
            let composed_css = if self.original_css.trim().is_empty() {
                generated_style.css.clone()
            } else {
                format!(
                    "/* BrowerAI Preserved Original CSS */\n{}\n\n/* BrowerAI Generated Variant CSS */\n{}",
                    self.original_css,
                    generated_style.css
                )
            };

            // 2. 生成功能桥接JS
            let bridge_js = self.generate_function_bridge(variant)?;

            // 3. 生成HTML（保留原始内容 + 注入新样式和JS）
            let html = self.generate_html_for_variant(variant, &composed_css, &bridge_js)?;

            // 4. 验证功能完整性
            let validation = self.validate_functions(&html, &bridge_js)?;

            if validation.all_functions_present {
                experiences.push(GeneratedExperience {
                    variant_id: variant.name.clone(),
                    html,
                    css: composed_css,
                    bridge_js,
                    function_validation: validation,
                });
            }
        }

        Ok(experiences)
    }

    fn generate_html_for_variant(
        &self,
        variant: &ExperienceVariant,
        css: &str,
        js: &str,
    ) -> Result<String> {
        // 如果有原始HTML，基于原始内容添加新样式
        if !self.original_html.is_empty() {
            return self.inject_styles_into_original(&self.original_html, variant, css, js);
        }

        // 否则生成简化版HTML（向后兼容）
        self.generate_minimal_html(variant)
    }

    /// 将新样式注入原始HTML
    fn inject_styles_into_original(
        &self,
        original_html: &str,
        variant: &ExperienceVariant,
        css: &str,
        js: &str,
    ) -> Result<String> {
        let mut modified_html = original_html.to_string();

        // 在 </head> 前注入变体特定的样式类
        let style_class = match variant.layout_scheme {
            LayoutScheme::Minimal => "browerai-variant-minimal",
            LayoutScheme::CardBased => "browerai-variant-card",
            _ => "browerai-variant-default",
        };

        // 在 <body> 标签上添加样式类
        if let Some(body_start) = modified_html.find("<body") {
            if let Some(body_tag_end) = modified_html[body_start..].find(">") {
                let insert_pos = body_start + body_tag_end;
                modified_html.insert_str(insert_pos, &format!(" class='{}'", style_class));
            }
        }

        // 注入内联CSS样式（替换占位符或添加到head）
        let css_injection = format!(
            "<style id='browerai-variant-style'>\n/* {} Experience Styles */\n{}\n</style>\n",
            variant.name, css
        );

        if let Some(head_end) = modified_html.find("</head>") {
            modified_html.insert_str(head_end, &css_injection);
        }

        // 注入功能桥接脚本
        let bridge_script = format!("<script id='browerai-bridge'>\n{}\n</script>\n", js);
        if let Some(body_end) = modified_html.rfind("</body>") {
            modified_html.insert_str(body_end, &bridge_script);
        }

        Ok(modified_html)
    }

    /// 生成最小化HTML（向后兼容）
    fn generate_minimal_html(&self, variant: &ExperienceVariant) -> Result<String> {
        let mut html = String::from("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!("  <title>{} Experience</title>\n", variant.name));
        html.push_str("  <meta charset='utf-8'>\n");
        html.push_str("</head>\n<body>\n");

        // 根据布局方案生成结构
        match variant.layout_scheme {
            LayoutScheme::Minimal => {
                html.push_str("  <div class='minimal-container'>\n");
                html.push_str("    <main class='minimal-content'>\n");
            }
            LayoutScheme::CardBased => {
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
            LayoutScheme::Minimal => {
                html.push_str("    </main>\n  </div>\n");
            }
            LayoutScheme::CardBased => {
                html.push_str("    </div>\n  </div>\n");
            }
            _ => {
                html.push_str("    </main>\n  </div>\n");
            }
        }

        html.push_str("</body>\n</html>");

        Ok(html)
    }

    /// 生成 CSS（现在由模型 API 或回退生成器处理，此函数保留用于兼容性）
    fn _generate_css_for_variant(&self, _variant: &ExperienceVariant) -> Result<String> {
        // 样式现在由 ModelApiClient 或 FallbackStyleGenerator 生成
        // 此函数保留用于向后兼容
        Ok(String::new())
    }

    fn generate_function_bridge(&self, variant: &ExperienceVariant) -> Result<String> {
        let mut bridge_code = String::from("// BrowerAI 功能桥接层 - 确保原始功能完全保持\n\n");

        bridge_code.push_str("(function() {\n");
        bridge_code.push_str("  'use strict';\n\n");
        bridge_code.push_str("  const BrowerAI = {\n");
        bridge_code.push_str("    variantName: '");
        bridge_code.push_str(&variant.name);
        bridge_code.push_str("',\n");
        bridge_code.push_str("    functionBridge: {},\n");
        bridge_code.push_str("    originalHandlers: {},\n\n");

        bridge_code.push_str("    init: function() {\n");
        bridge_code.push_str("      console.log('[BrowerAI] Initializing function bridges for variant: ' + this.variantName);\n\n");

        // 为每个核心功能生成桥接
        for (original_name, new_id) in &variant.function_mapping {
            bridge_code.push_str(&format!("      // Bridge function: {}\n", original_name));
            bridge_code.push_str(&format!(
                "      this.bridgeFunction('{}', '{}');\n",
                original_name, new_id
            ));
        }

        bridge_code
            .push_str("\n      console.log('[BrowerAI] All function bridges initialized');\n");
        bridge_code.push_str("    },\n\n");

        bridge_code.push_str("    bridgeFunction: function(originalName, newId) {\n");
        bridge_code.push_str("      const newElement = document.getElementById(newId);\n");
        bridge_code.push_str("      const originalElements = document.querySelectorAll('[data-original-function=\"' + originalName + '\"]');\n\n");
        bridge_code.push_str("      if (newElement) {\n");
        bridge_code.push_str("        newElement.addEventListener('click', (e) => {\n");
        bridge_code
            .push_str("          console.log('[BrowerAI] Function triggered: ' + originalName);\n");
        bridge_code.push_str("          // Trigger original functionality\n");
        bridge_code.push_str("          originalElements.forEach(el => {\n");
        bridge_code.push_str("            if (el.click) el.click();\n");
        bridge_code.push_str("          });\n");
        bridge_code.push_str("        });\n");
        bridge_code.push_str("      }\n");
        bridge_code.push_str("    }\n");
        bridge_code.push_str("  };\n\n");

        bridge_code.push_str("  // Initialize when DOM is ready\n");
        bridge_code.push_str("  if (document.readyState === 'loading') {\n");
        bridge_code.push_str(
            "    document.addEventListener('DOMContentLoaded', () => BrowerAI.init());\n",
        );
        bridge_code.push_str("  } else {\n");
        bridge_code.push_str("    BrowerAI.init();\n");
        bridge_code.push_str("  }\n");
        bridge_code.push_str("})();\n");

        Ok(bridge_code)
    }

    fn validate_functions(&self, html: &str, bridge_js: &str) -> Result<FunctionValidation> {
        let mut function_map = HashMap::new();
        let mut all_present = true;

        // 简化的验证：检查每个核心功能是否在HTML和JS中存在
        for core_func in &self.reasoning.core_functions {
            let in_html = html.contains(&format!("data-original-function='{}'", core_func.name))
                || html.contains(&format!("id='{}", core_func.name));
            let in_js = bridge_js.contains(&core_func.name);

            let is_mapped = in_html || in_js;
            all_present = all_present && is_mapped;

            function_map.insert(
                core_func.name.clone(),
                FunctionMapping {
                    original_function: core_func.name.clone(),
                    new_function: format!("new-{}", core_func.name),
                    preserved: is_mapped,
                    reason: if is_mapped {
                        "mapped from original"
                    } else {
                        "not found"
                    }
                    .to_string(),
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
    use crate::reasoning::IntelligentReasoning;
    use crate::site_understanding::SiteUnderstanding;

    #[tokio::test]
    async fn test_generation_process() {
        let html = "<html><body><input type='search'/></body></html>".to_string();
        let css = "".to_string();
        let js = "".to_string();

        let understanding =
            SiteUnderstanding::learn_from_content(html.clone(), css.clone(), js).unwrap();
        let reasoning = IntelligentReasoning::new(understanding);
        let reasoning_result = reasoning.reason().unwrap();

        let generation = IntelligentGeneration::with_content(reasoning_result, html, css);
        let experiences = generation.generate().await.unwrap();

        assert!(!experiences.is_empty());

        for exp in &experiences {
            assert!(!exp.html.is_empty());
            assert!(!exp.css.is_empty());
            assert!(!exp.bridge_js.is_empty());
            assert!(exp.function_validation.all_functions_present);
        }
    }

    #[tokio::test]
    async fn test_content_preservation() {
        let original_html =
            "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>".to_string();
        let original_css = "body { color: black; }".to_string();
        let js = "".to_string();

        let understanding =
            SiteUnderstanding::learn_from_content(original_html.clone(), original_css.clone(), js)
                .unwrap();
        let reasoning = IntelligentReasoning::new(understanding);
        let reasoning_result = reasoning.reason().unwrap();

        let generation = IntelligentGeneration::with_content(
            reasoning_result,
            original_html.clone(),
            original_css.clone(),
        );
        let experiences = generation.generate().await.unwrap();

        assert!(!experiences.is_empty());

        // 验证原始内容被保留
        let exp = &experiences[0];
        assert!(
            exp.html.contains("<h1>Hello</h1>"),
            "Original content should be preserved"
        );
        assert!(
            exp.css.contains("color: black"),
            "Original CSS should be preserved"
        );
    }
}
