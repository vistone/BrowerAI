//! 样式生成器

use crate::*;
use anyhow::Result;

pub struct StyleGenerator {
    config: GenerationConfig,
}

impl StyleGenerator {
    pub fn new(config: &GenerationConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub async fn generate_styles(&self, analysis: &visual_learner::VisualAnalysis) -> Result<Vec<GeneratedFile>> {
        let mut styles = Vec::new();

        // 生成CSS变量
        let css_variables = self.generate_css_variables(analysis);
        styles.push(GeneratedFile {
            path: "src/styles/variables.css".to_string(),
            content: css_variables,
            file_type: FileType::Style,
        });

        // 生成基础样式
        let base_styles = self.generate_base_styles(analysis);
        styles.push(GeneratedFile {
            path: "src/styles/base.css".to_string(),
            content: base_styles,
            file_type: FileType::Style,
        });

        // 生成组件样式
        let component_styles = self.generate_component_styles(analysis);
        styles.push(GeneratedFile {
            path: "src/styles/components.css".to_string(),
            content: component_styles,
            file_type: FileType::Style,
        });

        // 生成布局样式
        let layout_styles = self.generate_layout_styles(analysis);
        styles.push(GeneratedFile {
            path: "src/styles/layout.css".to_string(),
            content: layout_styles,
            file_type: FileType::Style,
        });

        // 生成主样式文件
        let main_styles = self.generate_main_stylesheet();
        styles.push(GeneratedFile {
            path: "src/styles/index.css".to_string(),
            content: main_styles,
            file_type: FileType::Style,
        });

        Ok(styles)
    }

    fn generate_css_variables(&self, analysis: &visual_learner::VisualAnalysis) -> String {
        let mut css = String::from(":root {\n");

        // 颜色变量
        if let Some(ref c) = analysis.color_scheme.primary {
            css.push_str(&format!("  --color-primary: {};\n", c.to_hex()));
            css.push_str(&format!("  --color-primary-rgb: {}, {}, {};\n", c.r, c.g, c.b));
        }
        if let Some(ref c) = analysis.color_scheme.secondary {
            css.push_str(&format!("  --color-secondary: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.background {
            css.push_str(&format!("  --color-background: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.surface {
            css.push_str(&format!("  --color-surface: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.text_primary {
            css.push_str(&format!("  --color-text-primary: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.text_secondary {
            css.push_str(&format!("  --color-text-secondary: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.accent {
            css.push_str(&format!("  --color-accent: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.error {
            css.push_str(&format!("  --color-error: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.warning {
            css.push_str(&format!("  --color-warning: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.success {
            css.push_str(&format!("  --color-success: {};\n", c.to_hex()));
        }

        css.push_str("\n");

        // 间距变量
        css.push_str(&format!("  --spacing-unit: {}px;\n", analysis.spacing.base_unit));
        for (i, &value) in analysis.spacing.scale.iter().enumerate() {
            css.push_str(&format!("  --spacing-{}: {}px;\n", i + 1, value));
        }

        css.push_str("\n");

        // 排版变量
        for (i, &size) in analysis.typography.font_sizes.iter().enumerate() {
            css.push_str(&format!("  --font-size-{}: {}px;\n", i + 1, size));
        }

        css.push_str("\n");

        // 圆角变量
        css.push_str("  --radius-sm: 4px;\n");
        css.push_str("  --radius-md: 8px;\n");
        css.push_str("  --radius-lg: 12px;\n");
        css.push_str("  --radius-full: 9999px;\n");

        css.push_str("\n");

        // 阴影变量
        css.push_str("  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);\n");
        css.push_str("  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);\n");
        css.push_str("  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);\n");

        // 过渡变量
        css.push_str("\n  --transition-fast: 150ms ease;\n");
        css.push_str("  --transition-normal: 250ms ease;\n");
        css.push_str("  --transition-slow: 350ms ease;\n");

        css.push_str("}\n");
        css
    }

    fn generate_base_styles(&self, analysis: &visual_learner::VisualAnalysis) -> String {
        let mut css = String::new();

        // Reset
        css.push_str("/* Reset */\n");
        css.push_str("*, *::before, *::after {\n");
        css.push_str("  box-sizing: border-box;\n");
        css.push_str("  margin: 0;\n");
        css.push_str("  padding: 0;\n");
        css.push_str("}\n\n");

        // Base
        css.push_str("/* Base */\n");
        css.push_str("html {\n");
        css.push_str("  font-size: 16px;\n");
        css.push_str("  -webkit-font-smoothing: antialiased;\n");
        css.push_str("  -moz-osx-font-smoothing: grayscale;\n");
        css.push_str("}\n\n");

        css.push_str("body {\n");
        css.push_str("  font-family: system-ui, -apple-system, sans-serif;\n");
        if let Some(ref c) = analysis.color_scheme.background {
            css.push_str(&format!("  background-color: {};\n", c.to_hex()));
        }
        if let Some(ref c) = analysis.color_scheme.text_primary {
            css.push_str(&format!("  color: {};\n", c.to_hex()));
        }
        css.push_str("  line-height: 1.5;\n");
        css.push_str("}\n\n");

        // 无障碍
        if self.config.accessibility != AccessibilityLevel::None {
            css.push_str("/* Accessibility */\n");
            css.push_str(":focus-visible {\n");
            css.push_str("  outline: 2px solid var(--color-primary);\n");
            css.push_str("  outline-offset: 2px;\n");
            css.push_str("}\n\n");

            css.push_str(".sr-only {\n");
            css.push_str("  position: absolute;\n");
            css.push_str("  width: 1px;\n");
            css.push_str("  height: 1px;\n");
            css.push_str("  padding: 0;\n");
            css.push_str("  margin: -1px;\n");
            css.push_str("  overflow: hidden;\n");
            css.push_str("  clip: rect(0, 0, 0, 0);\n");
            css.push_str("  white-space: nowrap;\n");
            css.push_str("  border-width: 0;\n");
            css.push_str("}\n\n");
        }

        css
    }

    fn generate_component_styles(&self, analysis: &visual_learner::VisualAnalysis) -> String {
        let mut css = String::new();

        css.push_str("/* Component Styles */\n\n");

        for component in &analysis.components {
            let class_name = format!("{:?}", component.component_type).to_lowercase();
            
            css.push_str(&format!(".{} {{\n", class_name));

            // 尺寸
            css.push_str(&format!("  width: {}px;\n", component.bounding_box.width));
            css.push_str(&format!("  height: {}px;\n", component.bounding_box.height));

            // 背景色
            if let Some(ref color) = component.visual_style.background_color {
                css.push_str(&format!("  background-color: {};\n", color.to_hex()));
            }

            // 文本色
            if let Some(ref color) = component.visual_style.text_color {
                css.push_str(&format!("  color: {};\n", color.to_hex()));
            }

            // 边框
            if component.visual_style.border_width > 0 {
                css.push_str(&format!("  border: {}px solid", component.visual_style.border_width));
                if let Some(ref color) = component.visual_style.border_color {
                    css.push_str(&format!(" {};\n", color.to_hex()));
                } else {
                    css.push_str(" transparent;\n");
                }
            }

            // 圆角
            if component.visual_style.border_radius > 0 {
                css.push_str(&format!("  border-radius: {}px;\n", component.visual_style.border_radius));
            }

            // 阴影
            if let Some(ref shadow) = component.visual_style.shadow {
                css.push_str(&format!(
                    "  box-shadow: {}px {}px {}px {}px {};\n",
                    shadow.offset_x,
                    shadow.offset_y,
                    shadow.blur_radius,
                    shadow.spread_radius,
                    shadow.color.to_rgba()
                ));
            }

            // 透明度
            if component.visual_style.opacity < 1.0 {
                css.push_str(&format!("  opacity: {};\n", component.visual_style.opacity));
            }

            // 过渡
            css.push_str("  transition: all var(--transition-normal);\n");

            css.push_str("}\n\n");

            // 悬停状态
            css.push_str(&format!(".{}:hover {{\n", class_name));
            css.push_str("  filter: brightness(1.05);\n");
            css.push_str("}\n\n");

            // 禁用状态
            css.push_str(&format!(".{}:disabled {{\n", class_name));
            css.push_str("  opacity: 0.5;\n");
            css.push_str("  cursor: not-allowed;\n");
            css.push_str("}\n\n");
        }

        css
    }

    fn generate_layout_styles(&self, analysis: &visual_learner::VisualAnalysis) -> String {
        let mut css = String::new();

        css.push_str("/* Layout Styles */\n\n");

        // 布局容器
        css.push_str(".layout {\n");
        css.push_str("  display: flex;\n");
        css.push_str("  flex-direction: column;\n");
        css.push_str("  min-height: 100vh;\n");
        css.push_str("}\n\n");

        // 根据布局类型生成特定样式
        match analysis.layout.layout_type {
            visual_learner::LayoutType::Grid => {
                css.push_str(".layout--grid {\n");
                css.push_str("  display: grid;\n");
                if let Some(cols) = analysis.layout.grid_columns {
                    css.push_str(&format!("  grid-template-columns: repeat({}, 1fr);\n", cols));
                }
                css.push_str(&format!("  gap: {}px;\n", analysis.layout.gap));
                css.push_str("}\n\n");
            }
            visual_learner::LayoutType::FlexRow => {
                css.push_str(".layout--flexrow {\n");
                css.push_str("  flex-direction: row;\n");
                css.push_str(&format!("  gap: {}px;\n", analysis.layout.gap));
                css.push_str("}\n\n");
            }
            _ => {}
        }

        // 区域样式
        for section in &analysis.layout.sections {
            let section_class = section.name.to_lowercase();
            css.push_str(&format!(".{} {{\n", section_class));
            css.push_str(&format!("  padding: var(--spacing-3);\n"));
            
            match section.section_type {
                visual_learner::SectionType::Header => {
                    css.push_str("  position: sticky;\n");
                    css.push_str("  top: 0;\n");
                    css.push_str("  z-index: 100;\n");
                }
                visual_learner::SectionType::Content => {
                    css.push_str("  flex: 1;\n");
                }
                _ => {}
            }
            
            css.push_str("}\n\n");
        }

        css
    }

    fn generate_main_stylesheet(&self) -> String {
        r#"/* Main Stylesheet */

@import './variables.css';
@import './base.css';
@import './components.css';
@import './layout.css';

/* Utilities */
.text-center { text-align: center; }
.text-left { text-align: left; }
.text-right { text-align: right; }

.hidden { display: none; }
.invisible { visibility: hidden; }

.flex { display: flex; }
.grid { display: grid; }
.block { display: block; }
.inline { display: inline; }

.w-full { width: 100%; }
.h-full { height: 100%; }

.rounded { border-radius: var(--radius-md); }
.rounded-sm { border-radius: var(--radius-sm); }
.rounded-lg { border-radius: var(--radius-lg); }
.rounded-full { border-radius: var(--radius-full); }

.shadow-sm { box-shadow: var(--shadow-sm); }
.shadow-md { box-shadow: var(--shadow-md); }
.shadow-lg { box-shadow: var(--shadow-lg); }

.transition { transition: all var(--transition-normal); }
"#.to_string()
    }
}
