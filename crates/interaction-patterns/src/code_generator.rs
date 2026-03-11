//! 模式代码生成器

use crate::*;
use anyhow::Result;

/// 模式代码生成器
pub struct PatternCodeGenerator {
    library: InteractionPatternLibrary,
}

impl PatternCodeGenerator {
    pub fn new() -> Self {
        Self {
            library: InteractionPatternLibrary::new(),
        }
    }

    /// 生成模式代码
    pub fn generate(
        &self,
        pattern: &InteractionPattern,
        language: CodeLanguage,
    ) -> Result<GeneratedCode> {
        self.library
            .generate_pattern_code(pattern.pattern_type.clone(), pattern, language)
    }

    /// 批量生成代码
    pub fn generate_all(
        &self,
        patterns: &[InteractionPattern],
        language: CodeLanguage,
    ) -> Vec<Result<GeneratedCode>> {
        patterns
            .iter()
            .map(|p| self.generate(p, language.clone()))
            .collect()
    }

    /// 生成组件库
    pub fn generate_component_library(
        &self,
        patterns: &[InteractionPattern],
        language: CodeLanguage,
    ) -> Result<ComponentLibrary> {
        let mut components = Vec::new();
        let mut css_parts = Vec::new();
        let mut test_parts = Vec::new();

        for pattern in patterns {
            match self.generate(pattern, language.clone()) {
                Ok(code) => {
                    if let Some(css) = &code.css {
                        css_parts.push(css.clone());
                    }
                    if let Some(tests) = &code.tests {
                        test_parts.push(tests.clone());
                    }
                    components.push(code);
                }
                Err(e) => {
                    eprintln!(
                        "Failed to generate code for {:?}: {}",
                        pattern.pattern_type, e
                    );
                }
            }
        }

        let index = self.generate_index(&components, language);

        Ok(ComponentLibrary {
            components,
            css: css_parts.join("\n\n"),
            tests: test_parts.join("\n\n"),
            index,
        })
    }

    fn generate_index(&self, components: &[GeneratedCode], language: CodeLanguage) -> String {
        let mut index = String::new();

        match language {
            CodeLanguage::TypeScript | CodeLanguage::JavaScript => {
                for component in components {
                    index.push_str(&format!(
                        "export {{ {} }} from './{}';\n",
                        component.component_name,
                        component.component_name.to_lowercase()
                    ));
                }
            }
            CodeLanguage::React => {
                for component in components {
                    index.push_str(&format!(
                        "export {{ {} }} from './{}';\n",
                        component.component_name, component.component_name
                    ));
                }
            }
            _ => {}
        }

        index
    }
}

impl Default for PatternCodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// 组件库
#[derive(Debug, Clone)]
pub struct ComponentLibrary {
    pub components: Vec<GeneratedCode>,
    pub css: String,
    pub tests: String,
    pub index: String,
}

impl ComponentLibrary {
    /// 保存到目录
    pub fn save_to_directory(&self, path: &str) -> Result<()> {
        use std::fs;

        // 创建目录
        fs::create_dir_all(path)?;
        fs::create_dir_all(format!("{}/components", path))?;
        fs::create_dir_all(format!("{}/styles", path))?;
        fs::create_dir_all(format!("{}/tests", path))?;

        // 保存组件
        for component in &self.components {
            let extension = match component.language {
                CodeLanguage::TypeScript => "ts",
                CodeLanguage::JavaScript => "js",
                CodeLanguage::React => "tsx",
                CodeLanguage::Vue => "vue",
                CodeLanguage::Svelte => "svelte",
                CodeLanguage::Rust => "rs",
            };

            let filename = format!(
                "{}/components/{}.{}",
                path,
                component.component_name.to_lowercase(),
                extension
            );
            fs::write(&filename, &component.code)?;
        }

        // 保存CSS
        fs::write(format!("{}/styles/index.css", path), &self.css)?;

        // 保存测试
        fs::write(format!("{}/tests/index.test.ts", path), &self.tests)?;

        // 保存索引
        fs::write(format!("{}/index.ts", path), &self.index)?;

        Ok(())
    }
}
