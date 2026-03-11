//! 智能代码生成器
//! 基于学习到的行为和视觉信息生成完整网站代码

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod component_builder;
pub mod project_scaffolder;
pub mod script_generator;
pub mod style_generator;
pub mod template_engine;

pub use component_builder::ComponentBuilder;
pub use project_scaffolder::ProjectScaffolder;
pub use script_generator::ScriptGenerator;
pub use style_generator::StyleGenerator;
pub use template_engine::TemplateEngine;

/// 生成配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub project_name: String,
    pub output_dir: String,
    pub target_framework: Framework,
    pub include_tests: bool,
    pub include_documentation: bool,
    pub optimization_level: OptimizationLevel,
    pub accessibility: AccessibilityLevel,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            project_name: "generated-website".to_string(),
            output_dir: "./output".to_string(),
            target_framework: Framework::React,
            include_tests: true,
            include_documentation: true,
            optimization_level: OptimizationLevel::Standard,
            accessibility: AccessibilityLevel::WCAG_AA,
        }
    }
}

/// 目标框架
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Framework {
    VanillaJS,
    React,
    Vue,
    Svelte,
    NextJS,
    Nuxt,
}

/// 优化级别
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizationLevel {
    None,
    Standard,
    Aggressive,
}

/// 无障碍级别
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(non_camel_case_types)]
pub enum AccessibilityLevel {
    None,
    WCAG_A,
    WCAG_AA,
    WCAG_AAA,
}

/// 生成的项目
#[derive(Debug, Clone)]
pub struct GeneratedProject {
    pub name: String,
    pub framework: Framework,
    pub files: Vec<GeneratedFile>,
    pub dependencies: Vec<Dependency>,
    pub scripts: HashMap<String, String>,
}

/// 生成的文件
#[derive(Debug, Clone)]
pub struct GeneratedFile {
    pub path: String,
    pub content: String,
    pub file_type: FileType,
}

/// 文件类型
#[derive(Debug, Clone, PartialEq)]
pub enum FileType {
    Component,
    Style,
    Script,
    Config,
    Documentation,
    Test,
    Asset,
}

/// 依赖
#[derive(Debug, Clone)]
pub struct Dependency {
    pub name: String,
    pub version: String,
    pub is_dev: bool,
}

/// 代码生成引擎
pub struct CodeGenerationEngine {
    config: GenerationConfig,
    _template_engine: TemplateEngine,
    component_builder: ComponentBuilder,
    style_generator: StyleGenerator,
    script_generator: ScriptGenerator,
    project_scaffolder: ProjectScaffolder,
}

impl CodeGenerationEngine {
    pub fn new(config: GenerationConfig) -> Self {
        Self {
            config: config.clone(),
            _template_engine: TemplateEngine::new(&config),
            component_builder: ComponentBuilder::new(&config),
            style_generator: StyleGenerator::new(&config),
            script_generator: ScriptGenerator::new(&config),
            project_scaffolder: ProjectScaffolder::new(&config),
        }
    }

    /// 生成完整项目
    pub async fn generate_project(
        &self,
        analysis: &visual_learner::VisualAnalysis,
        behaviors: &[interaction_patterns::InteractionPattern],
    ) -> Result<GeneratedProject> {
        log::info!("Starting project generation: {}", self.config.project_name);

        // 1. 搭建项目结构
        let mut project = self.project_scaffolder.scaffold().await?;

        // 2. 生成组件
        let components = self
            .component_builder
            .build_components(analysis, behaviors)
            .await?;
        project.files.extend(components);

        // 3. 生成样式
        let styles = self.style_generator.generate_styles(analysis).await?;
        project.files.extend(styles);

        // 4. 生成脚本
        let scripts = self.script_generator.generate_scripts(behaviors).await?;
        project.files.extend(scripts);

        // 5. 生成入口文件
        let entry = self.generate_entry_file(analysis).await?;
        project.files.push(entry);

        // 6. 生成配置文件
        let configs = self.generate_config_files().await?;
        project.files.extend(configs);

        // 7. 如果需要，生成测试
        if self.config.include_tests {
            let tests = self.generate_tests(&project.files).await?;
            project.files.extend(tests);
        }

        // 8. 如果需要，生成文档
        if self.config.include_documentation {
            let docs = self.generate_documentation(analysis, behaviors).await?;
            project.files.extend(docs);
        }

        log::info!(
            "Project generation completed: {} files",
            project.files.len()
        );

        Ok(project)
    }

    /// 生成入口文件
    async fn generate_entry_file(
        &self,
        analysis: &visual_learner::VisualAnalysis,
    ) -> Result<GeneratedFile> {
        let content = match self.config.target_framework {
            Framework::React => self.generate_react_entry(analysis),
            Framework::Vue => self.generate_vue_entry(analysis),
            Framework::Svelte => self.generate_svelte_entry(analysis),
            Framework::VanillaJS => self.generate_vanilla_entry(analysis),
            _ => anyhow::bail!("Framework not yet supported"),
        }?;

        let path = match self.config.target_framework {
            Framework::React | Framework::NextJS => "src/App.tsx",
            Framework::Vue | Framework::Nuxt => "src/App.vue",
            Framework::Svelte => "src/App.svelte",
            Framework::VanillaJS => "src/main.js",
        };

        Ok(GeneratedFile {
            path: path.to_string(),
            content,
            file_type: FileType::Component,
        })
    }

    fn generate_react_entry(&self, analysis: &visual_learner::VisualAnalysis) -> Result<String> {
        let mut imports = String::from("import React from 'react';\n");
        imports.push_str("import './styles/index.css';\n");

        let mut components = String::new();
        for comp in &analysis.components {
            let comp_name = format!("{:?}", comp.component_type);
            components.push_str(&format!("      <{} key=\"{}\" />\n", comp_name, comp.id));
        }

        Ok(format!(
            r#"{}

export function App() {{
  return (
    <div className="app">
{}
    </div>
  );
}}

export default App;
"#,
            imports, components
        ))
    }

    fn generate_vue_entry(&self, _analysis: &visual_learner::VisualAnalysis) -> Result<String> {
        Ok(r#"<template>
  <div class="app">
    <router-view />
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
@import './styles/index.css';
</style>
"#
        .to_string())
    }

    fn generate_svelte_entry(&self, _analysis: &visual_learner::VisualAnalysis) -> Result<String> {
        Ok(r#"<script>
  import './styles/index.css';
</script>

<main class="app">
  <slot />
</main>
"#
        .to_string())
    }

    fn generate_vanilla_entry(&self, _analysis: &visual_learner::VisualAnalysis) -> Result<String> {
        Ok(r#"import './styles/index.css';

function init() {
  const app = document.getElementById('app');
  if (!app) return;

  // Initialize components
  console.log('App initialized');
}

document.addEventListener('DOMContentLoaded', init);
"#
        .to_string())
    }

    /// 生成配置文件
    async fn generate_config_files(&self) -> Result<Vec<GeneratedFile>> {
        let mut configs = Vec::new();

        if let Framework::React = self.config.target_framework {
            configs.push(GeneratedFile {
                path: "package.json".to_string(),
                content: self.generate_package_json(),
                file_type: FileType::Config,
            });
            configs.push(GeneratedFile {
                path: "tsconfig.json".to_string(),
                content: self.generate_tsconfig(),
                file_type: FileType::Config,
            });
            configs.push(GeneratedFile {
                path: "vite.config.ts".to_string(),
                content: self.generate_vite_config(),
                file_type: FileType::Config,
            });
        }

        Ok(configs)
    }

    fn generate_package_json(&self) -> String {
        let dependencies = match self.config.target_framework {
            Framework::React => {
                r#"
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.0.0",
    "typescript": "^5.0.0",
    "vite": "^5.0.0"
  }"#
            }
            _ => "",
        };

        format!(
            r#"{{
  "name": "{}",
  "version": "1.0.0",
  "type": "module",
  "scripts": {{
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest"
  }},{}
}}
"#,
            self.config.project_name, dependencies
        )
    }

    fn generate_tsconfig(&self) -> String {
        r#"{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
"#
        .to_string()
    }

    fn generate_vite_config(&self) -> String {
        r#"import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
"#
        .to_string()
    }

    /// 生成测试
    async fn generate_tests(&self, files: &[GeneratedFile]) -> Result<Vec<GeneratedFile>> {
        let mut tests = Vec::new();

        for file in files {
            if file.file_type == FileType::Component {
                let test_content = self.generate_component_test(&file.path);
                tests.push(GeneratedFile {
                    path: file
                        .path
                        .replace("src/", "src/__tests__/")
                        .replace(".tsx", ".test.tsx"),
                    content: test_content,
                    file_type: FileType::Test,
                });
            }
        }

        Ok(tests)
    }

    fn generate_component_test(&self, component_path: &str) -> String {
        let component_name = component_path
            .split('/')
            .next_back()
            .unwrap_or("Component")
            .replace(".tsx", "");

        format!(
            r#"import {{ render, screen }} from '@testing-library/react';
import {{ {} }} from '../{}';

describe('{}', () => {{
  it('renders correctly', () => {{
    render(<{} />);
    expect(document.body).toBeInTheDocument();
  }});
}});
"#,
            component_name,
            component_path.replace("src/", "").replace(".tsx", ""),
            component_name,
            component_name
        )
    }

    /// 生成文档
    async fn generate_documentation(
        &self,
        analysis: &visual_learner::VisualAnalysis,
        behaviors: &[interaction_patterns::InteractionPattern],
    ) -> Result<Vec<GeneratedFile>> {
        let mut docs = Vec::new();

        // README
        let readme = self.generate_readme(analysis, behaviors);
        docs.push(GeneratedFile {
            path: "README.md".to_string(),
            content: readme,
            file_type: FileType::Documentation,
        });

        // API文档
        let api_doc = self.generate_api_documentation(analysis);
        docs.push(GeneratedFile {
            path: "docs/API.md".to_string(),
            content: api_doc,
            file_type: FileType::Documentation,
        });

        Ok(docs)
    }

    fn generate_readme(
        &self,
        analysis: &visual_learner::VisualAnalysis,
        behaviors: &[interaction_patterns::InteractionPattern],
    ) -> String {
        format!(
            r#"# {}

Generated website based on visual analysis and behavior learning.

## Overview

- **Components**: {}
- **Layout Type**: {:?}
- **Interaction Patterns**: {}

## Getting Started

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

## Project Structure

```
src/
├── components/    # UI components
├── styles/        # CSS styles
├── hooks/         # Custom React hooks
└── utils/         # Utility functions
```

## Design System

### Colors
{}

### Typography
- Font Sizes: {:?}
- Font Weights: {:?}

## License

MIT
"#,
            self.config.project_name,
            analysis.components.len(),
            analysis.layout.layout_type,
            behaviors.len(),
            self.format_color_scheme(&analysis.color_scheme),
            analysis.typography.font_sizes,
            analysis.typography.font_weights
        )
    }

    fn format_color_scheme(&self, scheme: &visual_learner::ColorScheme) -> String {
        let mut lines = Vec::new();

        if let Some(ref c) = scheme.primary {
            lines.push(format!("- Primary: {}", c.to_hex()));
        }
        if let Some(ref c) = scheme.secondary {
            lines.push(format!("- Secondary: {}", c.to_hex()));
        }
        if let Some(ref c) = scheme.background {
            lines.push(format!("- Background: {}", c.to_hex()));
        }

        lines.join("\n")
    }

    fn generate_api_documentation(&self, analysis: &visual_learner::VisualAnalysis) -> String {
        let mut doc = String::from("# Component API\n\n");

        for component in &analysis.components {
            doc.push_str(&format!("## {:?}\n\n", component.component_type));
            doc.push_str(&format!("- **ID**: {}\n", component.id));
            doc.push_str(&format!(
                "- **Position**: ({}, {})\n",
                component.bounding_box.x, component.bounding_box.y
            ));
            doc.push_str(&format!(
                "- **Size**: {}x{}\n\n",
                component.bounding_box.width, component.bounding_box.height
            ));
        }

        doc
    }

    /// 保存项目到磁盘
    pub async fn save_project(&self, project: &GeneratedProject) -> Result<()> {
        use tokio::fs;

        let base_path = std::path::Path::new(&self.config.output_dir);

        // 创建目录结构
        for file in &project.files {
            let full_path = base_path.join(&file.path);
            if let Some(parent) = full_path.parent() {
                fs::create_dir_all(parent).await?;
            }
            fs::write(&full_path, &file.content).await?;
        }

        log::info!("Project saved to: {}", base_path.display());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.project_name, "generated-website");
        assert_eq!(config.output_dir, "./output");
        assert!(config.include_tests);
        assert!(config.include_documentation);
    }

    #[test]
    fn test_framework_variants() {
        let frameworks = vec![
            Framework::VanillaJS,
            Framework::React,
            Framework::Vue,
            Framework::Svelte,
            Framework::NextJS,
            Framework::Nuxt,
        ];
        assert_eq!(frameworks.len(), 6);
    }

    #[test]
    fn test_optimization_level_variants() {
        assert_eq!(OptimizationLevel::None, OptimizationLevel::None);
        assert_ne!(OptimizationLevel::None, OptimizationLevel::Standard);
        assert_ne!(OptimizationLevel::Standard, OptimizationLevel::Aggressive);
    }

    #[test]
    fn test_accessibility_level_variants() {
        let levels = vec![
            AccessibilityLevel::None,
            AccessibilityLevel::WCAG_A,
            AccessibilityLevel::WCAG_AA,
            AccessibilityLevel::WCAG_AAA,
        ];
        assert_eq!(levels.len(), 4);
    }

    #[test]
    fn test_generated_file_creation() {
        let file = GeneratedFile {
            path: "src/App.tsx".to_string(),
            content: "export function App() {{}}".to_string(),
            file_type: FileType::Component,
        };
        assert_eq!(file.path, "src/App.tsx");
        assert_eq!(file.file_type, FileType::Component);
    }

    #[test]
    fn test_file_type_variants() {
        let types = vec![
            FileType::Component,
            FileType::Style,
            FileType::Script,
            FileType::Config,
            FileType::Documentation,
            FileType::Test,
            FileType::Asset,
        ];
        assert_eq!(types.len(), 7);
    }

    #[test]
    fn test_dependency_creation() {
        let dep = Dependency {
            name: "react".to_string(),
            version: "^18.2.0".to_string(),
            is_dev: false,
        };
        assert_eq!(dep.name, "react");
        assert!(!dep.is_dev);
    }

    #[test]
    fn test_generated_project_creation() {
        let project = GeneratedProject {
            name: "test-project".to_string(),
            framework: Framework::React,
            files: vec![],
            dependencies: vec![],
            scripts: HashMap::new(),
        };
        assert_eq!(project.name, "test-project");
    }
}
