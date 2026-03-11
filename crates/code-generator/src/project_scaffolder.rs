//! 项目脚手架

use crate::*;
use anyhow::Result;

pub struct ProjectScaffolder {
    config: GenerationConfig,
}

impl ProjectScaffolder {
    pub fn new(config: &GenerationConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub async fn scaffold(&self) -> Result<GeneratedProject> {
        let mut project = GeneratedProject {
            name: self.config.project_name.clone(),
            framework: self.config.target_framework.clone(),
            files: Vec::new(),
            dependencies: self.get_dependencies(),
            scripts: self.get_scripts(),
        };

        // 创建目录结构
        let directories = self.create_directory_structure();
        project.files.extend(directories);

        // 创建配置文件
        let configs = self.create_config_files().await?;
        project.files.extend(configs);

        Ok(project)
    }

    fn get_dependencies(&self) -> Vec<Dependency> {
        let mut deps = vec![];

        if let Framework::React = self.config.target_framework {
            deps.push(Dependency {
                name: "react".to_string(),
                version: "^18.2.0".to_string(),
                is_dev: false,
            });
            deps.push(Dependency {
                name: "react-dom".to_string(),
                version: "^18.2.0".to_string(),
                is_dev: false,
            });
            deps.push(Dependency {
                name: "@types/react".to_string(),
                version: "^18.2.0".to_string(),
                is_dev: true,
            });
            deps.push(Dependency {
                name: "@types/react-dom".to_string(),
                version: "^18.2.0".to_string(),
                is_dev: true,
            });
            deps.push(Dependency {
                name: "@vitejs/plugin-react".to_string(),
                version: "^4.0.0".to_string(),
                is_dev: true,
            });
            deps.push(Dependency {
                name: "vite".to_string(),
                version: "^5.0.0".to_string(),
                is_dev: true,
            });
            deps.push(Dependency {
                name: "typescript".to_string(),
                version: "^5.0.0".to_string(),
                is_dev: true,
            });
        }

        // 测试依赖
        if self.config.include_tests {
            deps.push(Dependency {
                name: "vitest".to_string(),
                version: "^1.0.0".to_string(),
                is_dev: true,
            });
            deps.push(Dependency {
                name: "@testing-library/react".to_string(),
                version: "^14.0.0".to_string(),
                is_dev: true,
            });
        }

        deps
    }

    fn get_scripts(&self) -> HashMap<String, String> {
        let mut scripts = HashMap::new();

        scripts.insert("dev".to_string(), "vite".to_string());
        scripts.insert("build".to_string(), "tsc && vite build".to_string());
        scripts.insert("preview".to_string(), "vite preview".to_string());

        if self.config.include_tests {
            scripts.insert("test".to_string(), "vitest".to_string());
        }

        scripts
    }

    fn create_directory_structure(&self) -> Vec<GeneratedFile> {
        // 目录结构通过空文件表示
        vec![
            GeneratedFile {
                path: "src/components/.gitkeep".to_string(),
                content: String::new(),
                file_type: FileType::Config,
            },
            GeneratedFile {
                path: "src/hooks/.gitkeep".to_string(),
                content: String::new(),
                file_type: FileType::Config,
            },
            GeneratedFile {
                path: "src/utils/.gitkeep".to_string(),
                content: String::new(),
                file_type: FileType::Config,
            },
            GeneratedFile {
                path: "src/styles/.gitkeep".to_string(),
                content: String::new(),
                file_type: FileType::Config,
            },
            GeneratedFile {
                path: "src/behaviors/.gitkeep".to_string(),
                content: String::new(),
                file_type: FileType::Config,
            },
            GeneratedFile {
                path: "public/.gitkeep".to_string(),
                content: String::new(),
                file_type: FileType::Config,
            },
        ]
    }

    #[allow(clippy::vec_init_then_push)]
    async fn create_config_files(&self) -> Result<Vec<GeneratedFile>> {
        let mut files = Vec::new();

        // index.html
        files.push(GeneratedFile {
            path: "index.html".to_string(),
            content: self.generate_index_html(),
            file_type: FileType::Config,
        });

        // .gitignore
        files.push(GeneratedFile {
            path: ".gitignore".to_string(),
            content: self.generate_gitignore(),
            file_type: FileType::Config,
        });

        // .eslintrc
        files.push(GeneratedFile {
            path: ".eslintrc.json".to_string(),
            content: self.generate_eslint_config(),
            file_type: FileType::Config,
        });

        // .prettierrc
        files.push(GeneratedFile {
            path: ".prettierrc".to_string(),
            content: self.generate_prettier_config(),
            file_type: FileType::Config,
        });

        Ok(files)
    }

    fn generate_index_html(&self) -> String {
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="description" content="Generated website" />
    <title>{}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
"#,
            self.config.project_name
        )
    }

    fn generate_gitignore(&self) -> String {
        r#"# Dependencies
node_modules
.pnp
.pnp.js

# Build
dist
dist-ssr
*.local

# Editor
.vscode/*
!.vscode/extensions.json
.idea
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?

# OS
.DS_Store
Thumbs.db

# Logs
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*
lerna-debug.log*

# Testing
coverage
"#
        .to_string()
    }

    fn generate_eslint_config(&self) -> String {
        r#"{
  "root": true,
  "env": { "browser": true, "es2020": true },
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:react-hooks/recommended"
  ],
  "ignorePatterns": ["dist", ".eslintrc.json"],
  "parser": "@typescript-eslint/parser",
  "plugins": ["react-refresh"],
  "rules": {
    "react-refresh/only-export-components": [
      "warn",
      { "allowConstantExport": true }
    ]
  }
}
"#
        .to_string()
    }

    fn generate_prettier_config(&self) -> String {
        r#"{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 100
}
"#
        .to_string()
    }
}
