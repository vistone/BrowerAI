/// ES Module Support and Modern Import/Export Features
///
/// Implements ES6+ module system including:
/// - Static import/export statements
/// - Dynamic import() expressions
/// - Top-level await support
/// - Module resolution and loading
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// ES Module parser and resolver
#[derive(Debug, Clone)]
pub struct ESModuleParser {
    /// Module cache to avoid re-parsing
    module_cache: HashMap<String, ParsedModule>,
    /// Resolved module paths
    resolved_paths: HashMap<String, PathBuf>,
    /// Base directory for module resolution
    base_dir: Option<PathBuf>,
}

/// Parsed ES Module with import/export information
#[derive(Debug, Clone)]
pub struct ParsedModule {
    /// Module identifier (URL or path)
    pub id: String,
    /// Import statements
    pub imports: Vec<ImportDeclaration>,
    /// Export statements
    pub exports: Vec<ExportDeclaration>,
    /// Whether module has top-level await
    pub has_top_level_await: bool,
    /// Whether this is a valid module
    pub is_valid: bool,
    /// Dynamic imports found in the module
    pub dynamic_imports: Vec<DynamicImport>,
}

/// Static import declaration
#[derive(Debug, Clone, PartialEq)]
pub struct ImportDeclaration {
    /// Import type
    pub import_type: ImportType,
    /// Module specifier (e.g., './utils', 'react')
    pub source: String,
    /// Imported bindings
    pub bindings: Vec<ImportBinding>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImportType {
    /// import x from 'mod'
    Default,
    /// import { a, b } from 'mod'
    Named,
    /// import * as x from 'mod'
    Namespace,
    /// import 'mod' (side effects only)
    SideEffect,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportBinding {
    /// Local name in current module
    pub local: String,
    /// Imported name from source module (if different)
    pub imported: Option<String>,
}

/// Export declaration
#[derive(Debug, Clone, PartialEq)]
pub struct ExportDeclaration {
    /// Export type
    pub export_type: ExportType,
    /// Exported bindings
    pub bindings: Vec<ExportBinding>,
    /// Re-export source (if applicable)
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExportType {
    /// export default x
    Default,
    /// export { a, b }
    Named,
    /// export * from 'mod'
    All,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExportBinding {
    /// Local name in current module
    pub local: String,
    /// Exported name (if different)
    pub exported: Option<String>,
}

/// Dynamic import expression
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicImport {
    /// Module specifier (can be a template string or expression)
    pub source: String,
    /// Whether the import is conditional
    pub is_conditional: bool,
}

impl ESModuleParser {
    /// Create a new ES Module parser
    pub fn new() -> Self {
        Self {
            module_cache: HashMap::new(),
            resolved_paths: HashMap::new(),
            base_dir: None,
        }
    }

    /// Set base directory for module resolution
    pub fn set_base_dir(&mut self, dir: PathBuf) {
        self.base_dir = Some(dir);
    }

    /// Parse ES Module from source code
    pub fn parse(&mut self, source: &str, module_id: &str) -> ParsedModule {
        // Check cache first
        if let Some(cached) = self.module_cache.get(module_id) {
            return cached.clone();
        }

        let mut module = ParsedModule {
            id: module_id.to_string(),
            imports: Vec::new(),
            exports: Vec::new(),
            has_top_level_await: false,
            is_valid: true,
            dynamic_imports: Vec::new(),
        };

        // Parse imports
        module.imports = self.parse_imports(source);

        // Parse exports
        module.exports = self.parse_exports(source);

        // Check for top-level await
        module.has_top_level_await = self.detect_top_level_await(source);

        // Parse dynamic imports
        module.dynamic_imports = self.parse_dynamic_imports(source);

        // Cache the module
        self.module_cache
            .insert(module_id.to_string(), module.clone());

        module
    }

    /// Parse import declarations from source
    fn parse_imports(&self, source: &str) -> Vec<ImportDeclaration> {
        let mut imports = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        for line in lines {
            let line = line.trim();

            // Skip comments
            if line.starts_with("//") || line.starts_with("/*") {
                continue;
            }

            if line.starts_with("import ") {
                if let Some(import) = self.parse_import_line(line) {
                    imports.push(import);
                }
            }
        }

        imports
    }

    /// Parse a single import line
    fn parse_import_line(&self, line: &str) -> Option<ImportDeclaration> {
        let line = line.trim();

        // Side effect import: import 'module'
        if line.contains("import '") || line.contains("import \"") {
            let start = line.find(|c| c == '\'' || c == '"')?;
            let end = line[start + 1..].find(|c| c == '\'' || c == '"')? + start + 1;
            let source = line[start + 1..end].to_string();

            return Some(ImportDeclaration {
                import_type: ImportType::SideEffect,
                source,
                bindings: Vec::new(),
            });
        }

        // Default import: import x from 'module'
        if line.contains(" from ") && !line.contains("{") && !line.contains("*") {
            let parts: Vec<&str> = line.split(" from ").collect();
            if parts.len() == 2 {
                let name = parts[0]
                    .trim()
                    .strip_prefix("import ")
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let source = parts[1]
                    .trim()
                    .trim_matches(|c| c == '\'' || c == '"' || c == ';')
                    .to_string();

                return Some(ImportDeclaration {
                    import_type: ImportType::Default,
                    source,
                    bindings: vec![ImportBinding {
                        local: name,
                        imported: None,
                    }],
                });
            }
        }

        // Namespace import: import * as x from 'module'
        if line.contains("* as ") {
            let parts: Vec<&str> = line.split(" from ").collect();
            if parts.len() == 2 {
                let name = parts[0]
                    .trim()
                    .strip_prefix("import * as ")
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let source = parts[1]
                    .trim()
                    .trim_matches(|c| c == '\'' || c == '"' || c == ';')
                    .to_string();

                return Some(ImportDeclaration {
                    import_type: ImportType::Namespace,
                    source,
                    bindings: vec![ImportBinding {
                        local: name,
                        imported: None,
                    }],
                });
            }
        }

        // Named import: import { a, b } from 'module'
        if line.contains("{") && line.contains("}") && line.contains(" from ") {
            let parts: Vec<&str> = line.split(" from ").collect();
            if parts.len() == 2 {
                let bindings_str = parts[0]
                    .trim()
                    .strip_prefix("import ")
                    .unwrap_or("")
                    .trim()
                    .trim_matches(|c| c == '{' || c == '}')
                    .trim();

                let source = parts[1]
                    .trim()
                    .trim_matches(|c| c == '\'' || c == '"' || c == ';')
                    .to_string();

                let mut bindings = Vec::new();
                for binding in bindings_str.split(',') {
                    let binding = binding.trim();
                    if binding.contains(" as ") {
                        let parts: Vec<&str> = binding.split(" as ").collect();
                        if parts.len() == 2 {
                            bindings.push(ImportBinding {
                                local: parts[1].trim().to_string(),
                                imported: Some(parts[0].trim().to_string()),
                            });
                        }
                    } else {
                        bindings.push(ImportBinding {
                            local: binding.to_string(),
                            imported: None,
                        });
                    }
                }

                return Some(ImportDeclaration {
                    import_type: ImportType::Named,
                    source,
                    bindings,
                });
            }
        }

        None
    }

    /// Parse export declarations from source
    fn parse_exports(&self, source: &str) -> Vec<ExportDeclaration> {
        let mut exports = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        for line in lines {
            let line = line.trim();

            if line.starts_with("export ") {
                if let Some(export) = self.parse_export_line(line) {
                    exports.push(export);
                }
            }
        }

        exports
    }

    /// Parse a single export line
    fn parse_export_line(&self, line: &str) -> Option<ExportDeclaration> {
        let line = line.trim();

        // Default export: export default x
        if line.starts_with("export default ") {
            let name = line
                .strip_prefix("export default ")
                .unwrap_or("")
                .trim()
                .trim_end_matches(';')
                .to_string();

            return Some(ExportDeclaration {
                export_type: ExportType::Default,
                bindings: vec![ExportBinding {
                    local: name,
                    exported: None,
                }],
                source: None,
            });
        }

        // Re-export all: export * from 'module'
        if line.contains("export * from ") {
            let source = line
                .split(" from ")
                .nth(1)?
                .trim()
                .trim_matches(|c| c == '\'' || c == '"' || c == ';')
                .to_string();

            return Some(ExportDeclaration {
                export_type: ExportType::All,
                bindings: Vec::new(),
                source: Some(source),
            });
        }

        // Named export: export { a, b }
        if line.contains("{") && line.contains("}") {
            let has_from = line.contains(" from ");
            let source = if has_from {
                let parts: Vec<&str> = line.split(" from ").collect();
                Some(
                    parts[1]
                        .trim()
                        .trim_matches(|c| c == '\'' || c == '"' || c == ';')
                        .to_string(),
                )
            } else {
                None
            };

            let bindings_part = if has_from {
                line.split(" from ").next()?
            } else {
                line
            };

            let bindings_str = bindings_part
                .trim()
                .strip_prefix("export ")
                .unwrap_or("")
                .trim()
                .trim_matches(|c| c == '{' || c == '}')
                .trim();

            let mut bindings = Vec::new();
            for binding in bindings_str.split(',') {
                let binding = binding.trim();
                if binding.contains(" as ") {
                    let parts: Vec<&str> = binding.split(" as ").collect();
                    if parts.len() == 2 {
                        bindings.push(ExportBinding {
                            local: parts[0].trim().to_string(),
                            exported: Some(parts[1].trim().to_string()),
                        });
                    }
                } else {
                    bindings.push(ExportBinding {
                        local: binding.to_string(),
                        exported: None,
                    });
                }
            }

            return Some(ExportDeclaration {
                export_type: ExportType::Named,
                bindings,
                source,
            });
        }

        None
    }

    /// Detect top-level await in module
    fn detect_top_level_await(&self, source: &str) -> bool {
        // Simple detection: look for await outside of function bodies
        let lines: Vec<&str> = source.lines().collect();
        let mut in_function = false;
        let mut brace_count = 0;

        for line in lines {
            let line = line.trim();

            // Track function scope
            if line.contains("function ") || line.contains("async ") {
                in_function = true;
            }

            brace_count += line.matches('{').count() as i32;
            brace_count -= line.matches('}').count() as i32;

            if brace_count <= 0 {
                in_function = false;
            }

            // Check for await outside function
            if !in_function && line.contains("await ") {
                return true;
            }
        }

        false
    }

    /// Parse dynamic imports
    fn parse_dynamic_imports(&self, source: &str) -> Vec<DynamicImport> {
        let mut dynamic_imports = Vec::new();

        // Look for import() calls
        for (i, _) in source.match_indices("import(") {
            let rest = &source[i + 7..];
            if let Some(end) = rest.find(')') {
                let import_arg = rest[..end].trim();
                let import_source = import_arg.trim_matches(|c| c == '\'' || c == '"' || c == '`');

                // Check if it's conditional (inside if/then/etc)
                let before = &source[..i];
                let is_conditional = before.contains("if ") || before.contains("? ");

                dynamic_imports.push(DynamicImport {
                    source: import_source.to_string(),
                    is_conditional,
                });
            }
        }

        dynamic_imports
    }

    /// Resolve module specifier to a path
    pub fn resolve_module(&mut self, specifier: &str, from_module: &str) -> Option<PathBuf> {
        // Check cache
        let cache_key = format!("{}::{}", from_module, specifier);
        if let Some(resolved) = self.resolved_paths.get(&cache_key) {
            return Some(resolved.clone());
        }

        // Relative path resolution
        if specifier.starts_with("./") || specifier.starts_with("../") {
            if let Some(base) = &self.base_dir {
                let from_path = Path::new(from_module);
                let from_dir = from_path.parent().unwrap_or(Path::new(""));
                let resolved = base.join(from_dir).join(specifier);

                self.resolved_paths
                    .insert(cache_key, resolved.clone());
                return Some(resolved);
            }
        }

        // Bare specifier (e.g., 'react', 'lodash')
        // In a real implementation, this would check node_modules
        None
    }

    /// Get all imported modules from a module
    pub fn get_dependencies(&self, module_id: &str) -> Vec<String> {
        if let Some(module) = self.module_cache.get(module_id) {
            module.imports.iter().map(|i| i.source.clone()).collect()
        } else {
            Vec::new()
        }
    }

    /// Check if a module is valid ES6 module
    pub fn is_valid_module(&self, source: &str) -> bool {
        // Basic validation: has import or export statements
        source.contains("import ") || source.contains("export ")
    }

    /// Get module graph (all transitive dependencies)
    pub fn get_module_graph(&self, entry_module: &str) -> HashMap<String, Vec<String>> {
        let mut graph = HashMap::new();
        let mut visited = HashSet::new();
        let mut to_visit = vec![entry_module.to_string()];

        while let Some(module_id) = to_visit.pop() {
            if visited.contains(&module_id) {
                continue;
            }

            visited.insert(module_id.clone());
            let deps = self.get_dependencies(&module_id);

            for dep in &deps {
                if !visited.contains(dep) {
                    to_visit.push(dep.clone());
                }
            }

            graph.insert(module_id, deps);
        }

        graph
    }

    /// Clear the module cache
    pub fn clear_cache(&mut self) {
        self.module_cache.clear();
        self.resolved_paths.clear();
    }
}

impl Default for ESModuleParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_default_import() {
        let mut parser = ESModuleParser::new();
        let source = "import React from 'react';";
        let module = parser.parse(source, "test.js");

        assert_eq!(module.imports.len(), 1);
        assert_eq!(module.imports[0].import_type, ImportType::Default);
        assert_eq!(module.imports[0].source, "react");
        assert_eq!(module.imports[0].bindings[0].local, "React");
    }

    #[test]
    fn test_parse_named_import() {
        let mut parser = ESModuleParser::new();
        let source = "import { useState, useEffect } from 'react';";
        let module = parser.parse(source, "test.js");

        assert_eq!(module.imports.len(), 1);
        assert_eq!(module.imports[0].import_type, ImportType::Named);
        assert_eq!(module.imports[0].bindings.len(), 2);
        assert_eq!(module.imports[0].bindings[0].local, "useState");
        assert_eq!(module.imports[0].bindings[1].local, "useEffect");
    }

    #[test]
    fn test_parse_namespace_import() {
        let mut parser = ESModuleParser::new();
        let source = "import * as Utils from './utils';";
        let module = parser.parse(source, "test.js");

        assert_eq!(module.imports.len(), 1);
        assert_eq!(module.imports[0].import_type, ImportType::Namespace);
        assert_eq!(module.imports[0].source, "./utils");
        assert_eq!(module.imports[0].bindings[0].local, "Utils");
    }

    #[test]
    fn test_parse_side_effect_import() {
        let mut parser = ESModuleParser::new();
        let source = "import './styles.css';";
        let module = parser.parse(source, "test.js");

        assert_eq!(module.imports.len(), 1);
        assert_eq!(module.imports[0].import_type, ImportType::SideEffect);
        assert_eq!(module.imports[0].source, "./styles.css");
    }

    #[test]
    fn test_parse_default_export() {
        let mut parser = ESModuleParser::new();
        let source = "export default function App() {}";
        let module = parser.parse(source, "test.js");

        assert_eq!(module.exports.len(), 1);
        assert_eq!(module.exports[0].export_type, ExportType::Default);
    }

    #[test]
    fn test_parse_named_export() {
        let mut parser = ESModuleParser::new();
        let source = "export { foo, bar };";
        let module = parser.parse(source, "test.js");

        assert_eq!(module.exports.len(), 1);
        assert_eq!(module.exports[0].export_type, ExportType::Named);
        assert_eq!(module.exports[0].bindings.len(), 2);
    }

    #[test]
    fn test_parse_re_export() {
        let mut parser = ESModuleParser::new();
        let source = "export * from './utils';";
        let module = parser.parse(source, "test.js");

        assert_eq!(module.exports.len(), 1);
        assert_eq!(module.exports[0].export_type, ExportType::All);
        assert_eq!(module.exports[0].source, Some("./utils".to_string()));
    }

    #[test]
    fn test_detect_top_level_await() {
        let mut parser = ESModuleParser::new();
        let source = "const data = await fetch('/api'); export default data;";
        let module = parser.parse(source, "test.js");

        assert!(module.has_top_level_await);
    }

    #[test]
    fn test_parse_dynamic_import() {
        let mut parser = ESModuleParser::new();
        let source = "const module = await import('./dynamic.js');";
        let module = parser.parse(source, "test.js");

        assert_eq!(module.dynamic_imports.len(), 1);
        assert_eq!(module.dynamic_imports[0].source, "./dynamic.js");
    }

    #[test]
    fn test_is_valid_module() {
        let parser = ESModuleParser::new();

        assert!(parser.is_valid_module("import x from 'y';"));
        assert!(parser.is_valid_module("export const a = 1;"));
        assert!(!parser.is_valid_module("const a = 1;"));
    }

    #[test]
    fn test_module_cache() {
        let mut parser = ESModuleParser::new();
        let source = "import x from 'y';";

        // Parse once
        let module1 = parser.parse(source, "test.js");

        // Parse again - should come from cache
        let module2 = parser.parse(source, "test.js");

        assert_eq!(module1.id, module2.id);
    }

    #[test]
    fn test_get_dependencies() {
        let mut parser = ESModuleParser::new();
        let source = "import a from 'moduleA';\nimport b from 'moduleB';";
        parser.parse(source, "test.js");

        let deps = parser.get_dependencies("test.js");
        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&"moduleA".to_string()));
        assert!(deps.contains(&"moduleB".to_string()));
    }
}
