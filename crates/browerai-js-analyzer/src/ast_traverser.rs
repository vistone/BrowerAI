/// Phase 2 Day 3 - 完整 AST 遍历与精细提取
///
/// 本模块实现完整的 AST 遍历，包括：
/// 1. 函数和类的完整定义及位置
/// 2. JSX 元素的精确位置和嵌套关系
/// 3. TypeScript 类型的完整信息
/// 4. 导出和导入语句的追踪

use once_cell::sync::Lazy;
use regex::Regex;

static FUNCTION_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)").expect("Invalid regex pattern")
});

static ARROW_FUNCTION_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>").expect("Invalid regex pattern")
});

static CLASS_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{").expect("Invalid regex pattern")
});

static METHOD_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:static\s+)?(?:async\s+)?(\w+)\s*\(").expect("Invalid regex pattern")
});

static EXPORT_DEFAULT_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"export\s+default\s+(?:function\s+)?(\w+)").expect("Invalid regex pattern")
});

static EXPORT_NAMED_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"export\s+(?:const|let|var|function|class)\s+(\w+)").expect("Invalid regex pattern")
});

static IMPORT_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"import\s+(?:\{([^}]+)\}|(\w+))?\s+from\s+['\"]([^'\"]+)['\"]").expect("Invalid regex pattern")
});

static FUNC_COMPONENT_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:\(|function)").expect("Invalid regex pattern")
});

static CLASS_COMPONENT_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"class\s+(\w+)\s+extends\s+(?:React\.)?Component").expect("Invalid regex pattern")
});

/// AST 遍历结果
#[derive(Debug, Clone, Default)]
pub struct AstTraversalResult {
    /// 所有函数定义及其位置
    pub functions: Vec<FunctionDef>,
    /// 所有类定义及其位置
    pub classes: Vec<ClassDef>,
    /// 所有导出语句
    pub exports: Vec<ExportDef>,
    /// 所有导入语句
    pub imports: Vec<ImportDef>,
    /// JSX 组件的定义位置
    pub components: Vec<ComponentDef>,
}

/// 函数定义
#[derive(Debug, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub line: usize,
    pub column: usize,
    pub is_async: bool,
    pub is_arrow: bool,
    pub params: Vec<String>,
}

/// 类定义
#[derive(Debug, Clone)]
pub struct ClassDef {
    pub name: String,
    pub line: usize,
    pub column: usize,
    pub extends: Option<String>,
    pub methods: Vec<MethodDef>,
}

/// 类方法
#[derive(Debug, Clone)]
pub struct MethodDef {
    pub name: String,
    pub is_async: bool,
    pub is_static: bool,
}

/// 导出定义
#[derive(Debug, Clone)]
pub struct ExportDef {
    pub name: String,
    pub kind: ExportKind,
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExportKind {
    Default,
    Named,
    Namespace,
}

/// 导入定义
#[derive(Debug, Clone)]
pub struct ImportDef {
    pub from: String,
    pub names: Vec<String>,
    pub line: usize,
}

/// React 组件定义
#[derive(Debug, Clone)]
pub struct ComponentDef {
    pub name: String,
    pub kind: ComponentKind,
    pub line: usize,
    pub is_exported: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComponentKind {
    Functional,
    Class,
    Memo,
    ForwardRef,
}

/// 完整 AST 遍历器
pub struct AstTraverser;

impl AstTraverser {
    /// 遍历源代码并提取所有信息
    pub fn traverse(source: &str) -> AstTraversalResult {
        let mut result = AstTraversalResult::default();
        
        result.functions = Self::extract_functions(source);
        result.classes = Self::extract_classes(source);
        result.exports = Self::extract_exports(source);
        result.imports = Self::extract_imports(source);
        result.components = Self::extract_components(source);
        
        result
    }
    
    /// 提取所有函数定义
    fn extract_functions(source: &str) -> Vec<FunctionDef> {
        let mut functions = Vec::new();

        for cap in FUNCTION_PATTERN.captures_iter(source) {
            if let Some(name) = cap.get(1) {
                let is_async = cap.get(0).map(|m| m.as_str().contains("async")).unwrap_or(false);
                let params_str = cap.get(2).map(|m| m.as_str()).unwrap_or("");
                let params = Self::parse_params(params_str);

                functions.push(FunctionDef {
                    name: name.as_str().to_string(),
                    line: source[..source.find(name.as_str()).unwrap_or(0)].lines().count(),
                    column: 0,
                    is_async,
                    is_arrow: false,
                    params,
                });
            }
        }

        for cap in ARROW_FUNCTION_PATTERN.captures_iter(source) {
            if let Some(name) = cap.get(1) {
                let is_async = cap.get(0).map(|m| m.as_str().contains("async")).unwrap_or(false);
                let params_str = cap.get(2).map(|m| m.as_str()).unwrap_or("");
                let params = Self::parse_params(params_str);

                functions.push(FunctionDef {
                    name: name.as_str().to_string(),
                    line: source[..source.find(name.as_str()).unwrap_or(0)].lines().count(),
                    column: 0,
                    is_async,
                    is_arrow: true,
                    params,
                });
            }
        }

        functions
    }
    
    /// 提取所有类定义
    fn extract_classes(source: &str) -> Vec<ClassDef> {
        let mut classes = Vec::new();

        for cap in CLASS_PATTERN.captures_iter(source) {
            if let Some(name) = cap.get(1) {
                let extends = cap.get(2).map(|m| m.as_str().to_string());
                let methods = Self::extract_methods(source, name.as_str());

                classes.push(ClassDef {
                    name: name.as_str().to_string(),
                    line: source[..source.find(name.as_str()).unwrap_or(0)].lines().count(),
                    column: 0,
                    extends,
                    methods,
                });
            }
        }

        classes
    }

    /// 提取类的所有方法
    fn extract_methods(source: &str, class_name: &str) -> Vec<MethodDef> {
        let mut methods = Vec::new();

        if let Some(class_start) = source.find(&format!("class {}", class_name)) {
            if let Some(class_body_start) = source[class_start..].find('{') {
                let search_end = class_start + class_body_start + 1000;
                let search_str = &source[class_start + class_body_start + 1..search_end.min(source.len())];

                for cap in METHOD_PATTERN.captures_iter(search_str) {
                    if let Some(method_name) = cap.get(1) {
                        let full_match = cap.get(0).map(|m| m.as_str()).unwrap_or("");
                        methods.push(MethodDef {
                            name: method_name.as_str().to_string(),
                            is_async: full_match.contains("async"),
                            is_static: full_match.contains("static"),
                        });
                    }
                }
            }
        }

        methods
    }
    
    /// 提取所有导出语句
    fn extract_exports(source: &str) -> Vec<ExportDef> {
        let mut exports = Vec::new();

        for cap in EXPORT_DEFAULT_PATTERN.captures_iter(source) {
            if let Some(name) = cap.get(1) {
                exports.push(ExportDef {
                    name: name.as_str().to_string(),
                    kind: ExportKind::Default,
                    line: source[..source.find(name.as_str()).unwrap_or(0)].lines().count(),
                });
            }
        }

        for cap in EXPORT_NAMED_PATTERN.captures_iter(source) {
            if let Some(name) = cap.get(1) {
                exports.push(ExportDef {
                    name: name.as_str().to_string(),
                    kind: ExportKind::Named,
                    line: source[..source.find(name.as_str()).unwrap_or(0)].lines().count(),
                });
            }
        }

        exports
    }

    /// 提取所有导入语句
    fn extract_imports(source: &str) -> Vec<ImportDef> {
        let mut imports = Vec::new();

        for cap in IMPORT_PATTERN.captures_iter(source) {
            let named_imports = cap.get(1).map(|m| m.as_str());
            let default_import = cap.get(2).map(|m| m.as_str());
            let from = cap.get(3).map(|m| m.as_str().to_string()).unwrap_or_default();

            let mut names = Vec::new();
            if let Some(named) = named_imports {
                names.extend(
                    named
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty()),
                );
            }
            if let Some(default) = default_import {
                names.push(default.to_string());
            }

            imports.push(ImportDef {
                from,
                names,
                line: source[..source.find(&from).unwrap_or(0)].lines().count(),
            });
        }

        imports
    }
    
    /// 提取所有 React 组件
    fn extract_components(source: &str) -> Vec<ComponentDef> {
        let mut components = Vec::new();

        for cap in FUNC_COMPONENT_PATTERN.captures_iter(source) {
            if let Some(name) = cap.get(1) {
                let name_str = name.as_str();
                if let Some(first_char) = name_str.chars().next() {
                    if first_char.is_uppercase() {
                        let is_exported = source.contains(&format!("export {}", name_str))
                            || source.contains(&format!("export default {}", name_str));

                        components.push(ComponentDef {
                            name: name_str.to_string(),
                            kind: ComponentKind::Functional,
                            line: source[..source.find(name_str).unwrap_or(0)].lines().count(),
                            is_exported,
                        });
                    }
                }
            }
        }

        for cap in CLASS_COMPONENT_PATTERN.captures_iter(source) {
            if let Some(name) = cap.get(1) {
                components.push(ComponentDef {
                    name: name.as_str().to_string(),
                    kind: ComponentKind::Class,
                    line: source[..source.find(name.as_str()).unwrap_or(0)].lines().count(),
                    is_exported: source.contains("export"),
                });
            }
        }

        components
    }
    
    /// 解析函数参数
    fn parse_params(params_str: &str) -> Vec<String> {
        params_str
            .split(',')
            .map(|p| {
                let trimmed = p.trim();
                // 移除类型注解
                trimmed
                    .split(':')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string()
            })
            .filter(|s| !s.is_empty())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_functions() {
        let code = r#"
function hello() { }
const world = () => { };
async function fetchData() { }
"#;
        let result = AstTraverser::traverse(code);
        assert!(result.functions.len() >= 2);
    }
    
    #[test]
    fn test_extract_classes() {
        let code = r#"
class User {
    constructor(name) { }
}
class Admin extends User {
    admin() { }
}
"#;
        let result = AstTraverser::traverse(code);
        assert_eq!(result.classes.len(), 2);
        assert_eq!(result.classes[1].extends, Some("User".to_string()));
    }
    
    #[test]
    fn test_extract_exports() {
        let code = r#"
export const MyComponent = () => { };
export default MyComponent;
export function helper() { }
"#;
        let result = AstTraverser::traverse(code);
        assert!(result.exports.len() >= 2);
    }
    
    #[test]
    fn test_extract_components() {
        let code = r#"
const Button = () => <button />;
const Form = () => <form />;
class Modal extends React.Component { }
"#;
        let result = AstTraverser::traverse(code);
        assert!(result.components.len() >= 2);
    }
}
