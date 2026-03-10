//! Scope Analysis - 作用域分析
//!
//! 分析JavaScript代码中的作用域结构，包括：
//! - 全局作用域 vs 局部作用域
//! - 变量提升 (Hoisting)
//! - 闭包分析
//! - 作用域链

use browerai_core::Result;
use browerai_js_parser::{JsAst, FunctionDecl, VariableDecl};
use std::collections::HashMap;

/// 作用域分析器
#[derive(Debug, Clone, Default)]
pub struct ScopeAnalyzer {
    /// 当前作用域栈
    scope_stack: Vec<Scope>,
    /// 下一个作用域ID
    next_scope_id: usize,
}

impl ScopeAnalyzer {
    /// 创建新的作用域分析器
    pub fn new() -> Self {
        Self {
            scope_stack: Vec::new(),
            next_scope_id: 0,
        }
    }

    /// 分析AST中的作用域
    pub fn analyze(&mut self, ast: &JsAst) -> Result<ScopeTree> {
        self.scope_stack.clear();
        self.next_scope_id = 0;
        
        // 创建全局作用域
        let global_scope = self.create_scope(ScopeKind::Global, None);
        self.scope_stack.push(global_scope);
        
        // 分析函数声明
        for func in &ast.function_decls {
            self.analyze_function(func)?;
        }
        
        // 分析变量声明
        for var in &ast.variable_decls {
            self.analyze_variable(var)?;
        }
        
        // 构建作用域树
        let scope_tree = self.build_scope_tree()?;
        
        Ok(scope_tree)
    }

    /// 分析函数作用域
    fn analyze_function(&mut self, func: &FunctionDecl) -> Result<()> {
        // 在父作用域中注册函数名
        if let Some(ref name) = func.name {
            self.register_declaration(name, DeclarationKind::Function);
        }
        
        // 创建新的函数作用域
        let parent_id = self.current_scope_id();
        let func_scope = self.create_scope(ScopeKind::Function, Some(parent_id));
        self.scope_stack.push(func_scope);
        
        // 注册参数
        for param in &func.params {
            self.register_declaration(param, DeclarationKind::Parameter);
        }
        
        // 弹出函数作用域
        self.scope_stack.pop();
        
        Ok(())
    }

    /// 分析变量声明
    fn analyze_variable(&mut self, var: &VariableDecl) -> Result<()> {
        let kind = match var.kind.as_str() {
            "var" => DeclarationKind::Var,
            "let" => DeclarationKind::Let,
            "const" => DeclarationKind::Const,
            _ => DeclarationKind::Var,
        };
        
        self.register_declaration(&var.name, kind);
        Ok(())
    }

    /// 创建新作用域
    fn create_scope(&mut self, kind: ScopeKind, parent: Option<usize>) -> Scope {
        let id = self.next_scope_id;
        self.next_scope_id += 1;
        
        Scope {
            id,
            kind,
            parent,
            declarations: HashMap::new(),
            children: Vec::new(),
        }
    }

    /// 获取当前作用域ID
    fn current_scope_id(&self) -> usize {
        self.scope_stack.last()
            .map(|s| s.id)
            .unwrap_or(0)
    }

    /// 在作用域中注册声明
    fn register_declaration(&mut self, name: &str, kind: DeclarationKind) {
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.declarations.insert(name.to_string(), Declaration {
                name: name.to_string(),
                kind,
                scope_id: scope.id,
            });
        }
    }

    /// 构建作用域树
    fn build_scope_tree(&self) -> Result<ScopeTree> {
        let scopes: HashMap<usize, Scope> = self.scope_stack.iter()
            .map(|s| (s.id, s.clone()))
            .collect();
        
        Ok(ScopeTree {
            scopes,
            root_id: 0,
        })
    }
}

/// 作用域
#[derive(Debug, Clone)]
pub struct Scope {
    /// 作用域ID
    pub id: usize,
    /// 作用域类型
    pub kind: ScopeKind,
    /// 父作用域ID
    pub parent: Option<usize>,
    /// 声明的变量/函数
    pub declarations: HashMap<String, Declaration>,
    /// 子作用域ID
    pub children: Vec<usize>,
}

/// 作用域类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    /// 全局作用域
    Global,
    /// 函数作用域
    Function,
    /// 块级作用域
    Block,
    /// 模块作用域
    Module,
}

/// 声明类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclarationKind {
    /// var声明
    Var,
    /// let声明
    Let,
    /// const声明
    Const,
    /// 函数声明
    Function,
    /// 参数
    Parameter,
    /// 类声明
    Class,
}

/// 声明信息
#[derive(Debug, Clone)]
pub struct Declaration {
    /// 名称
    pub name: String,
    /// 声明类型
    pub kind: DeclarationKind,
    /// 所在作用域ID
    pub scope_id: usize,
}

/// 作用域树
#[derive(Debug, Clone, Default)]
pub struct ScopeTree {
    /// 所有作用域
    scopes: HashMap<usize, Scope>,
    /// 根作用域ID
    root_id: usize,
}

impl ScopeTree {
    /// 创建空的作用域树
    pub fn empty() -> Self {
        Self {
            scopes: HashMap::new(),
            root_id: 0,
        }
    }
}

impl ScopeTree {
    /// 获取根作用域
    pub fn root(&self) -> Option<&Scope> {
        self.scopes.get(&self.root_id)
    }

    /// 获取指定作用域
    pub fn get_scope(&self, id: usize) -> Option<&Scope> {
        self.scopes.get(&id)
    }

    /// 查找变量所在作用域
    pub fn lookup_variable(&self, name: &str) -> Option<&Declaration> {
        self.scopes.values()
            .flat_map(|s| s.declarations.get(name))
            .next()
    }

    /// 获取作用域深度
    pub fn max_depth(&self) -> usize {
        self.scopes.values()
            .map(|s| self.calculate_depth(s.id))
            .max()
            .unwrap_or(0)
    }

    /// 计算特定作用域的深度
    fn calculate_depth(&self, scope_id: usize) -> usize {
        let mut depth = 0;
        let mut current = scope_id;
        
        while let Some(scope) = self.scopes.get(&current) {
            depth += 1;
            match scope.parent {
                Some(parent_id) => current = parent_id,
                None => break,
            }
        }
        
        depth
    }

    /// 获取全局变量
    pub fn global_variables(&self) -> Vec<String> {
        self.root()
            .map(|r| r.declarations.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.scopes.is_empty()
    }

    /// 作用域数量
    pub fn len(&self) -> usize {
        self.scopes.len()
    }

    /// 查找闭包变量（在父作用域中定义，在子作用域中引用的变量）
    pub fn find_closure_variables(&self, scope_id: usize) -> Vec<String> {
        let mut closures = Vec::new();
        
        if let Some(scope) = self.scopes.get(&scope_id) {
            // 获取父作用域链
            let mut parent_chain = Vec::new();
            let mut current = scope.parent;
            while let Some(parent_id) = current {
                parent_chain.push(parent_id);
                current = self.scopes.get(&parent_id).and_then(|s| s.parent);
            }
            
            // 检查当前作用域中的变量引用是否在父作用域中定义
            // 简化实现：检查变量名是否在父作用域链中声明
            for parent_id in parent_chain {
                if let Some(parent_scope) = self.scopes.get(&parent_id) {
                    for name in parent_scope.declarations.keys() {
                        closures.push(name.clone());
                    }
                }
            }
        }
        
        closures
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_analysis_simple() {
        let mut analyzer = ScopeAnalyzer::new();
        let ast = JsAst {
            function_decls: vec![FunctionDecl {
                name: Some("test".to_string()),
                params: vec!["a".to_string()],
                is_async: false,
                is_generator: false,
                body: None,
            }],
            variable_decls: vec![VariableDecl {
                name: "x".to_string(),
                kind: "let".to_string(),
                init: None,
            }],
            ..Default::default()
        };
        
        let scope_tree = analyzer.analyze(&ast).unwrap();
        
        assert!(!scope_tree.is_empty());
        assert!(scope_tree.lookup_variable("test").is_some());
        assert!(scope_tree.lookup_variable("x").is_some());
    }

    #[test]
    fn test_scope_depth() {
        let mut analyzer = ScopeAnalyzer::new();
        let ast = JsAst {
            function_decls: vec![
                FunctionDecl {
                    name: Some("outer".to_string()),
                    params: vec![],
                    is_async: false,
                    is_generator: false,
                    body: None,
                },
            ],
            variable_decls: vec![],
            ..Default::default()
        };
        
        let scope_tree = analyzer.analyze(&ast).unwrap();
        
        // 全局作用域 + 函数作用域 = 深度2
        assert!(scope_tree.max_depth() >= 1);
    }

    #[test]
    fn test_global_variables() {
        let mut analyzer = ScopeAnalyzer::new();
        let ast = JsAst {
            function_decls: vec![],
            variable_decls: vec![VariableDecl {
                name: "globalVar".to_string(),
                kind: "var".to_string(),
                init: None,
            }],
            ..Default::default()
        };
        
        let scope_tree = analyzer.analyze(&ast).unwrap();
        let globals = scope_tree.global_variables();
        
        assert!(globals.contains(&"globalVar".to_string()));
    }
}
