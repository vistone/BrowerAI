//! Scope Analyzer - Advanced scope and variable binding analysis
//!
//! This module provides sophisticated scope analysis capabilities including:
//! - Scope chain construction
//! - Variable binding tracking
//! - Closure capture analysis
//! - Variable shadowing detection
//! - Hoisting analysis

use anyhow::{Context, Result};
use std::collections::HashMap;

use crate::parser::js_analyzer::extractor::ExtractedAst;
use crate::parser::js_analyzer::swc_extractor::LocationInfo;
use crate::parser::js_analyzer::types::{
    BindingKind, ClosureInfo, JsSemanticInfo, Scope, ScopeTree, ScopeType, VariableBinding,
};

/// Scope analyzer for JavaScript code
pub struct ScopeAnalyzer {
    /// Next scope ID counter
    next_scope_id: usize,
}

impl ScopeAnalyzer {
    /// Create a new scope analyzer
    pub fn new() -> Self {
        Self { next_scope_id: 1 }
    }

    /// Analyze scopes in the given AST
    pub fn analyze(&mut self, ast: &ExtractedAst) -> Result<ScopeTree> {
        let mut tree = ScopeTree::new();

        // Analyze semantic information
        self.analyze_semantic(&mut tree, &ast.semantic)
            .context("Failed to analyze semantic information")?;

        Ok(tree)
    }

    /// Analyze semantic information and build scope tree
    fn analyze_semantic(&mut self, tree: &mut ScopeTree, semantic: &JsSemanticInfo) -> Result<()> {
        let global_scope_id = tree.global_scope_id.clone();

        // Add global variables
        for var in &semantic.global_vars {
            self.add_variable_to_scope(tree, &global_scope_id, var, BindingKind::Var, None);
        }

        // Add functions and their scopes
        for func in &semantic.functions {
            // Add function to global scope
            self.add_variable_to_scope(
                tree,
                &global_scope_id,
                &func.name.clone().unwrap_or_default(),
                BindingKind::Function,
                None,
            );

            // Create function scope
            let func_scope_id =
                self.create_scope(tree, Some(global_scope_id.clone()), ScopeType::Function);

            // Add function parameters
            for param in &func.parameters {
                self.add_variable_to_scope(
                    tree,
                    &func_scope_id,
                    &param.name,
                    BindingKind::Parameter,
                    None,
                );
            }

            // Add local variables
            for local_var in &func.local_vars {
                self.add_variable_to_scope(tree, &func_scope_id, local_var, BindingKind::Let, None);
            }

            // Analyze closures
            if !func.captured_vars.is_empty() {
                self.analyze_closure(tree, &func_scope_id, &global_scope_id, func);
            }
        }

        // Add classes
        for class in &semantic.classes {
            self.add_variable_to_scope(
                tree,
                &global_scope_id,
                &class.name,
                BindingKind::Class,
                None,
            );

            // Create class scope
            let class_scope_id =
                self.create_scope(tree, Some(global_scope_id.clone()), ScopeType::Class);

            // Add methods
            for method in &class.methods {
                if let Some(scope) = tree.scopes.get_mut(&class_scope_id) {
                    scope.functions.push(method.name.clone());
                }
            }
        }

        Ok(())
    }

    /// Create a new scope
    fn create_scope(
        &mut self,
        tree: &mut ScopeTree,
        parent: Option<String>,
        scope_type: ScopeType,
    ) -> String {
        let scope_id = format!("scope_{}", self.next_scope_id);
        self.next_scope_id += 1;

        let scope = Scope {
            id: scope_id.clone(),
            parent: parent.clone(),
            variables: HashMap::new(),
            functions: Vec::new(),
            closures: Vec::new(),
            children: Vec::new(),
            scope_type,
        };

        // Add to parent's children
        if let Some(parent_id) = &parent {
            if let Some(parent_scope) = tree.scopes.get_mut(parent_id) {
                parent_scope.children.push(scope_id.clone());
            }
        }

        tree.scopes.insert(scope_id.clone(), scope);
        scope_id
    }

    /// Add a variable to a scope
    fn add_variable_to_scope(
        &self,
        tree: &mut ScopeTree,
        scope_id: &str,
        var_name: &str,
        kind: BindingKind,
        location: Option<LocationInfo>,
    ) {
        if var_name.is_empty() {
            return;
        }

        // Check for shadowing first (before getting mutable borrow)
        let is_shadowing = self.check_shadowing(tree, scope_id, var_name);

        if is_shadowing {
            tree.shadowing_warnings.push(format!(
                "Variable '{}' shadows a variable from outer scope",
                var_name
            ));
        }

        // Now we can safely get mutable borrow
        if let Some(scope) = tree.scopes.get_mut(scope_id) {
            let binding = VariableBinding {
                name: var_name.to_string(),
                kind,
                scope_id: scope_id.to_string(),
                is_captured: false,
                is_shadowing,
                location,
            };

            scope.variables.insert(var_name.to_string(), binding);
        }
    }

    /// Check if a variable shadows another variable
    fn check_shadowing(&self, tree: &ScopeTree, scope_id: &str, var_name: &str) -> bool {
        if let Some(scope) = tree.scopes.get(scope_id) {
            // Check parent scopes
            if let Some(parent_id) = &scope.parent {
                if let Some(parent_scope) = tree.scopes.get(parent_id) {
                    if parent_scope.variables.contains_key(var_name) {
                        return true;
                    }
                    // Recursively check parent's parent
                    return self.check_shadowing(tree, parent_id, var_name);
                }
            }
        }
        false
    }

    /// Analyze closure capture
    fn analyze_closure(
        &self,
        tree: &mut ScopeTree,
        func_scope_id: &str,
        parent_scope_id: &str,
        func: &crate::parser::js_analyzer::types::JsFunctionInfo,
    ) {
        let closure_info = ClosureInfo {
            function_id: func.id.clone(),
            captured_variables: func.captured_vars.clone(),
            parent_scope_id: parent_scope_id.to_string(),
        };

        // Mark captured variables
        for captured_var in &func.captured_vars {
            self.mark_variable_as_captured(tree, parent_scope_id, captured_var);
        }

        // Add closure info to function scope
        if let Some(scope) = tree.scopes.get_mut(func_scope_id) {
            scope.closures.push(closure_info);
        }
    }

    /// Mark a variable as captured by a closure
    fn mark_variable_as_captured(&self, tree: &mut ScopeTree, scope_id: &str, var_name: &str) {
        if let Some(scope) = tree.scopes.get_mut(scope_id) {
            if let Some(binding) = scope.variables.get_mut(var_name) {
                binding.is_captured = true;
            } else if let Some(parent_id) = &scope.parent.clone() {
                // Recursively search parent scopes
                self.mark_variable_as_captured(tree, parent_id, var_name);
            }
        }
    }
}

impl Default for ScopeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::js_analyzer::types::{
        JsAstMetadata, JsClassInfo, JsFunctionInfo, JsParameter,
    };

    fn create_test_ast() -> ExtractedAst {
        ExtractedAst {
            metadata: JsAstMetadata::default(),
            semantic: JsSemanticInfo {
                global_vars: vec!["globalVar".to_string()],
                functions: vec![JsFunctionInfo {
                    id: "func1".to_string(),
                    name: Some("myFunction".to_string()),
                    scope_level: 0,
                    parameters: vec![JsParameter {
                        name: "param1".to_string(),
                        type_hint: None,
                        has_default: false,
                        is_rest: false,
                    }],
                    return_type_hint: None,
                    statement_count: 5,
                    cyclomatic_complexity: 1,
                    is_async: false,
                    is_generator: false,
                    captured_vars: vec!["globalVar".to_string()],
                    local_vars: vec!["localVar".to_string()],
                    called_functions: vec![],
                    start_line: 1,
                    end_line: 10,
                }],
                classes: vec![],
                event_handlers: vec![],
                uses_eval: false,
                uses_dynamic_require: false,
                detected_frameworks: vec![],
                special_features: vec![],
            },
            warnings: vec![],
        }
    }

    #[test]
    fn test_scope_analyzer_creation() {
        let analyzer = ScopeAnalyzer::new();
        assert_eq!(analyzer.next_scope_id, 1);
    }

    #[test]
    fn test_basic_scope_analysis() {
        let mut analyzer = ScopeAnalyzer::new();
        let ast = create_test_ast();

        let tree = analyzer.analyze(&ast).expect("Analysis should succeed");

        // Should have at least global scope
        assert!(!tree.scopes.is_empty());
        assert!(tree.scopes.contains_key(&tree.global_scope_id));
    }

    #[test]
    fn test_global_variables() {
        let mut analyzer = ScopeAnalyzer::new();
        let ast = create_test_ast();

        let tree = analyzer.analyze(&ast).expect("Analysis should succeed");

        // Check global variable
        let global_scope = tree.scopes.get(&tree.global_scope_id).unwrap();
        assert!(global_scope.variables.contains_key("globalVar"));
    }

    #[test]
    fn test_function_scope_creation() {
        let mut analyzer = ScopeAnalyzer::new();
        let ast = create_test_ast();

        let tree = analyzer.analyze(&ast).expect("Analysis should succeed");

        // Should have global scope + function scope
        assert!(tree.scopes.len() >= 2);

        // Check function is in global scope
        let global_scope = tree.scopes.get(&tree.global_scope_id).unwrap();
        assert!(global_scope.variables.contains_key("myFunction"));
    }

    #[test]
    fn test_function_parameters() {
        let mut analyzer = ScopeAnalyzer::new();
        let ast = create_test_ast();

        let tree = analyzer.analyze(&ast).expect("Analysis should succeed");

        // Find function scope
        let func_scope = tree
            .scopes
            .values()
            .find(|s| s.scope_type == ScopeType::Function)
            .expect("Should have function scope");

        // Check parameter
        assert!(func_scope.variables.contains_key("param1"));
        let param_binding = func_scope.variables.get("param1").unwrap();
        assert_eq!(param_binding.kind, BindingKind::Parameter);
    }

    #[test]
    fn test_local_variables() {
        let mut analyzer = ScopeAnalyzer::new();
        let ast = create_test_ast();

        let tree = analyzer.analyze(&ast).expect("Analysis should succeed");

        // Find function scope
        let func_scope = tree
            .scopes
            .values()
            .find(|s| s.scope_type == ScopeType::Function)
            .expect("Should have function scope");

        // Check local variable
        assert!(func_scope.variables.contains_key("localVar"));
    }

    #[test]
    fn test_closure_analysis() {
        let mut analyzer = ScopeAnalyzer::new();
        let ast = create_test_ast();

        let tree = analyzer.analyze(&ast).expect("Analysis should succeed");

        // Find function scope
        let func_scope = tree
            .scopes
            .values()
            .find(|s| s.scope_type == ScopeType::Function)
            .expect("Should have function scope");

        // Should have closure info
        assert!(!func_scope.closures.is_empty());
        assert_eq!(func_scope.closures[0].captured_variables, vec!["globalVar"]);
    }

    #[test]
    fn test_captured_variables_marked() {
        let mut analyzer = ScopeAnalyzer::new();
        let ast = create_test_ast();

        let tree = analyzer.analyze(&ast).expect("Analysis should succeed");

        // Check that captured variable is marked
        let global_scope = tree.scopes.get(&tree.global_scope_id).unwrap();
        let global_var = global_scope.variables.get("globalVar").unwrap();
        assert!(global_var.is_captured);
    }

    #[test]
    fn test_variable_lookup() {
        let mut analyzer = ScopeAnalyzer::new();
        let ast = create_test_ast();

        let tree = analyzer.analyze(&ast).expect("Analysis should succeed");

        // Lookup global variable
        let binding = tree.lookup_variable("globalVar");
        assert!(binding.is_some());
        assert_eq!(binding.unwrap().name, "globalVar");

        // Lookup non-existent variable
        let binding = tree.lookup_variable("nonExistent");
        assert!(binding.is_none());
    }

    #[test]
    fn test_class_scope() {
        let mut analyzer = ScopeAnalyzer::new();
        let mut ast = create_test_ast();

        // Add a class
        ast.semantic.classes.push(JsClassInfo {
            id: "class1".to_string(),
            name: "MyClass".to_string(),
            parent_class: None,
            implements: vec![],
            properties: vec![],
            methods: vec![],
            static_methods: vec![],
            constructor: None,
            start_line: 1,
            end_line: 10,
        });

        let tree = analyzer.analyze(&ast).expect("Analysis should succeed");

        // Check class is in global scope
        let global_scope = tree.scopes.get(&tree.global_scope_id).unwrap();
        assert!(global_scope.variables.contains_key("MyClass"));

        // Should have class scope
        let has_class_scope = tree
            .scopes
            .values()
            .any(|s| s.scope_type == ScopeType::Class);
        assert!(has_class_scope);
    }
}
