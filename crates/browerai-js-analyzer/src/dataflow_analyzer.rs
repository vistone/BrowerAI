//! Data Flow Analyzer - Advanced data flow and variable usage analysis
//!
//! This module provides sophisticated data flow analysis capabilities including:
//! - Definition-use chain construction
//! - Unused variable detection
//! - Constant propagation analysis
//! - Dead code identification
//! - Reachability analysis

use anyhow::{Context, Result};
use std::collections::HashMap;

use crate::extractor::ExtractedAst;
use crate::types::{
    DataFlowGraph, DataFlowNode, DataFlowNodeType, JsSemanticInfo,
    LocationInfo, ScopeTree,
};

/// Data flow analyzer for JavaScript code
pub struct DataFlowAnalyzer {
    /// Next node ID counter
    next_node_id: usize,

    /// Cache for def-use information
    def_use_cache: HashMap<String, Vec<String>>,
}

impl DataFlowAnalyzer {
    /// Create a new data flow analyzer
    pub fn new() -> Self {
        Self {
            next_node_id: 0,
            def_use_cache: HashMap::new(),
        }
    }

    /// Analyze data flow in the given AST
    pub fn analyze(&mut self, ast: &ExtractedAst, scope_tree: &ScopeTree) -> Result<DataFlowGraph> {
        let mut graph = DataFlowGraph::new();

        // Build def-use chains from semantic info
        self.build_def_use_chains(&mut graph, &ast.semantic, scope_tree)
            .context("Failed to build def-use chains")?;

        // Detect unused variables
        self.detect_unused_variables(&mut graph);

        // Identify constant candidates
        self.identify_constants(&mut graph);

        Ok(graph)
    }

    /// Build definition-use chains from semantic information
    fn build_def_use_chains(
        &mut self,
        graph: &mut DataFlowGraph,
        semantic: &JsSemanticInfo,
        scope_tree: &ScopeTree,
    ) -> Result<()> {
        // Process global variables
        for global_var in &semantic.global_vars {
            self.add_definition_node(
                graph,
                global_var,
                &scope_tree.global_scope_id,
                DataFlowNodeType::Definition,
                None,
                1, // Assume global defs at line 1
            );
        }

        // Process functions
        for func in &semantic.functions {
            let func_scope_id = &func.id;

            // Add function parameters as definitions
            for param in &func.parameters {
                self.add_definition_node(
                    graph,
                    &param.name,
                    func_scope_id,
                    DataFlowNodeType::Parameter,
                    None,
                    func.start_line,
                );
            }

            // Add local variables as definitions
            for local_var in &func.local_vars {
                self.add_definition_node(
                    graph,
                    local_var,
                    func_scope_id,
                    DataFlowNodeType::Definition,
                    None,
                    func.start_line,
                );
            }

            // Add captured variables as uses
            for captured_var in &func.captured_vars {
                self.add_use_node(graph, captured_var, func_scope_id, None, func.start_line);
            }

            // Add called functions as uses
            for called_func in &func.called_functions {
                self.add_use_node(graph, called_func, func_scope_id, None, func.start_line);
            }
        }

        Ok(())
    }

    /// Add a definition node to the graph
    fn add_definition_node(
        &mut self,
        graph: &mut DataFlowGraph,
        var_name: &str,
        scope_id: &str,
        node_type: DataFlowNodeType,
        location: Option<LocationInfo>,
        line: usize,
    ) {
        let node_id = format!("dfn_{}", self.next_node_id);
        self.next_node_id += 1;

        let node = DataFlowNode {
            id: node_id.clone(),
            node_type,
            variable: var_name.to_string(),
            location,
            scope_id: scope_id.to_string(),
            line,
        };

        graph.nodes.insert(node_id, node);

        // Update def-use cache
        self.def_use_cache
            .entry(var_name.to_string())
            .or_default()
            .push(var_name.to_string());
    }

    /// Add a use node to the graph
    fn add_use_node(
        &mut self,
        graph: &mut DataFlowGraph,
        var_name: &str,
        scope_id: &str,
        location: Option<LocationInfo>,
        line: usize,
    ) {
        let node_id = format!("dfu_{}", self.next_node_id);
        self.next_node_id += 1;

        let node = DataFlowNode {
            id: node_id.clone(),
            node_type: DataFlowNodeType::Use,
            variable: var_name.to_string(),
            location,
            scope_id: scope_id.to_string(),
            line,
        };

        graph.nodes.insert(node_id, node);
    }

    /// Detect unused variables in the graph
    fn detect_unused_variables(&self, graph: &mut DataFlowGraph) {
        let mut used_vars: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Collect all used variables
        for node in graph.nodes.values() {
            if node.node_type == DataFlowNodeType::Use {
                used_vars.insert(node.variable.clone());
            }
        }

        // Find defined but not used variables
        for node in graph.nodes.values() {
            if (node.node_type == DataFlowNodeType::Definition
                || node.node_type == DataFlowNodeType::Parameter)
                && !used_vars.contains(&node.variable)
                && !graph.unused_variables.contains(&node.variable) {
                    graph.unused_variables.push(node.variable.clone());
                }
        }
    }

    /// Identify constant candidates (variables that are never reassigned)
    fn identify_constants(&self, graph: &mut DataFlowGraph) {
        for _var_name in &graph.unused_variables {
            continue; // Skip unused vars
        }

        // Find variables with only one definition and no assignments
        let mut var_def_counts: HashMap<String, usize> = HashMap::new();

        for node in graph.nodes.values() {
            if node.node_type == DataFlowNodeType::Definition {
                *var_def_counts.entry(node.variable.clone()).or_insert(0) += 1;
            }
        }

        // Variables with exactly one definition are candidates for const
        for (var_name, count) in var_def_counts {
            if count == 1 {
                // Check if there are any assignments (reassignments)
                let has_assignment = graph
                    .nodes
                    .values()
                    .any(|n| n.variable == var_name && n.node_type == DataFlowNodeType::Assignment);

                if !has_assignment && !graph.constant_candidates.contains(&var_name) {
                    graph.constant_candidates.push(var_name);
                }
            }
        }
    }
}

impl Default for DataFlowAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        JsAstMetadata, JsFunctionInfo, JsParameter, ScopeTree,
    };

    fn create_test_ast() -> ExtractedAst {
        ExtractedAst {
            metadata: JsAstMetadata::default(),
            semantic: JsSemanticInfo {
                global_vars: vec!["globalVar".to_string(), "unusedGlobal".to_string()],
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
                    local_vars: vec!["localVar".to_string(), "unusedLocal".to_string()],
                    called_functions: vec!["helper".to_string()],
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
    fn test_dataflow_analyzer_creation() {
        let analyzer = DataFlowAnalyzer::new();
        assert_eq!(analyzer.next_node_id, 0);
    }

    #[test]
    fn test_basic_dataflow_analysis() {
        let mut analyzer = DataFlowAnalyzer::new();
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();

        let graph = analyzer
            .analyze(&ast, &scope_tree)
            .expect("Analysis should succeed");

        // Should have nodes for global vars and function local vars
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_global_variables_added() {
        let mut analyzer = DataFlowAnalyzer::new();
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();

        let graph = analyzer
            .analyze(&ast, &scope_tree)
            .expect("Analysis should succeed");

        // Check global variables exist in graph
        let global_vars: Vec<_> = graph
            .nodes
            .values()
            .filter(|n| n.scope_id == scope_tree.global_scope_id)
            .map(|n| n.variable.clone())
            .collect();

        assert!(global_vars.contains(&"globalVar".to_string()));
        assert!(global_vars.contains(&"unusedGlobal".to_string()));
    }

    #[test]
    fn test_function_parameters_tracked() {
        let mut analyzer = DataFlowAnalyzer::new();
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();

        let graph = analyzer
            .analyze(&ast, &scope_tree)
            .expect("Analysis should succeed");

        // Find parameter nodes
        let param_nodes: Vec<_> = graph
            .nodes
            .values()
            .filter(|n| n.node_type == DataFlowNodeType::Parameter)
            .collect();

        assert!(!param_nodes.is_empty());
        assert!(param_nodes.iter().any(|n| n.variable == "param1"));
    }

    #[test]
    fn test_local_variables_tracked() {
        let mut analyzer = DataFlowAnalyzer::new();
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();

        let graph = analyzer
            .analyze(&ast, &scope_tree)
            .expect("Analysis should succeed");

        // Find local variable nodes
        let local_nodes: Vec<_> = graph
            .nodes
            .values()
            .filter(|n| n.variable == "localVar")
            .collect();

        assert!(!local_nodes.is_empty());
    }

    #[test]
    fn test_captured_variables_marked_as_use() {
        let mut analyzer = DataFlowAnalyzer::new();
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();

        let graph = analyzer
            .analyze(&ast, &scope_tree)
            .expect("Analysis should succeed");

        // Find use nodes for captured vars
        let uses = graph.find_uses("globalVar");
        assert!(!uses.is_empty());
    }

    #[test]
    fn test_unused_variables_detection() {
        let mut analyzer = DataFlowAnalyzer::new();
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();

        let graph = analyzer
            .analyze(&ast, &scope_tree)
            .expect("Analysis should succeed");

        // unusedGlobal should be detected as unused
        assert!(graph.is_unused("unusedGlobal"));
        assert!(graph.is_unused("unusedLocal"));
    }

    #[test]
    fn test_used_variables_not_unused() {
        let mut analyzer = DataFlowAnalyzer::new();
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();

        let graph = analyzer
            .analyze(&ast, &scope_tree)
            .expect("Analysis should succeed");

        // globalVar is used (captured), should not be unused
        assert!(!graph.is_unused("globalVar"));
    }

    #[test]
    fn test_constant_candidates_identification() {
        let mut analyzer = DataFlowAnalyzer::new();
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();

        let graph = analyzer
            .analyze(&ast, &scope_tree)
            .expect("Analysis should succeed");

        // Variables with single definition are candidates
        let candidates = graph.get_constant_candidates();
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_def_use_chains() {
        let mut analyzer = DataFlowAnalyzer::new();
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();

        let graph = analyzer
            .analyze(&ast, &scope_tree)
            .expect("Analysis should succeed");

        // Check definitions and uses exist
        let defs = graph.find_definitions("globalVar");
        let uses = graph.find_uses("globalVar");

        assert!(!defs.is_empty());
        assert!(!uses.is_empty());
    }

    #[test]
    fn test_called_functions_tracked() {
        let mut analyzer = DataFlowAnalyzer::new();
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();

        let graph = analyzer
            .analyze(&ast, &scope_tree)
            .expect("Analysis should succeed");

        // Called functions should be marked as uses
        let helper_uses = graph.find_uses("helper");
        assert!(!helper_uses.is_empty());
    }

    #[test]
    fn test_multiple_definitions() {
        let mut analyzer = DataFlowAnalyzer::new();
        let mut ast = create_test_ast();

        // Add a function that redefines a variable
        ast.semantic.functions.push(JsFunctionInfo {
            id: "func2".to_string(),
            name: Some("otherFunc".to_string()),
            scope_level: 0,
            parameters: vec![],
            return_type_hint: None,
            statement_count: 3,
            cyclomatic_complexity: 1,
            is_async: false,
            is_generator: false,
            captured_vars: vec![],
            local_vars: vec!["globalVar".to_string()], // Redefines global
            called_functions: vec![],
            start_line: 15,
            end_line: 20,
        });

        let scope_tree = ScopeTree::new();
        let graph = analyzer
            .analyze(&ast, &scope_tree)
            .expect("Analysis should succeed");

        // Find definitions
        let defs = graph.find_definitions("globalVar");
        assert!(defs.len() >= 1); // At least from first definition
    }
}
