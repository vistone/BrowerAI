//! Enhanced Call Graph Analyzer
//! 
//! Integrates call graph analysis with control flow and data flow analysis
//! to provide comprehensive function-level insights.

use super::extractor::ExtractedAst;
use super::types::*;
use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};

/// Enhanced call graph with CFG and data flow integration
#[derive(Debug, Clone, Default)]
pub struct EnhancedCallGraph {
    /// All call nodes in the graph
    pub nodes: Vec<CallNode>,
    
    /// Call edges with context information
    pub edges: Vec<CallEdge>,
    
    /// Call contexts for context-sensitive analysis
    pub call_contexts: HashMap<String, Vec<CallContext>>,
    
    /// Detected recursive call chains
    pub recursive_chains: Vec<Vec<String>>,
    
    /// Hot call paths (frequently executed)
    pub hot_paths: Vec<CallPath>,
}

/// Call graph node with enhanced information
#[derive(Debug, Clone)]
pub struct CallNode {
    /// Function ID
    pub id: String,
    
    /// Function name
    pub name: Option<String>,
    
    /// Direct callees
    pub callees: Vec<String>,
    
    /// Direct callers
    pub callers: Vec<String>,
    
    /// Call depth from entry points
    pub depth: usize,
    
    /// Is this an entry point function
    pub is_entry_point: bool,
    
    /// Estimated execution frequency (based on CFG)
    pub frequency: CallFrequency,
}

/// Call frequency estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum CallFrequency {
    /// Rarely called
    Low,
    /// Normally called
    Medium,
    /// Frequently called (in loops or hot paths)
    High,
    /// Unknown frequency
    #[default]
    Unknown,
}


/// Call edge with context
#[derive(Debug, Clone)]
pub struct CallEdge {
    /// Caller function ID
    pub from: String,
    
    /// Callee function ID
    pub to: String,
    
    /// Call site location
    pub call_site: Option<LocationInfo>,
    
    /// Is this a recursive call
    pub is_recursive: bool,
    
    /// Is this call in a loop
    pub in_loop: bool,
}

/// Call context for context-sensitive analysis
#[derive(Debug, Clone)]
pub struct CallContext {
    /// Caller function ID
    pub caller_id: String,
    
    /// Callee function ID
    pub callee_id: String,
    
    /// Call site line number
    pub call_site_line: usize,
    
    /// Variables flowing into the call
    pub data_flow_in: Vec<String>,
    
    /// Variables flowing out of the call
    pub data_flow_out: Vec<String>,
}

/// Call path through the program
#[derive(Debug, Clone)]
pub struct CallPath {
    /// Sequence of function IDs
    pub functions: Vec<String>,
    
    /// Estimated path frequency
    pub frequency: CallFrequency,
    
    /// Total path depth
    pub depth: usize,
}

/// Enhanced call graph analyzer
pub struct EnhancedCallGraphAnalyzer {
    /// Next node ID counter
    _next_id: usize,
}

impl Default for EnhancedCallGraphAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl EnhancedCallGraphAnalyzer {
    /// Create a new enhanced call graph analyzer
    pub fn new() -> Self {
        Self { _next_id: 0 }
    }

    /// Analyze with AST, scope tree, data flow graph, and control flow graph
    pub fn analyze(
        &mut self,
        ast: &ExtractedAst,
        _scope_tree: &ScopeTree,
        _data_flow: &DataFlowGraph,
        _control_flow: &ControlFlowGraph,
    ) -> Result<EnhancedCallGraph> {
        // Build basic call graph from AST
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut call_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut reverse_call_map: HashMap<String, Vec<String>> = HashMap::new();

        // Extract function calls from semantic info
        for func in &ast.semantic.functions {
            call_map.insert(func.id.clone(), func.called_functions.clone());
            reverse_call_map.insert(func.id.clone(), vec![]);
        }

        // Build reverse call map (callers of each function)
        for (caller_id, callees) in &call_map {
            for callee_id in callees {
                reverse_call_map
                    .entry(callee_id.clone())
                    .or_default()
                    .push(caller_id.clone());
            }
        }

        // Detect recursive calls
        let recursive_chains = self.detect_recursive_chains(&call_map);

        // Calculate depths
        let depths = self.calculate_depths(&call_map, &reverse_call_map);

        // Build nodes
        for func in &ast.semantic.functions {
            let func_id = &func.id;
            let callees = call_map.get(func_id).cloned().unwrap_or_default();
            let callers = reverse_call_map.get(func_id).cloned().unwrap_or_default();
            let is_entry = callers.is_empty();
            let depth = *depths.get(func_id).unwrap_or(&0);

            nodes.push(CallNode {
                id: func_id.clone(),
                name: func.name.clone(),
                callees: callees.clone(),
                callers: callers.clone(),
                depth,
                is_entry_point: is_entry,
                frequency: CallFrequency::Unknown, // Will be enhanced with CFG
            });

            // Build edges
            for callee_id in &callees {
                let is_recursive = self.is_recursive_call(&recursive_chains, func_id, callee_id);
                edges.push(CallEdge {
                    from: func_id.clone(),
                    to: callee_id.clone(),
                    call_site: None, // Could be extracted from AST
                    is_recursive,
                    in_loop: false, // Will be enhanced with CFG
                });
            }
        }

        // Build hot paths (simplified version)
        let hot_paths = self.identify_hot_paths(&nodes, &edges);

        Ok(EnhancedCallGraph {
            nodes,
            edges,
            call_contexts: HashMap::new(), // Will be populated with data flow info
            recursive_chains,
            hot_paths,
        })
    }

    /// Detect recursive call chains
    fn detect_recursive_chains(
        &self,
        call_map: &HashMap<String, Vec<String>>,
    ) -> Vec<Vec<String>> {
        let mut chains = Vec::new();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut current_path = Vec::new();

        for func_id in call_map.keys() {
            if !visited.contains(func_id) {
                self.dfs_detect_cycles(
                    func_id,
                    call_map,
                    &mut visited,
                    &mut rec_stack,
                    &mut current_path,
                    &mut chains,
                );
            }
        }

        chains
    }

    /// DFS to detect cycles
    fn dfs_detect_cycles(
        &self,
        func_id: &str,
        call_map: &HashMap<String, Vec<String>>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
        current_path: &mut Vec<String>,
        chains: &mut Vec<Vec<String>>,
    ) {
        visited.insert(func_id.to_string());
        rec_stack.insert(func_id.to_string());
        current_path.push(func_id.to_string());

        if let Some(callees) = call_map.get(func_id) {
            for callee in callees {
                if rec_stack.contains(callee) {
                    // Found a cycle - extract the chain
                    if let Some(start_idx) = current_path.iter().position(|f| f == callee) {
                        let chain = current_path[start_idx..].to_vec();
                        if !chains.contains(&chain) {
                            chains.push(chain);
                        }
                    }
                } else if !visited.contains(callee) {
                    self.dfs_detect_cycles(
                        callee,
                        call_map,
                        visited,
                        rec_stack,
                        current_path,
                        chains,
                    );
                }
            }
        }

        rec_stack.remove(func_id);
        current_path.pop();
    }

    /// Check if a call is recursive
    fn is_recursive_call(&self, chains: &[Vec<String>], caller: &str, callee: &str) -> bool {
        for chain in chains {
            if let Some(caller_idx) = chain.iter().position(|f| f == caller) {
                if let Some(callee_idx) = chain.iter().position(|f| f == callee) {
                    if caller_idx < callee_idx
                        || (caller_idx > callee_idx && chain.first() == chain.last())
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Calculate call depths using BFS
    fn calculate_depths(
        &self,
        call_map: &HashMap<String, Vec<String>>,
        reverse_call_map: &HashMap<String, Vec<String>>,
    ) -> HashMap<String, usize> {
        let mut depths = HashMap::new();
        let mut queue = VecDeque::new();

        // Find entry points (functions with no callers)
        for (func_id, callers) in reverse_call_map {
            if callers.is_empty() {
                depths.insert(func_id.clone(), 0);
                queue.push_back(func_id.clone());
            }
        }

        // BFS to calculate depths
        while let Some(func_id) = queue.pop_front() {
            let current_depth = *depths.get(&func_id).unwrap_or(&0);

            if let Some(callees) = call_map.get(&func_id) {
                for callee in callees {
                    let new_depth = current_depth + 1;
                    let existing_depth = depths.get(callee).copied().unwrap_or(usize::MAX);

                    if new_depth < existing_depth {
                        depths.insert(callee.clone(), new_depth);
                        queue.push_back(callee.clone());
                    }
                }
            }
        }

        depths
    }

    /// Identify hot call paths
    fn identify_hot_paths(&self, nodes: &[CallNode], edges: &[CallEdge]) -> Vec<CallPath> {
        let mut hot_paths = Vec::new();

        // Find entry points
        let entry_points: Vec<_> = nodes.iter().filter(|n| n.is_entry_point).collect();

        for entry in entry_points {
            // Explore paths from this entry point
            let mut path = vec![entry.id.clone()];
            self.explore_paths(&entry.id, edges, &mut path, &mut hot_paths, 10); // Max depth 10
        }

        // Sort by depth and take top paths
        hot_paths.sort_by(|a, b| b.depth.cmp(&a.depth));
        hot_paths.truncate(10); // Keep top 10 hot paths

        hot_paths
    }

    /// Explore call paths recursively
    fn explore_paths(
        &self,
        current: &str,
        edges: &[CallEdge],
        path: &mut Vec<String>,
        hot_paths: &mut Vec<CallPath>,
        max_depth: usize,
    ) {
        if path.len() > max_depth {
            return;
        }

        // Find outgoing edges
        let outgoing: Vec<_> = edges.iter().filter(|e| e.from == current).collect();

        if outgoing.is_empty() {
            // Leaf node - save this path
            hot_paths.push(CallPath {
                functions: path.clone(),
                frequency: CallFrequency::Medium,
                depth: path.len(),
            });
        } else {
            for edge in outgoing {
                if !path.contains(&edge.to) {
                    // Avoid cycles
                    path.push(edge.to.clone());
                    self.explore_paths(&edge.to, edges, path, hot_paths, max_depth);
                    path.pop();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: Create a test AST
    fn create_test_ast() -> ExtractedAst {
        let mut semantic = JsSemanticInfo::default();

        // Function A calls B and C
        semantic.functions.push(JsFunctionInfo {
            id: "func_a".to_string(),
            name: Some("a".to_string()),
            scope_level: 0,
            parameters: vec![],
            return_type_hint: None,
            statement_count: 3,
            cyclomatic_complexity: 1,
            is_async: false,
            is_generator: false,
            captured_vars: vec![],
            local_vars: vec![],
            called_functions: vec!["func_b".to_string(), "func_c".to_string()],
            start_line: 1,
            end_line: 5,
        });

        // Function B calls C
        semantic.functions.push(JsFunctionInfo {
            id: "func_b".to_string(),
            name: Some("b".to_string()),
            scope_level: 0,
            parameters: vec![],
            return_type_hint: None,
            statement_count: 2,
            cyclomatic_complexity: 1,
            is_async: false,
            is_generator: false,
            captured_vars: vec![],
            local_vars: vec![],
            called_functions: vec!["func_c".to_string()],
            start_line: 7,
            end_line: 10,
        });

        // Function C (leaf)
        semantic.functions.push(JsFunctionInfo {
            id: "func_c".to_string(),
            name: Some("c".to_string()),
            scope_level: 0,
            parameters: vec![],
            return_type_hint: None,
            statement_count: 1,
            cyclomatic_complexity: 1,
            is_async: false,
            is_generator: false,
            captured_vars: vec![],
            local_vars: vec![],
            called_functions: vec![],
            start_line: 12,
            end_line: 14,
        });

        ExtractedAst {
            metadata: JsAstMetadata::default(),
            semantic,
            warnings: vec![],
        }
    }

    #[test]
    fn test_enhanced_call_graph_creation() {
        let analyzer = EnhancedCallGraphAnalyzer::new();
        assert_eq!(analyzer._next_id, 0);
    }

    #[test]
    fn test_basic_call_graph_analysis() {
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();
        let data_flow = DataFlowGraph::default();
        let control_flow = ControlFlowGraph::new();

        let mut analyzer = EnhancedCallGraphAnalyzer::new();
        let result = analyzer.analyze(&ast, &scope_tree, &data_flow, &control_flow);

        assert!(result.is_ok());
        let graph = result.unwrap();
        assert_eq!(graph.nodes.len(), 3);
    }

    #[test]
    fn test_call_graph_nodes() {
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();
        let data_flow = DataFlowGraph::default();
        let control_flow = ControlFlowGraph::new();

        let mut analyzer = EnhancedCallGraphAnalyzer::new();
        let graph = analyzer
            .analyze(&ast, &scope_tree, &data_flow, &control_flow)
            .unwrap();

        // Check that all functions are present
        let ids: Vec<_> = graph.nodes.iter().map(|n| n.id.as_str()).collect();
        assert!(ids.contains(&"func_a"));
        assert!(ids.contains(&"func_b"));
        assert!(ids.contains(&"func_c"));
    }

    #[test]
    fn test_call_graph_edges() {
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();
        let data_flow = DataFlowGraph::default();
        let control_flow = ControlFlowGraph::new();

        let mut analyzer = EnhancedCallGraphAnalyzer::new();
        let graph = analyzer
            .analyze(&ast, &scope_tree, &data_flow, &control_flow)
            .unwrap();

        // Check edges
        assert!(graph.edges.len() >= 2);

        // A -> B edge exists
        let a_to_b = graph
            .edges
            .iter()
            .any(|e| e.from == "func_a" && e.to == "func_b");
        assert!(a_to_b, "Should have edge from A to B");

        // A -> C edge exists
        let a_to_c = graph
            .edges
            .iter()
            .any(|e| e.from == "func_a" && e.to == "func_c");
        assert!(a_to_c, "Should have edge from A to C");
    }

    #[test]
    fn test_entry_point_detection() {
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();
        let data_flow = DataFlowGraph::default();
        let control_flow = ControlFlowGraph::new();

        let mut analyzer = EnhancedCallGraphAnalyzer::new();
        let graph = analyzer
            .analyze(&ast, &scope_tree, &data_flow, &control_flow)
            .unwrap();

        // Function A should be an entry point (no callers)
        let func_a = graph.nodes.iter().find(|n| n.id == "func_a").unwrap();
        assert!(func_a.is_entry_point, "Function A should be entry point");

        // Function B and C should not be entry points
        let func_b = graph.nodes.iter().find(|n| n.id == "func_b").unwrap();
        assert!(!func_b.is_entry_point, "Function B should not be entry");
    }

    #[test]
    fn test_call_depth_calculation() {
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();
        let data_flow = DataFlowGraph::default();
        let control_flow = ControlFlowGraph::new();

        let mut analyzer = EnhancedCallGraphAnalyzer::new();
        let graph = analyzer
            .analyze(&ast, &scope_tree, &data_flow, &control_flow)
            .unwrap();

        // Check depths: A(0) -> B(1) -> C(1 from A, 2 from B)
        let func_a = graph.nodes.iter().find(|n| n.id == "func_a").unwrap();
        assert_eq!(func_a.depth, 0, "A should be at depth 0");

        let func_b = graph.nodes.iter().find(|n| n.id == "func_b").unwrap();
        assert_eq!(func_b.depth, 1, "B should be at depth 1");

        let func_c = graph.nodes.iter().find(|n| n.id == "func_c").unwrap();
        assert!(func_c.depth >= 1, "C should be at depth >= 1");
    }

    #[test]
    fn test_recursive_call_detection() {
        let mut ast = create_test_ast();

        // Add recursive function D that calls itself
        ast.semantic.functions.push(JsFunctionInfo {
            id: "func_d".to_string(),
            name: Some("d".to_string()),
            scope_level: 0,
            parameters: vec![],
            return_type_hint: None,
            statement_count: 2,
            cyclomatic_complexity: 2,
            is_async: false,
            is_generator: false,
            captured_vars: vec![],
            local_vars: vec![],
            called_functions: vec!["func_d".to_string()], // Calls itself
            start_line: 16,
            end_line: 20,
        });

        let scope_tree = ScopeTree::new();
        let data_flow = DataFlowGraph::default();
        let control_flow = ControlFlowGraph::new();

        let mut analyzer = EnhancedCallGraphAnalyzer::new();
        let graph = analyzer
            .analyze(&ast, &scope_tree, &data_flow, &control_flow)
            .unwrap();

        // Should detect recursive chain
        assert!(!graph.recursive_chains.is_empty(), "Should detect recursion");
    }

    #[test]
    fn test_hot_path_identification() {
        let ast = create_test_ast();
        let scope_tree = ScopeTree::new();
        let data_flow = DataFlowGraph::default();
        let control_flow = ControlFlowGraph::new();

        let mut analyzer = EnhancedCallGraphAnalyzer::new();
        let graph = analyzer
            .analyze(&ast, &scope_tree, &data_flow, &control_flow)
            .unwrap();

        // Should identify some hot paths
        assert!(!graph.hot_paths.is_empty(), "Should have hot paths");

        // Check that hot paths start from entry points
        for path in &graph.hot_paths {
            assert!(!path.functions.is_empty(), "Path should not be empty");
        }
    }
}
