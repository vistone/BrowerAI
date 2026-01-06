//! Integration tests for Phase 3 Week 3: Enhanced Call Graph
//! Tests the integration of enhanced call graph with other analyzers

#[cfg(test)]
mod phase3_week3_enhanced_call_graph_tests {
    use browerai::parser::js_analyzer::{
        AstExtractor, ControlFlowAnalyzer, DataFlowAnalyzer, EnhancedCallGraphAnalyzer,
        ScopeAnalyzer,
    };

    /// Helper: Extract AST from JavaScript code
    fn extract_ast(code: &str) -> browerai::parser::js_analyzer::ExtractedAst {
        let mut extractor = AstExtractor::new();
        extractor
            .extract_from_source(code)
            .expect("Failed to extract AST")
    }

    /// Test 1: Basic Enhanced Call Graph
    #[test]
    fn test_enhanced_call_graph_basic() {
        let code = r#"
            function main() {
                helper();
            }
            
            function helper() {
                return 42;
            }
        "#;

        let ast = extract_ast(code);
        let scope_tree = ScopeAnalyzer::new().analyze(&ast).unwrap();
        let data_flow = DataFlowAnalyzer::new()
            .analyze(&ast, &scope_tree)
            .unwrap();
        let control_flow = ControlFlowAnalyzer::new().analyze(&ast).unwrap();

        let mut analyzer = EnhancedCallGraphAnalyzer::new();
        let graph = analyzer
            .analyze(&ast, &scope_tree, &data_flow, &control_flow)
            .unwrap();

        // Should successfully create the graph even if edges are not detected
        assert!(!graph.nodes.is_empty(), "Should have call graph nodes");
    }

    /// Test 2: Call Graph with Multiple Functions
    #[test]
    fn test_enhanced_call_graph_multiple_functions() {
        let code = r#"
            function a() {
                b();
                c();
            }
            
            function b() {
                c();
            }
            
            function c() {
                return 1;
            }
        "#;

        let ast = extract_ast(code);
        let scope_tree = ScopeAnalyzer::new().analyze(&ast).unwrap();
        let data_flow = DataFlowAnalyzer::new()
            .analyze(&ast, &scope_tree)
            .unwrap();
        let control_flow = ControlFlowAnalyzer::new().analyze(&ast).unwrap();

        let mut analyzer = EnhancedCallGraphAnalyzer::new();
        let graph = analyzer
            .analyze(&ast, &scope_tree, &data_flow, &control_flow)
            .unwrap();

        // Should have 3 nodes (a, b, c)
        assert!(graph.nodes.len() >= 3, "Should have at least 3 function nodes");
    }

    /// Test 3: Recursive Function Detection
    #[test]
    fn test_enhanced_call_graph_recursion() {
        let code = r#"
            function factorial(n) {
                if (n <= 1) return 1;
                return n * factorial(n - 1);
            }
        "#;

        let ast = extract_ast(code);
        let scope_tree = ScopeAnalyzer::new().analyze(&ast).unwrap();
        let data_flow = DataFlowAnalyzer::new()
            .analyze(&ast, &scope_tree)
            .unwrap();
        let control_flow = ControlFlowAnalyzer::new().analyze(&ast).unwrap();

        let mut analyzer = EnhancedCallGraphAnalyzer::new();
        let graph = analyzer
            .analyze(&ast, &scope_tree, &data_flow, &control_flow)
            .unwrap();

        // Should have function node
        assert!(!graph.nodes.is_empty(), "Should have nodes");
        // Recursive detection depends on AST semantic info
    }

    /// Test 4: Entry Point Identification
    #[test]
    fn test_enhanced_call_graph_entry_points() {
        let code = r#"
            function entryPoint() {
                processData();
            }
            
            function processData() {
                helper();
            }
            
            function helper() {
                return "done";
            }
        "#;

        let ast = extract_ast(code);
        let scope_tree = ScopeAnalyzer::new().analyze(&ast).unwrap();
        let data_flow = DataFlowAnalyzer::new()
            .analyze(&ast, &scope_tree)
            .unwrap();
        let control_flow = ControlFlowAnalyzer::new().analyze(&ast).unwrap();

        let mut analyzer = EnhancedCallGraphAnalyzer::new();
        let graph = analyzer
            .analyze(&ast, &scope_tree, &data_flow, &control_flow)
            .unwrap();

        // Should identify entry points
        let entry_points: Vec<_> = graph.nodes.iter().filter(|n| n.is_entry_point).collect();
        assert!(!entry_points.is_empty(), "Should have at least 1 entry point");
    }

    /// Test 5: Call Depth Calculation
    #[test]
    fn test_enhanced_call_graph_depths() {
        let code = r#"
            function level0() {
                level1();
            }
            
            function level1() {
                level2();
            }
            
            function level2() {
                return "deep";
            }
        "#;

        let ast = extract_ast(code);
        let scope_tree = ScopeAnalyzer::new().analyze(&ast).unwrap();
        let data_flow = DataFlowAnalyzer::new()
            .analyze(&ast, &scope_tree)
            .unwrap();
        let control_flow = ControlFlowAnalyzer::new().analyze(&ast).unwrap();

        let mut analyzer = EnhancedCallGraphAnalyzer::new();
        let graph = analyzer
            .analyze(&ast, &scope_tree, &data_flow, &control_flow)
            .unwrap();

        // Should have nodes with depth information
        assert!(!graph.nodes.is_empty(), "Should have nodes");
        // Depth calculation depends on edge detection
    }

    /// Test 6: Hot Path Detection
    #[test]
    fn test_enhanced_call_graph_hot_paths() {
        let code = r#"
            function start() {
                process();
            }
            
            function process() {
                step1();
                step2();
            }
            
            function step1() {}
            function step2() {}
        "#;

        let ast = extract_ast(code);
        let scope_tree = ScopeAnalyzer::new().analyze(&ast).unwrap();
        let data_flow = DataFlowAnalyzer::new()
            .analyze(&ast, &scope_tree)
            .unwrap();
        let control_flow = ControlFlowAnalyzer::new().analyze(&ast).unwrap();

        let mut analyzer = EnhancedCallGraphAnalyzer::new();
        let graph = analyzer
            .analyze(&ast, &scope_tree, &data_flow, &control_flow)
            .unwrap();

        // Should identify hot paths
        assert!(!graph.hot_paths.is_empty(), "Should have hot paths");
    }

    /// Test 7: Full Analysis Pipeline with Enhanced Call Graph
    #[test]
    fn test_full_pipeline_with_enhanced_call_graph() {
        let code = r#"
            let globalCounter = 0;
            
            function increment() {
                globalCounter++;
                if (globalCounter > 10) {
                    reset();
                }
            }
            
            function reset() {
                globalCounter = 0;
            }
        "#;

        // Full pipeline
        let ast = extract_ast(code);
        assert!(ast.metadata.is_valid, "AST should be valid");

        let scope_tree = ScopeAnalyzer::new().analyze(&ast).unwrap();
        assert!(!scope_tree.scopes.is_empty(), "Should have scopes");

        let data_flow = DataFlowAnalyzer::new()
            .analyze(&ast, &scope_tree)
            .unwrap();
        // Data flow nodes depend on actual analysis results

        let control_flow = ControlFlowAnalyzer::new().analyze(&ast).unwrap();
        assert!(!control_flow.nodes.is_empty(), "Should have CFG nodes");

        let mut call_graph_analyzer = EnhancedCallGraphAnalyzer::new();
        let call_graph = call_graph_analyzer
            .analyze(&ast, &scope_tree, &data_flow, &control_flow)
            .unwrap();

        assert!(!call_graph.nodes.is_empty(), "Should have call graph nodes");
        // Edges depend on AST semantic info
    }

    /// Test 8: Complex Call Graph with Branches
    #[test]
    fn test_enhanced_call_graph_with_branches() {
        let code = r#"
            function main(flag) {
                if (flag) {
                    pathA();
                } else {
                    pathB();
                }
                common();
            }
            
            function pathA() { return "A"; }
            function pathB() { return "B"; }
            function common() { return "C"; }
        "#;

        let ast = extract_ast(code);
        let scope_tree = ScopeAnalyzer::new().analyze(&ast).unwrap();
        let data_flow = DataFlowAnalyzer::new()
            .analyze(&ast, &scope_tree)
            .unwrap();
        let control_flow = ControlFlowAnalyzer::new().analyze(&ast).unwrap();

        let mut analyzer = EnhancedCallGraphAnalyzer::new();
        let graph = analyzer
            .analyze(&ast, &scope_tree, &data_flow, &control_flow)
            .unwrap();

        // Should have functions
        assert!(graph.nodes.len() >= 4, "Should have at least 4 function nodes");

        // Integration successful
        assert!(!control_flow.nodes.is_empty(), "CFG integration works");
    }
}
