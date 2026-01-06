//! Integration tests for Phase 3 Week 2: Control Flow Analyzer
//! Tests interaction between ControlFlowAnalyzer, DataFlowAnalyzer, ScopeAnalyzer

#[cfg(test)]
mod phase3_week2_integration_tests {
    use browerai::parser::js_analyzer::{
        ControlFlowAnalyzer, DataFlowAnalyzer, ScopeAnalyzer, AstExtractor,
    };

    /// Helper: Extract AST from JavaScript code
    fn extract_ast(code: &str) -> browerai::parser::js_analyzer::ExtractedAst {
        let mut extractor = AstExtractor::new();
        extractor
            .extract_from_source(code)
            .expect("Failed to extract AST")
    }

    /// Test 1: Basic CFG Analysis
    #[test]
    fn test_cfg_basic_analysis() {
        let code = r#"
            function process(a, b) {
                let result = a + b;
                if (result > 0) {
                    return result;
                }
                return 0;
            }
        "#;

        let ast = extract_ast(code);
        assert!(ast.metadata.is_valid, "AST should be valid");

        let mut cfg_analyzer = ControlFlowAnalyzer::new();
        let cfg = cfg_analyzer.analyze(&ast).expect("CFG analysis failed");

        assert!(!cfg.nodes.is_empty(), "CFG should have nodes");
    }

    /// Test 2: CFG + Scope Integration
    #[test]
    fn test_cfg_scope_integration() {
        let code = r#"
            function process(a, b, c) {
                let result = a + b;
                if (c > 0) {
                    result = result * c;
                }
                return result;
            }
        "#;

        let ast = extract_ast(code);

        let mut scope_analyzer = ScopeAnalyzer::new();
        let scope_tree = scope_analyzer.analyze(&ast).expect("Scope analysis failed");

        assert!(!scope_tree.scopes.is_empty(), "Should have scopes");

        let mut cfg_analyzer = ControlFlowAnalyzer::new();
        let cfg = cfg_analyzer.analyze(&ast).expect("CFG analysis failed");

        assert!(!cfg.nodes.is_empty(), "CFG should have nodes");
    }

    /// Test 3: CFG + DataFlow Integration
    #[test]
    fn test_cfg_dataflow_integration() {
        let code = r#"
            function analyze(flag) {
                let x = 10;
                let y = 20;
                
                if (flag) {
                    let z = x + y;
                    console.log(z);
                }
                
                return x;
            }
        "#;

        let ast = extract_ast(code);

        let mut scope_analyzer = ScopeAnalyzer::new();
        let scope_tree = scope_analyzer.analyze(&ast).expect("Scope analysis failed");

        let mut dataflow = DataFlowAnalyzer::new();
        let _df_graph = dataflow.analyze(&ast, &scope_tree).expect("Data flow analysis failed");

        let mut cfg_analyzer = ControlFlowAnalyzer::new();
        let cfg = cfg_analyzer.analyze(&ast).expect("CFG analysis failed");

        let reachable_count = cfg.nodes.iter().filter(|n| n.is_reachable).count();
        assert!(reachable_count > 0, "Should have reachable nodes");
    }

    /// Test 4: Nested Conditions
    #[test]
    fn test_cfg_nested_conditions() {
        let code = r#"
            function nested(a, b, c) {
                if (a > 0) {
                    if (b > 0) {
                        return a + b;
                    } else {
                        return a - b;
                    }
                } else {
                    if (c > 0) {
                        return c;
                    } else {
                        return 0;
                    }
                }
            }
        "#;

        let ast = extract_ast(code);

        let mut analyzer = ControlFlowAnalyzer::new();
        let cfg = analyzer.analyze(&ast).expect("CFG analysis failed");

        // Should have multiple edges (at least more than 1 for basic flow)
        assert!(cfg.edges.len() >= 2, "Should have multiple edges");
    }

    /// Test 5: Loop Detection
    #[test]
    fn test_cfg_loop_detection() {
        let code = r#"
            function count(n) {
                let i = 0;
                while (i < n) {
                    console.log(i);
                    i = i + 1;
                }
                return i;
            }
        "#;

        let ast = extract_ast(code);

        let mut analyzer = ControlFlowAnalyzer::new();
        let cfg = analyzer.analyze(&ast).expect("CFG analysis failed");

        // Loops and branches should create multiple edges
        assert!(cfg.edges.len() >= 2, "Should have multiple edges");
    }

    /// Test 6: Multiple Functions
    #[test]
    fn test_cfg_multiple_functions() {
        let code = r#"
            function func1(x) {
                return x * 2;
            }
            
            function func2(y) {
                if (y > 0) {
                    return func1(y);
                }
                return 0;
            }
        "#;

        let ast = extract_ast(code);

        let mut analyzer = ControlFlowAnalyzer::new();
        let cfg = analyzer.analyze(&ast).expect("CFG analysis failed");

        assert!(!cfg.nodes.is_empty(), "Should have nodes");
    }

    /// Test 7: Reachability Marking
    #[test]
    fn test_cfg_reachability_marking() {
        let code = r#"
            function reachable() {
                let x = 10;
                console.log(x);
                return x;
            }
        "#;

        let ast = extract_ast(code);

        let mut analyzer = ControlFlowAnalyzer::new();
        let cfg = analyzer.analyze(&ast).expect("CFG analysis failed");

        if let Some(entry_id) = &cfg.entry {
            let entry_reachable = cfg
                .nodes
                .iter()
                .find(|n| n.id == *entry_id)
                .map(|n| n.is_reachable)
                .unwrap_or(false);
            assert!(entry_reachable, "Entry node should be reachable");
        }
    }

    /// Test 8: Complex Expressions
    #[test]
    fn test_cfg_complex_expressions() {
        let code = r#"
            function complex(a, b, c, d) {
                if (a > 0 && b < 10 || (c === d && !d)) {
                    return "complex";
                } else if (a === b) {
                    return "equal";
                }
                return "other";
            }
        "#;

        let ast = extract_ast(code);

        let mut analyzer = ControlFlowAnalyzer::new();
        let cfg = analyzer.analyze(&ast).expect("CFG analysis failed");

        // Should handle complex expressions and create graph structure
        assert!(!cfg.nodes.is_empty(), "Should have CFG nodes for complex expression");
    }

    /// Test 9: Nested Scopes
    #[test]
    fn test_cfg_nested_scopes() {
        let code = r#"
            function outer(x) {
                let result = x;
                
                function inner(y) {
                    return x + y;
                }
                
                result = inner(5);
                return result;
            }
        "#;

        let ast = extract_ast(code);

        let mut scope_analyzer = ScopeAnalyzer::new();
        let scope_tree = scope_analyzer.analyze(&ast).expect("Failed");

        assert!(scope_tree.scopes.len() > 1, "Should have nested scopes");

        let mut cfg_analyzer = ControlFlowAnalyzer::new();
        let cfg = cfg_analyzer.analyze(&ast).expect("CFG analysis failed");

        assert!(!cfg.nodes.is_empty(), "Should have CFG nodes");
    }

    /// Test 10: Full Integration Pipeline
    #[test]
    fn test_full_analysis_pipeline() {
        let code = r#"
            let globalVar = 100;
            
            function processData(input, flag) {
                let local = globalVar;
                let processed = local + input;
                
                if (flag) {
                    processed = processed * 2;
                    console.log(processed);
                }
                
                return processed;
            }
        "#;

        let ast = extract_ast(code);
        assert!(ast.metadata.is_valid, "AST should be valid");

        let mut scope_analyzer = ScopeAnalyzer::new();
        let scope_tree = scope_analyzer.analyze(&ast).expect("Failed");
        assert!(!scope_tree.scopes.is_empty(), "Should create scopes");

        let mut df_analyzer = DataFlowAnalyzer::new();
        let df_graph = df_analyzer.analyze(&ast, &scope_tree).expect("Failed");
        assert!(!df_graph.nodes.is_empty(), "Should create data flow nodes");

        let mut cfg_analyzer = ControlFlowAnalyzer::new();
        let cfg = cfg_analyzer.analyze(&ast).expect("Failed");
        assert!(!cfg.nodes.is_empty(), "Should create CFG nodes");

        assert!(cfg.entry.is_some(), "Should have entry node");
        assert!(cfg.exit.is_some(), "Should have exit node");
    }
}
