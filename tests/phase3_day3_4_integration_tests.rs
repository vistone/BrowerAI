//! Phase 3 Day 3-4 Integration Tests
//!
//! This test file demonstrates the integration of:
//! - ScopeAnalyzer (Day 1-2)
//! - DataFlowAnalyzer (Day 3-4)
//! - And their interaction with Phase 1-2 components

#[cfg(test)]
mod phase3_day3_4_integration_tests {
    use browerai::parser::js_analyzer::{
        DataFlowAnalyzer, ExtractedAst, JsAstMetadata, JsClassInfo, JsFunctionInfo, JsParameter,
        JsSemanticInfo, ScopeAnalyzer, ScopeTree, ScopeType,
    };

    fn create_test_program() -> ExtractedAst {
        // Simulates a JavaScript program with:
        // - global variable used by function
        // - unused global variable
        // - function parameter
        // - local variable
        // - captured variables
        // - called functions

        ExtractedAst {
            metadata: JsAstMetadata::default(),
            semantic: JsSemanticInfo {
                global_vars: vec!["counter".to_string(), "unused".to_string()],
                functions: vec![
                    JsFunctionInfo {
                        id: "increment".to_string(),
                        name: Some("increment".to_string()),
                        scope_level: 0,
                        parameters: vec![JsParameter {
                            name: "value".to_string(),
                            type_hint: Some("number".to_string()),
                            has_default: false,
                            is_rest: false,
                        }],
                        return_type_hint: Some("number".to_string()),
                        statement_count: 3,
                        cyclomatic_complexity: 1,
                        is_async: false,
                        is_generator: false,
                        captured_vars: vec!["counter".to_string()],
                        local_vars: vec!["result".to_string()],
                        called_functions: vec!["log".to_string()],
                        start_line: 1,
                        end_line: 5,
                    },
                    JsFunctionInfo {
                        id: "log".to_string(),
                        name: Some("log".to_string()),
                        scope_level: 0,
                        parameters: vec![JsParameter {
                            name: "msg".to_string(),
                            type_hint: None,
                            has_default: false,
                            is_rest: false,
                        }],
                        return_type_hint: None,
                        statement_count: 1,
                        cyclomatic_complexity: 1,
                        is_async: false,
                        is_generator: false,
                        captured_vars: vec![],
                        local_vars: vec![],
                        called_functions: vec![],
                        start_line: 7,
                        end_line: 9,
                    },
                ],
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
    fn test_scope_and_dataflow_integration() {
        let ast = create_test_program();

        // Step 1: Analyze scopes
        let mut scope_analyzer = ScopeAnalyzer::new();
        let scope_tree = scope_analyzer.analyze(&ast).expect("Scope analysis failed");

        // Verify scope tree structure
        assert!(scope_tree.scopes.len() >= 3); // global + at least 2 functions
        assert!(scope_tree.scopes.contains_key(&scope_tree.global_scope_id));

        // Step 2: Analyze data flow
        let mut dataflow_analyzer = DataFlowAnalyzer::new();
        let dataflow_graph = dataflow_analyzer
            .analyze(&ast, &scope_tree)
            .expect("Data flow analysis failed");

        // Verify data flow graph
        assert!(!dataflow_graph.nodes.is_empty());

        // Step 3: Cross-validation
        // Verify that unused global variable is detected
        assert!(dataflow_graph.is_unused("unused"));

        // Verify that used variable is not marked as unused
        assert!(!dataflow_graph.is_unused("counter"));
    }

    #[test]
    fn test_closure_capture_with_dataflow() {
        let ast = create_test_program();

        // Analyze scopes to understand closure captures
        let mut scope_analyzer = ScopeAnalyzer::new();
        let scope_tree = scope_analyzer.analyze(&ast).expect("Scope analysis failed");

        // Find captured variables in scope tree
        let mut captured_count = 0;
        for scope in scope_tree.scopes.values() {
            for var in scope.variables.values() {
                if var.is_captured {
                    captured_count += 1;
                }
            }
        }
        assert!(captured_count > 0);

        // Verify data flow also tracks captures
        let mut dataflow_analyzer = DataFlowAnalyzer::new();
        let dataflow_graph = dataflow_analyzer
            .analyze(&ast, &scope_tree)
            .expect("Data flow analysis failed");

        // "counter" should have both definition and uses
        let defs = dataflow_graph.find_definitions("counter");
        let uses = dataflow_graph.find_uses("counter");

        assert!(!defs.is_empty(), "counter should have definitions");
        assert!(!uses.is_empty(), "counter should have uses");
    }

    #[test]
    fn test_unused_variable_detection() {
        let ast = create_test_program();

        let mut scope_analyzer = ScopeAnalyzer::new();
        let scope_tree = scope_analyzer.analyze(&ast).expect("Scope analysis failed");

        let mut dataflow_analyzer = DataFlowAnalyzer::new();
        let dataflow_graph = dataflow_analyzer
            .analyze(&ast, &scope_tree)
            .expect("Data flow analysis failed");

        // "unused" global variable should be detected as unused
        assert!(dataflow_graph.is_unused("unused"));

        // Verify it's in the list
        assert!(dataflow_graph
            .unused_variables
            .contains(&"unused".to_string()));
    }

    #[test]
    fn test_function_parameter_tracking() {
        let ast = create_test_program();

        let mut scope_analyzer = ScopeAnalyzer::new();
        let scope_tree = scope_analyzer.analyze(&ast).expect("Scope analysis failed");

        // Find function scopes with parameters
        let mut param_count = 0;
        for scope in scope_tree.scopes.values() {
            if scope.scope_type == ScopeType::Function {
                for var in scope.variables.values() {
                    if let browerai::parser::js_analyzer::BindingKind::Parameter = var.kind {
                        param_count += 1;
                    }
                }
            }
        }
        assert!(param_count > 0, "Should find function parameters");

        // Data flow should also track parameters
        let mut dataflow_analyzer = DataFlowAnalyzer::new();
        let dataflow_graph = dataflow_analyzer
            .analyze(&ast, &scope_tree)
            .expect("Data flow analysis failed");

        // "value" parameter should be in the graph
        assert!(
            dataflow_graph.nodes.values().any(|n| n.variable == "value"),
            "Parameter 'value' should be in data flow graph"
        );
    }

    #[test]
    fn test_constant_candidate_identification() {
        let ast = create_test_program();

        let mut scope_analyzer = ScopeAnalyzer::new();
        let scope_tree = scope_analyzer.analyze(&ast).expect("Scope analysis failed");

        let mut dataflow_analyzer = DataFlowAnalyzer::new();
        let dataflow_graph = dataflow_analyzer
            .analyze(&ast, &scope_tree)
            .expect("Data flow analysis failed");

        // Variables with single definition are candidates
        let candidates = dataflow_graph.get_constant_candidates();
        assert!(
            !candidates.is_empty(),
            "Should identify constant candidates"
        );

        // "result" is defined once in increment function
        assert!(
            candidates.contains(&"result".to_string()),
            "result should be a constant candidate"
        );
    }

    #[test]
    fn test_called_function_tracking() {
        let ast = create_test_program();

        let mut scope_analyzer = ScopeAnalyzer::new();
        let scope_tree = scope_analyzer.analyze(&ast).expect("Scope analysis failed");

        let mut dataflow_analyzer = DataFlowAnalyzer::new();
        let dataflow_graph = dataflow_analyzer
            .analyze(&ast, &scope_tree)
            .expect("Data flow analysis failed");

        // "log" function is called by "increment"
        let log_uses = dataflow_graph.find_uses("log");
        assert!(
            !log_uses.is_empty(),
            "Called function 'log' should be tracked"
        );
    }

    #[test]
    fn test_scope_tree_consistency() {
        let ast = create_test_program();

        let mut scope_analyzer = ScopeAnalyzer::new();
        let scope_tree = scope_analyzer.analyze(&ast).expect("Scope analysis failed");

        // Verify all scopes have proper parent-child relationships
        for scope in scope_tree.scopes.values() {
            if let Some(parent_id) = &scope.parent {
                assert!(
                    scope_tree.scopes.contains_key(parent_id),
                    "Parent scope should exist"
                );

                // Verify parent has this scope as child
                if let Some(parent) = scope_tree.scopes.get(parent_id) {
                    assert!(
                        parent.children.contains(&scope.id),
                        "Scope should be in parent's children"
                    );
                }
            }
        }
    }

    #[test]
    fn test_shadowing_detection() {
        // Create an AST that demonstrates variable shadowing
        let mut ast = create_test_program();

        // Add a function that shadows global "counter"
        ast.semantic.functions.push(JsFunctionInfo {
            id: "shadowFunc".to_string(),
            name: Some("shadowFunc".to_string()),
            scope_level: 0,
            parameters: vec![],
            return_type_hint: None,
            statement_count: 2,
            cyclomatic_complexity: 1,
            is_async: false,
            is_generator: false,
            captured_vars: vec![],
            local_vars: vec!["counter".to_string()], // Shadows global counter
            called_functions: vec![],
            start_line: 10,
            end_line: 12,
        });

        let mut scope_analyzer = ScopeAnalyzer::new();
        let scope_tree = scope_analyzer.analyze(&ast).expect("Scope analysis failed");

        // Shadowing warning should be generated
        assert!(!scope_tree.get_shadowing_warnings().is_empty());
    }

    #[test]
    fn test_def_use_chain_analysis() {
        let ast = create_test_program();

        let mut scope_analyzer = ScopeAnalyzer::new();
        let scope_tree = scope_analyzer.analyze(&ast).expect("Scope analysis failed");

        let mut dataflow_analyzer = DataFlowAnalyzer::new();
        let dataflow_graph = dataflow_analyzer
            .analyze(&ast, &scope_tree)
            .expect("Data flow analysis failed");

        // For "counter":
        // 1. It's defined globally
        // 2. It's used (captured) in increment function

        let counter_defs = dataflow_graph.find_definitions("counter");
        let counter_uses = dataflow_graph.find_uses("counter");

        assert!(!counter_defs.is_empty(), "counter should have definitions");
        assert!(!counter_uses.is_empty(), "counter should have uses");

        // Verify the chain exists
        // (In a full implementation, we'd verify edges connect def to use)
        assert!(counter_defs.len() <= counter_uses.len() + 1);
    }
}
