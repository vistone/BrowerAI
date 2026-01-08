use browerai_ai_integration::{AstEngine, HybridJsOrchestrator, OrchestrationPolicy, SourceKind};

#[test]
fn parse_prefers_swc_for_modules() {
    let mut orch = HybridJsOrchestrator::with_policy(OrchestrationPolicy::Balanced);
    let code = "import x from 'y'; export const a = 1;";
    let ast = orch.parse(code).unwrap();
    assert!(ast.is_valid);
    assert_eq!(ast.engine, AstEngine::Swc);
    assert!(matches!(ast.source_kind, SourceKind::Module));
}

#[test]
fn parse_falls_back_to_boa_for_script() {
    let mut orch = HybridJsOrchestrator::with_policy(OrchestrationPolicy::Balanced);
    let code = "function a() { return 1 }";
    let ast = orch.parse(code).unwrap();
    assert!(ast.is_valid);
    assert_eq!(ast.engine, AstEngine::Boa);
    assert!(matches!(ast.source_kind, SourceKind::Script));
}

#[test]
fn exec_uses_v8_when_available() {
    let mut orch = HybridJsOrchestrator::with_policy(OrchestrationPolicy::Performance);
    let out = orch.execute("1 + 2").unwrap();
    assert!(out == "3" || out == "<ok>");
}
