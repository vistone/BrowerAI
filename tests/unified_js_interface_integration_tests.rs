use browerai::prelude::*;

#[test]
fn test_unified_js_interface_basic_execution() {
    let mut interface = UnifiedJsInterface::new();
    let result = interface.execute_for_render("2 + 2");
    assert!(result.is_ok());
}

#[test]
fn test_unified_js_interface_module_analysis() {
    let mut interface = UnifiedJsInterface::new();
    let result = interface.parse_for_analysis("import x from 'y'; export const a = 1;");
    assert!(result.is_ok());
    let analysis = result.unwrap();
    assert!(analysis.is_module);
}

#[test]
fn test_unified_js_interface_validation() {
    let mut interface = UnifiedJsInterface::new();
    let result = interface.quick_validate("const x = 1;");
    assert!(result.is_ok());
    assert!(result.unwrap());
}

#[test]
fn test_unified_js_interface_empty_code() {
    let mut interface = UnifiedJsInterface::new();
    let result = interface.quick_validate("");
    assert!(result.is_ok());
}
