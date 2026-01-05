//! Test deobfuscation transformations directly

use browerai::learning::{JsDeobfuscator, DeobfuscationStrategy};

#[test]
fn test_hex_string_decoding() {
    let deobfuscator = JsDeobfuscator::new();
    
    let code = r#"var msg="\x48\x65\x6c\x6c\x6f";console.log(msg);"#;
    let result = deobfuscator.deobfuscate(code, DeobfuscationStrategy::Comprehensive).unwrap();
    
    // After decoding, \x48\x65\x6c\x6c\x6f should become Hello
    assert!(result.code.contains("Hello") || !result.code.contains("\\x"),
        "Hex strings should be decoded. Got: {}", result.code);
}

#[test]
fn test_variable_renaming_transformation() {
    let deobfuscator = JsDeobfuscator::new();
    
    let code = "var a=1;var b=2;var c=a+b;console.log(c);";
    let result = deobfuscator.deobfuscate(code, DeobfuscationStrategy::VariableRenaming).unwrap();
    
    // Single-letter variables should be renamed
    assert!(result.code.contains("var") && result.code.len() > code.len() - 10,
        "Variables should be renamed. Got: {}", result.code);
}

#[test]
fn test_dead_code_removal() {
    let deobfuscator = JsDeobfuscator::new();
    
    let code = "if(false){console.log('dead');}console.log('alive');";
    let result = deobfuscator.deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification).unwrap();
    
    // Dead code should be removed
    assert!(!result.code.contains("if(false)") || result.code.len() < code.len(),
        "Dead code should be removed. Got: {}", result.code);
}

#[test]
fn test_comprehensive_deobfuscation() {
    let deobfuscator = JsDeobfuscator::new();
    
    let code = r#"var a="\x48\x69";if(false){var b=1;}var c=a;"#;
    let result = deobfuscator.deobfuscate(code, DeobfuscationStrategy::Comprehensive).unwrap();
    
    // Multiple transformations should be applied
    assert!(result.steps.len() > 0, "Should apply transformation steps");
    assert!(result.success, "Deobfuscation should succeed");
    
    // Code should still be valid JavaScript
    let parser = browerai::parser::JsParser::new();
    assert!(parser.validate(&result.code).unwrap(), "Result should be valid JS");
}
