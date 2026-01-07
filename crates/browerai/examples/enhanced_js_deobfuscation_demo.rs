/// Enhanced JavaScript Deobfuscation Demo
///
/// Demonstrates the new enhanced deobfuscation techniques learned from
/// popular GitHub projects (webcrack, synchrony, decode-js, etc.)
///
/// Run with:
/// ```bash
/// cargo run --example enhanced_js_deobfuscation_demo
/// ```

use browerai::learning::EnhancedDeobfuscator;

fn main() -> anyhow::Result<()> {
    println!("==============================================");
    println!("ENHANCED JAVASCRIPT DEOBFUSCATION DEMO");
    println!("==============================================");
    println!("\nBased on techniques from:");
    println!("  - webcrack (j4k0xb)");
    println!("  - synchrony (relative)");
    println!("  - decode-js (echo094)");
    println!("  - javascript-deobfuscator (ben-sb)");
    println!();

    let mut deobfuscator = EnhancedDeobfuscator::new();

    // Test 1: String Array Unpacking
    println!("=== Test 1: String Array Unpacking ===");
    let obfuscated_1 = r#"
var _0xabc = ['Hello', 'World', 'JavaScript'];
console.log(_0xabc[0x0] + ' ' + _0xabc[0x1] + ' ' + _0xabc[0x2]);
"#;

    println!("Before:");
    println!("{}", obfuscated_1);

    let result_1 = deobfuscator.deobfuscate(obfuscated_1)?;
    println!("After:");
    println!("{}", result_1.code);
    println!("Transformations: {}", result_1.transformations.len());
    println!("String arrays unpacked: {}", result_1.stats.string_arrays_unpacked);
    println!();

    // Test 2: Proxy Function Removal
    println!("=== Test 2: Proxy Function Removal ===");
    let obfuscated_2 = r#"
function _proxy(a, b) {
    return console.log(a, b);
}
function _add(x, y) {
    return x + y;
}
_proxy('Result:', _add(5, 10));
"#;

    println!("Before:");
    println!("{}", obfuscated_2);

    let result_2 = deobfuscator.deobfuscate(obfuscated_2)?;
    println!("After:");
    println!("{}", result_2.code);
    println!("Proxy functions removed: {}", result_2.stats.proxy_functions_removed);
    println!();

    // Test 3: Self-Defending Code Removal
    println!("=== Test 3: Self-Defending Code Removal ===");
    let obfuscated_3 = r#"
debugger;
console.log = function() { return null; };
if (window.outerHeight - window.innerHeight > 100) {
    throw new Error('DevTools detected!');
}
var message = 'Hello World';
console.log(message);
"#;

    println!("Before:");
    println!("{}", obfuscated_3);

    let result_3 = deobfuscator.deobfuscate(obfuscated_3)?;
    println!("After:");
    println!("{}", result_3.code);
    println!("Self-defending patterns removed: {}", result_3.stats.self_defending_removed);
    println!();

    // Test 4: Opaque Predicate Simplification
    println!("=== Test 4: Opaque Predicate Simplification ===");
    let obfuscated_4 = r#"
if (true) {
    console.log('This is always executed');
}
if (false) {
    console.log('This is never executed');
    doEvil();
}
var x = !![];
var y = ![];
"#;

    println!("Before:");
    println!("{}", obfuscated_4);

    let result_4 = deobfuscator.deobfuscate(obfuscated_4)?;
    println!("After:");
    println!("{}", result_4.code);
    println!();

    // Test 5: Constant Folding
    println!("=== Test 5: Constant Folding ===");
    let obfuscated_5 = r#"
var x = 0x10;
var y = 0xFF;
var sum = 10 + 20;
console.log(x, y, sum);
"#;

    println!("Before:");
    println!("{}", obfuscated_5);

    let result_5 = deobfuscator.deobfuscate(obfuscated_5)?;
    println!("After:");
    println!("{}", result_5.code);
    println!("Constants folded: {}", result_5.stats.constants_folded);
    println!();

    // Test 6: Member Expression Simplification
    println!("=== Test 6: Member Expression Simplification ===");
    let obfuscated_6 = r#"
obj["property"]["nested"]["value"] = 42;
console.log(obj["property"]["nested"]["value"]);
"#;

    println!("Before:");
    println!("{}", obfuscated_6);

    let result_6 = deobfuscator.deobfuscate(obfuscated_6)?;
    println!("After:");
    println!("{}", result_6.code);
    println!();

    // Test 7: Comprehensive Obfuscation
    println!("=== Test 7: Comprehensive Obfuscation ===");
    let obfuscated_7 = r#"
var _0xstr = ['log', 'Hello', 'from', 'obfuscated', 'code'];
function _p(f) { return console[f]; }
debugger;
if (true) {
    _p(_0xstr[0x0])(_0xstr[0x1] + ' ' + _0xstr[0x2] + ' ' + _0xstr[0x3] + ' ' + _0xstr[0x4]);
}
if (false) {
    doEvil();
}
var result = 0x10 + 0x20;
obj["method"]();
"#;

    println!("Before:");
    println!("{}", obfuscated_7);

    let result_7 = deobfuscator.deobfuscate(obfuscated_7)?;
    println!("After:");
    println!("{}", result_7.code);
    println!("\n--- Statistics ---");
    println!("Transformations applied: {}", result_7.transformations.len());
    for (i, transform) in result_7.transformations.iter().enumerate() {
        println!("  {}. {}", i + 1, transform);
    }
    println!("\n--- Detailed Stats ---");
    println!("String arrays unpacked: {}", result_7.stats.string_arrays_unpacked);
    println!("Proxy functions removed: {}", result_7.stats.proxy_functions_removed);
    println!("Control flow simplified: {}", result_7.stats.control_flow_simplified);
    println!("Constants folded: {}", result_7.stats.constants_folded);
    println!("Self-defending patterns removed: {}", result_7.stats.self_defending_removed);
    println!("Size reduction: {} bytes", result_7.stats.size_reduction);
    println!(
        "Readability improvement: {:.1}%",
        result_7.stats.readability_improvement * 100.0
    );
    println!();

    // Test 8: Real-world Obfuscator.io Sample
    println!("=== Test 8: Simulated Obfuscator.io Pattern ===");
    let obfuscated_8 = r#"
var _0x1234 = ['message', 'log', 'Hello World'];
(function(_0x456, _0x789) {
    var _0xabc = function(_0xdef) {
        while (true) {
            try {
                console[_0x1234[0x1]](_0x1234[0x2]);
                break;
            } catch (e) {}
        }
    };
    return _0xabc();
}());
"#;

    println!("Before:");
    println!("{}", obfuscated_8);

    let result_8 = deobfuscator.deobfuscate(obfuscated_8)?;
    println!("After:");
    println!("{}", result_8.code);
    println!();

    println!("==============================================");
    println!("SUMMARY");
    println!("==============================================");
    println!("✓ Demonstrated 8 different deobfuscation techniques");
    println!("✓ String array unpacking and replacement");
    println!("✓ Proxy function detection and removal");
    println!("✓ Self-defending code removal (debugger, console hijacking)");
    println!("✓ Opaque predicate simplification");
    println!("✓ Constant folding (hex to decimal)");
    println!("✓ Member expression simplification");
    println!("✓ Multi-pass iterative deobfuscation");
    println!();
    println!("These techniques were learned from popular GitHub projects:");
    println!("  - webcrack: String array rotation, decoder wrappers");
    println!("  - synchrony: Control flow unflattening");
    println!("  - decode-js: AST-based transformations");
    println!("  - ben-sb/javascript-deobfuscator: Proxy function removal");
    println!();
    println!("Key Features:");
    println!("  • Multi-pass deobfuscation with convergence detection");
    println!("  • Safe transformations that preserve code semantics");
    println!("  • Detailed statistics and transformation tracking");
    println!("  • Extensible architecture for new patterns");
    println!();

    Ok(())
}
