/// Demonstration of advanced JavaScript deobfuscation
///
/// Shows how BrowerAI handles:
/// - Modern framework bundling (Webpack, React, Vue)
/// - Dynamic HTML injection via JavaScript
/// - Event-driven content loading
/// - Template extraction from obfuscated code
use browerai::learning::{AdvancedDeobfuscator, FrameworkObfuscation};
use std::fs;

fn main() -> anyhow::Result<()> {
    println!("========================================");
    println!("ADVANCED JAVASCRIPT DEOBFUSCATION DEMO");
    println!("========================================\n");

    let deob = AdvancedDeobfuscator::new();

    // Demo 1: Webpack bundled code
    println!("Demo 1: Webpack Bundled Code");
    println!("----------------------------");
    let webpack_code = r#"
(function(modules) {
    function __webpack_require__(moduleId) {
        var module = { exports: {} };
        modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
        return module.exports;
    }
    __webpack_require__(0);
})([
    function(module, exports) {
        const template = `<div class="app"><h1>My App</h1><p>Content</p></div>`;
        document.getElementById('root').innerHTML = template;
    }
]);
    "#;

    let analysis = deob.analyze(webpack_code)?;
    println!("✓ Detected patterns: {:?}", analysis.framework_patterns);
    println!("✓ Found {} templates", analysis.templates.len());
    println!("✓ Confidence: {:.2}", analysis.confidence);

    let result = deob.deobfuscate(webpack_code)?;
    println!("✓ Deobfuscation success: {}", result.success);
    println!("✓ Extracted {} HTML templates", result.html_templates.len());
    for (i, template) in result.html_templates.iter().enumerate() {
        println!("  Template {}: {}", i + 1, template);
    }
    println!("✓ Applied {} steps", result.steps.len());
    println!();

    // Demo 2: React compiled code
    println!("Demo 2: React JSX Compiled Code");
    println!("--------------------------------");
    let react_code = r#"
function App() {
    return React.createElement('div', { className: 'container' },
        React.createElement('h1', null, 'Welcome'),
        React.createElement('button', { 
            onClick: function() { 
                document.getElementById('content').innerHTML = '<p>Loaded dynamically!</p>';
            }
        }, 'Load Content')
    );
}
    "#;

    let analysis = deob.analyze(react_code)?;
    println!("✓ Detected patterns: {:?}", analysis.framework_patterns);
    println!(
        "✓ Found {} injection points",
        analysis.dynamic_injection_points.len()
    );
    for point in &analysis.dynamic_injection_points {
        println!(
            "  - Line {}: {} targeting '{}'",
            point.line, point.method, point.target
        );
    }
    println!();

    // Demo 3: Dynamic HTML injection
    println!("Demo 3: Dynamic HTML Injection");
    println!("------------------------------");
    let dynamic_code = r#"
function loadContent() {
    const header = document.createElement('header');
    header.innerHTML = `<nav><a href="/">Home</a><a href="/about">About</a></nav>`;
    document.body.appendChild(header);
    
    const main = document.getElementById('main');
    main.innerHTML = `
        <article>
            <h2>Article Title</h2>
            <p>Article content goes here...</p>
        </article>
    `;
}

document.addEventListener('DOMContentLoaded', loadContent);
    "#;

    let analysis = deob.analyze(dynamic_code)?;
    println!("✓ Detected patterns: {:?}", analysis.framework_patterns);
    println!(
        "✓ Found {} injection points",
        analysis.dynamic_injection_points.len()
    );
    println!("✓ Found {} event loaders", analysis.event_loaders.len());

    let result = deob.deobfuscate(dynamic_code)?;
    println!(
        "✓ Extracted {} HTML templates:",
        result.html_templates.len()
    );
    for (i, template) in result.html_templates.iter().enumerate() {
        let preview = if template.len() > 60 {
            format!("{}...", &template[..60])
        } else {
            template.clone()
        };
        println!("  Template {}: {}", i + 1, preview);
    }
    println!("✓ Event mappings: {}", result.event_content_map.len());
    for (event, content) in &result.event_content_map {
        let preview = if content.len() > 40 {
            format!("{}...", &content[..40])
        } else {
            content.clone()
        };
        println!("  - {} → {}", event, preview);
    }
    println!();

    // Demo 4: Vue template compilation
    println!("Demo 4: Vue Template (Simulated)");
    println!("---------------------------------");
    let vue_code = r#"
const render = function() {
    return _createElementVNode("div", { class: "vue-app" }, [
        _createElementVNode("h1", null, "Vue App"),
        _createElementVNode("ul", null, [
            _createElementVNode("li", null, "Item 1"),
            _createElementVNode("li", null, "Item 2")
        ])
    ]);
};
    "#;

    let analysis = deob.analyze(vue_code)?;
    println!("✓ Detected patterns: {:?}", analysis.framework_patterns);
    println!();

    // Demo 5: Complex event-driven scenario
    println!("Demo 5: Event-Driven Content Loading");
    println!("------------------------------------");
    let event_code = r#"
const buttons = document.querySelectorAll('.load-btn');

buttons.forEach(btn => {
    btn.addEventListener('click', function(e) {
        const targetId = e.target.dataset.target;
        const content = `<div class="loaded"><h3>Dynamic Content</h3><p>Loaded for ${targetId}</p></div>`;
        document.getElementById(targetId).innerHTML = content;
    });
});

window.addEventListener('scroll', function() {
    if (window.scrollY > 500) {
        const footer = `<footer><p>&copy; 2024 Company</p></footer>`;
        document.body.insertAdjacentHTML('beforeend', footer);
    }
});
    "#;

    let analysis = deob.analyze(event_code)?;
    println!("✓ Detected {} event loaders", analysis.event_loaders.len());
    for loader in &analysis.event_loaders {
        println!(
            "  - Event '{}' calls '{}'",
            loader.event_type, loader.function
        );
        println!("    Condition: {}", loader.trigger_condition);
    }

    let result = deob.deobfuscate(event_code)?;
    println!("✓ Extracted {} HTML snippets", result.html_templates.len());
    println!();

    // Save results to files
    let output_dir = "/tmp/browerai_advanced_deobfuscation";
    fs::create_dir_all(output_dir)?;

    fs::write(format!("{}/webpack_original.js", output_dir), webpack_code)?;
    fs::write(
        format!("{}/webpack_deobfuscated.js", output_dir),
        result.javascript.clone(),
    )?;

    fs::write(format!("{}/dynamic_original.js", output_dir), dynamic_code)?;

    // Save extracted templates
    for (i, template) in result.html_templates.iter().enumerate() {
        fs::write(
            format!("{}/extracted_template_{}.html", output_dir, i + 1),
            template,
        )?;
    }

    println!("========================================");
    println!("SUMMARY");
    println!("========================================");
    println!("✓ Demonstrated 5 complex scenarios");
    println!("✓ Detected multiple framework patterns");
    println!("✓ Extracted HTML from JavaScript");
    println!("✓ Mapped event handlers to content");
    println!("✓ Results saved to: {}", output_dir);
    println!();
    println!("Key Capabilities:");
    println!("  • Webpack bundle unwrapping");
    println!("  • React/Vue framework detection");
    println!("  • Dynamic HTML extraction");
    println!("  • Event-driven content mapping");
    println!("  • Template literal parsing");
    println!();
    println!("This demonstrates BrowerAI's ability to handle");
    println!("modern JavaScript frameworks and dynamically");
    println!("generated content - not just static HTML!");

    Ok(())
}
