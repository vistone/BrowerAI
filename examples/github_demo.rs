use browerai::intelligent_rendering::generation::IntelligentGeneration;
use browerai::intelligent_rendering::reasoning::IntelligentReasoning;
use browerai::intelligent_rendering::renderer::IntelligentRenderer;
use browerai::intelligent_rendering::site_understanding::SiteUnderstanding;
use std::fs;

fn main() -> anyhow::Result<()> {
    println!("========================================");
    println!("BrowerAI GitHub.com Demo");
    println!("========================================\n");

    // Simulate fetching github.com - in reality this would use reqwest
    let github_html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>GitHub: Let's build from here · GitHub</title>
</head>
<body>
    <header class="header-logged-out">
        <div class="header-logo">
            <a href="/">GitHub</a>
        </div>
        <nav>
            <a href="/features">Features</a>
            <a href="/enterprise">Enterprise</a>
            <a href="/pricing">Pricing</a>
        </nav>
        <div class="header-search">
            <form action="/search" method="get">
                <input type="text" name="q" placeholder="Search GitHub" />
                <button type="submit">Search</button>
            </form>
        </div>
        <div class="header-actions">
            <a href="/login">Sign in</a>
            <a href="/signup" class="btn-signup">Sign up</a>
        </div>
    </header>

    <main>
        <section class="hero">
            <h1>Let's build from here</h1>
            <p>The world's leading AI-powered developer platform.</p>
            <div class="hero-actions">
                <form action="/signup" method="post">
                    <input type="email" placeholder="Email address" name="email" />
                    <button type="submit">Sign up for GitHub</button>
                </form>
            </div>
        </section>

        <section class="features">
            <h2>Productivity</h2>
            <div class="feature-grid">
                <article class="feature">
                    <h3>GitHub Copilot</h3>
                    <p>Write better code with AI</p>
                </article>
                <article class="feature">
                    <h3>Actions</h3>
                    <p>Automate your workflow</p>
                </article>
                <article class="feature">
                    <h3>Codespaces</h3>
                    <p>Instant dev environments</p>
                </article>
            </div>
        </section>

        <section class="collaboration">
            <h2>Collaboration</h2>
            <div class="collab-features">
                <div class="feature">
                    <h3>Pull Requests</h3>
                    <p>Review code, share knowledge</p>
                </div>
                <div class="feature">
                    <h3>Issues</h3>
                    <p>Plan and track work</p>
                </div>
                <div class="feature">
                    <h3>Discussions</h3>
                    <p>Collaborate outside of code</p>
                </div>
            </div>
        </section>
    </main>

    <footer>
        <div class="footer-links">
            <a href="/about">About</a>
            <a href="/blog">Blog</a>
            <a href="/careers">Careers</a>
            <a href="/contact">Contact</a>
        </div>
        <p>&copy; 2024 GitHub, Inc.</p>
    </footer>
</body>
</html>"#;

    println!("Step 1: Learning Phase - Analyzing GitHub.com structure...\n");

    // Learn from the content - passing HTML, CSS (empty), JS (empty)
    let understanding = SiteUnderstanding::learn_from_content(
        github_html.to_string(),
        String::new(), // Empty CSS
        String::new(), // Empty JS
    )?;

    println!("✓ Site Understanding Complete:");
    println!("  - Page Type: {:?}", understanding.structure.page_type);
    println!("  - Regions: {}", understanding.structure.regions.len());
    for region in &understanding.structure.regions {
        println!(
            "    • {:?} (importance: {})",
            region.region_type, region.importance
        );
    }
    println!(
        "  - Functionalities: {}",
        understanding.functionalities.len()
    );
    for func in &understanding.functionalities {
        println!("    • {:?}", func.function_type);
    }
    println!(
        "  - Interaction Patterns: {}",
        understanding.interactions.len()
    );
    for pattern in &understanding.interactions {
        println!("    • {:?}", pattern.pattern_type);
    }
    println!();

    println!("Step 2: Reasoning Phase - Identifying optimization opportunities...\n");

    // Create reasoning instance
    let reasoning_instance = IntelligentReasoning::new(understanding);
    let reasoning = reasoning_instance.reason()?;

    println!("✓ Reasoning Complete:");
    println!(
        "  - Core Functions (must preserve): {}",
        reasoning.core_functions.len()
    );
    for func in &reasoning.core_functions {
        println!("    • {} ({:?})", func.name, func.function_type);
    }
    println!(
        "  - Optimizable Regions: {}",
        reasoning.optimizable_regions.len()
    );
    for region in &reasoning.optimizable_regions {
        println!(
            "    • {} ({:?}) - potential improvement: {:.2}%",
            region.region_id,
            region.optimization_type,
            region.potential_improvement * 100.0
        );
    }
    println!(
        "  - Experience Variants: {}",
        reasoning.experience_variants.len()
    );
    for variant in &reasoning.experience_variants {
        println!("    • {}: {:?} layout", variant.name, variant.layout_scheme);
    }
    println!();

    println!("Step 3: Generation Phase - Creating multiple experience variants...\n");

    // Generate variants
    let generation_instance = IntelligentGeneration::new(reasoning);
    let variants = generation_instance.generate()?;

    println!("✓ Generation Complete: {} variants created", variants.len());
    println!();

    // Create output directory
    let output_dir = "/tmp/browerai_github_demo";
    fs::create_dir_all(output_dir)?;
    println!("📁 Output directory: {}\n", output_dir);

    // Save each variant
    for (idx, variant) in variants.iter().enumerate() {
        println!("Variant {}: {} Layout", idx + 1, variant.variant_id);

        // Save HTML
        let html_path = format!(
            "{}/github_{}.html",
            output_dir,
            variant.variant_id.to_lowercase().replace(" ", "_")
        );
        fs::write(&html_path, &variant.html)?;
        println!("  ✓ HTML saved: {}", html_path);

        // Save CSS
        let css_path = format!(
            "{}/github_{}.css",
            output_dir,
            variant.variant_id.to_lowercase().replace(" ", "_")
        );
        fs::write(&css_path, &variant.css)?;
        println!("  ✓ CSS saved: {}", css_path);

        // Save JS
        let js_path = format!(
            "{}/github_{}.js",
            output_dir,
            variant.variant_id.to_lowercase().replace(" ", "_")
        );
        fs::write(&js_path, &variant.bridge_js)?;
        println!("  ✓ JS bridge saved: {}", js_path);

        // Show function validation
        println!(
            "  ✓ Functions validated: {}",
            if variant.function_validation.all_functions_present {
                "All present"
            } else {
                "Some missing"
            }
        );
        println!(
            "    - Function mappings: {}",
            variant.function_validation.function_map.len()
        );
        println!(
            "    - Interaction tests: {} passed",
            variant
                .function_validation
                .interaction_tests
                .iter()
                .filter(|t| t.passed)
                .count()
        );
        println!();
    }

    println!("Step 4: Rendering Phase - Creating final renderable page...\n");

    // Render the first variant
    let renderer = IntelligentRenderer::new(variants[0].clone(), variants.clone());
    let rendered = renderer.render()?;

    println!("✓ Rendering Complete:");
    println!("  - Final HTML size: {} bytes", rendered.stats.html_size);
    println!("  - Final CSS size: {} bytes", rendered.stats.css_size);
    println!("  - Final JS size: {} bytes", rendered.stats.js_size);
    println!(
        "  - Functions bridged: {}",
        rendered.stats.functions_bridged
    );
    println!();

    // Save final rendered page
    let final_path = format!("{}/github_final.html", output_dir);
    fs::write(&final_path, &rendered.final_html)?;
    let final_css_path = format!("{}/github_final.css", output_dir);
    fs::write(&final_css_path, &rendered.final_css)?;
    let final_js_path = format!("{}/github_final.js", output_dir);
    fs::write(&final_js_path, &rendered.final_js)?;
    println!("✓ Final page saved:");
    println!("  - {}", final_path);
    println!("  - {}", final_css_path);
    println!("  - {}", final_js_path);
    println!();

    // Show a preview of the generated HTML
    println!("========================================");
    println!("Preview of Generated HTML (First Variant):");
    println!("========================================");
    let preview: String = variants[0].html.chars().take(800).collect();
    println!("{}", preview);
    if variants[0].html.len() > 800 {
        println!("... ({} more characters)", variants[0].html.len() - 800);
    }
    println!();

    // Show CSS preview
    println!("========================================");
    println!("Preview of Generated CSS:");
    println!("========================================");
    let css_preview: String = variants[0].css.chars().take(600).collect();
    println!("{}", css_preview);
    if variants[0].css.len() > 600 {
        println!("... ({} more characters)", variants[0].css.len() - 600);
    }
    println!();

    // Show JS bridge preview
    println!("========================================");
    println!("Preview of Function Bridge JavaScript:");
    println!("========================================");
    let js_preview: String = variants[0].bridge_js.chars().take(600).collect();
    println!("{}", js_preview);
    if variants[0].bridge_js.len() > 600 {
        println!(
            "... ({} more characters)",
            variants[0].bridge_js.len() - 600
        );
    }
    println!();

    println!("========================================");
    println!("✅ Demo Complete!");
    println!("========================================");
    println!();
    println!("What happened:");
    println!("1. ✓ Learned GitHub.com structure (functions, layout, interactions)");
    println!("2. ✓ Reasoned about core functions that must be preserved");
    println!(
        "3. ✓ Generated {} experience variants with different layouts",
        variants.len()
    );
    println!("4. ✓ Created function bridges to preserve all functionality");
    println!("5. ✓ Rendered final page ready for display");
    println!();
    println!("All original GitHub.com functions are preserved:");
    println!("  • Search functionality");
    println!("  • Navigation links");
    println!("  • Sign in/Sign up actions");
    println!("  • Form submissions");
    println!();
    println!(
        "But users get {} different layout experiences!",
        variants.len()
    );
    println!();
    println!("📂 Open files in browser:");
    for variant in &variants {
        let filename = variant.variant_id.to_lowercase().replace(" ", "_");
        println!("   file://{}/github_{}.html", output_dir, filename);
    }
    println!(
        "   file://{}/github_final.html (fully rendered)",
        output_dir
    );
    println!();

    Ok(())
}
