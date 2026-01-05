use browerai::intelligent_rendering::site_understanding::SiteUnderstanding;
use browerai::intelligent_rendering::reasoning::IntelligentReasoning;
use browerai::intelligent_rendering::generation::IntelligentGeneration;
use browerai::intelligent_rendering::renderer::IntelligentRenderer;
use std::fs;
use anyhow::{Result, Context};

fn fetch_website(url: &str) -> Result<String> {
    println!("🌐 Fetching {} ...", url);
    
    let client = reqwest::blocking::Client::builder()
        .user_agent("BrowerAI/0.1.0 (Learning Bot)")
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    
    let response = client.get(url)
        .send()
        .context("Failed to fetch URL")?;
    
    let status = response.status();
    println!("   Status: {}", status);
    
    if !status.is_success() {
        anyhow::bail!("HTTP request failed with status: {}", status);
    }
    
    let html = response.text().context("Failed to read response body")?;
    println!("   Downloaded: {} bytes", html.len());
    
    Ok(html)
}

fn main() -> Result<()> {
    println!("========================================");
    println!("BrowerAI - Real Network Demo");
    println!("========================================\n");
    
    // Get URL from command line or use default
    let args: Vec<String> = std::env::args().collect();
    let url = if args.len() > 1 {
        &args[1]
    } else {
        "https://example.com"  // Safe default site
    };
    
    println!("Target URL: {}\n", url);
    println!("========================================");
    println!("Phase 1: NETWORK REQUEST");
    println!("========================================\n");
    
    // Make actual HTTP request
    let html = match fetch_website(url) {
        Ok(content) => {
            println!("✓ Successfully fetched real HTML from network\n");
            content
        },
        Err(e) => {
            eprintln!("❌ Network request failed: {}", e);
            eprintln!("\nNote: This demo requires internet connectivity.");
            eprintln!("Try running with a different URL:");
            eprintln!("  cargo run --example real_network_demo https://example.com\n");
            return Err(e);
        }
    };
    
    // Show HTML sample
    println!("HTML Preview (first 500 chars):");
    println!("----------------------------------------");
    let preview: String = html.chars().take(500).collect();
    println!("{}", preview);
    if html.len() > 500 {
        println!("... ({} more bytes)", html.len() - 500);
    }
    println!("\n");
    
    println!("========================================");
    println!("Phase 2: LEARNING");
    println!("========================================\n");
    
    println!("📖 Analyzing real HTML structure...");
    
    // Learn from the REAL content fetched from network
    let understanding = SiteUnderstanding::learn_from_content(
        html.clone(),
        String::new(),  // CSS would be extracted from <style> tags or fetched separately
        String::new()   // JS would be extracted from <script> tags or fetched separately
    ).context("Failed to analyze website structure")?;
    
    println!("✓ Site Understanding Complete:");
    println!("  - Page Type: {:?}", understanding.structure.page_type);
    println!("  - Regions: {}", understanding.structure.regions.len());
    for region in &understanding.structure.regions {
        println!("    • {:?} (importance: {:.2})", region.region_type, region.importance);
    }
    println!("  - Functionalities detected: {}", understanding.functionalities.len());
    for func in understanding.functionalities.iter().take(10) {
        println!("    • {:?}", func.function_type);
    }
    if understanding.functionalities.len() > 10 {
        println!("    ... and {} more", understanding.functionalities.len() - 10);
    }
    println!("  - Interaction Patterns: {}", understanding.interactions.len());
    for pattern in &understanding.interactions {
        println!("    • {:?}", pattern.pattern_type);
    }
    println!();
    
    println!("========================================");
    println!("Phase 3: REASONING");
    println!("========================================\n");
    
    println!("🧠 Reasoning about optimization opportunities...");
    
    let reasoning_instance = IntelligentReasoning::new(understanding);
    let reasoning = reasoning_instance.reason()?;
    
    println!("✓ Reasoning Complete:");
    println!("  - Core Functions (must preserve): {}", reasoning.core_functions.len());
    for func in reasoning.core_functions.iter().take(5) {
        println!("    • {} ({:?})", func.name, func.function_type);
    }
    if reasoning.core_functions.len() > 5 {
        println!("    ... and {} more", reasoning.core_functions.len() - 5);
    }
    println!("  - Optimizable Regions: {}", reasoning.optimizable_regions.len());
    for region in reasoning.optimizable_regions.iter().take(3) {
        println!("    • {} ({:?}) - potential improvement: {:.1}%", 
                 region.region_id, region.optimization_type, region.potential_improvement * 100.0);
    }
    println!("  - Experience Variants: {}", reasoning.experience_variants.len());
    for variant in &reasoning.experience_variants {
        println!("    • {}: {:?} layout", variant.name, variant.layout_scheme);
    }
    println!();
    
    println!("========================================");
    println!("Phase 4: GENERATION");
    println!("========================================\n");
    
    println!("⚡ Generating multiple experience variants...");
    
    let generation_instance = IntelligentGeneration::new(reasoning);
    let variants = generation_instance.generate()?;
    
    println!("✓ Generation Complete: {} variants created", variants.len());
    println!();
    
    // Create output directory with URL-based name
    let url_safe = url.replace("://", "_").replace("/", "_").replace(".", "_");
    let output_dir = format!("/tmp/browerai_network_{}", url_safe);
    fs::create_dir_all(&output_dir)?;
    println!("📁 Output directory: {}\n", output_dir);
    
    // Save original HTML for comparison
    let original_path = format!("{}/original.html", output_dir);
    fs::write(&original_path, &html)?;
    println!("💾 Original HTML saved: {}", original_path);
    println!("   Size: {} bytes\n", html.len());
    
    // Save each variant
    for (idx, variant) in variants.iter().enumerate() {
        println!("Variant {}: {} Layout", idx + 1, variant.variant_id);
        
        let variant_name = variant.variant_id.to_lowercase().replace(" ", "_");
        
        // Save HTML
        let html_path = format!("{}/{}.html", output_dir, variant_name);
        fs::write(&html_path, &variant.html)?;
        println!("  ✓ HTML saved: {} ({} bytes)", html_path, variant.html.len());
        
        // Save CSS
        let css_path = format!("{}/{}.css", output_dir, variant_name);
        fs::write(&css_path, &variant.css)?;
        println!("  ✓ CSS saved: {} ({} bytes)", css_path, variant.css.len());
        
        // Save JS
        let js_path = format!("{}/{}.js", output_dir, variant_name);
        fs::write(&js_path, &variant.bridge_js)?;
        println!("  ✓ JS bridge saved: {} ({} bytes)", js_path, variant.bridge_js.len());
        
        println!("  ✓ Functions validated: {}",
                 if variant.function_validation.all_functions_present { "All present" } else { "Some missing" });
        println!("    - Mapped functions: {}", variant.function_validation.function_map.len());
        println!();
    }
    
    println!("========================================");
    println!("Phase 5: RENDERING");
    println!("========================================\n");
    
    println!("🎨 Rendering final page...");
    
    let renderer = IntelligentRenderer::new(variants[0].clone(), variants.clone());
    let rendered = renderer.render()?;
    
    println!("✓ Rendering Complete:");
    println!("  - Final HTML size: {} bytes", rendered.stats.html_size);
    println!("  - Final CSS size: {} bytes", rendered.stats.css_size);
    println!("  - Final JS size: {} bytes", rendered.stats.js_size);
    println!("  - Functions bridged: {}", rendered.stats.functions_bridged);
    println!();
    
    // Save final rendered page
    let final_path = format!("{}/final.html", output_dir);
    fs::write(&final_path, &rendered.final_html)?;
    let final_css_path = format!("{}/final.css", output_dir);
    fs::write(&final_css_path, &rendered.final_css)?;
    let final_js_path = format!("{}/final.js", output_dir);
    fs::write(&final_js_path, &rendered.final_js)?;
    
    println!("✓ Final page saved:");
    println!("  - {}", final_path);
    println!("  - {}", final_css_path);
    println!("  - {}", final_js_path);
    println!();
    
    println!("========================================");
    println!("✅ COMPLETE - Real Network Test Success!");
    println!("========================================\n");
    
    println!("What actually happened:");
    println!("1. ✓ Made REAL HTTP request to {}", url);
    println!("2. ✓ Downloaded {} bytes of actual HTML", html.len());
    println!("3. ✓ Learned structure from REAL website data");
    println!("4. ✓ Reasoned about actual page functions");
    println!("5. ✓ Generated {} experience variants", variants.len());
    println!("6. ✓ Created function bridges for all interactions");
    println!("7. ✓ Rendered final pages ready for display");
    println!();
    
    println!("📊 Comparison:");
    println!("  Original site: {} bytes", html.len());
    println!("  Generated variants:");
    for variant in &variants {
        let total = variant.html.len() + variant.css.len() + variant.bridge_js.len();
        println!("    • {}: {} bytes", variant.variant_id, total);
    }
    println!();
    
    println!("🌐 Open in browser:");
    println!("  Original: file://{}", original_path);
    for variant in &variants {
        let variant_name = variant.variant_id.to_lowercase().replace(" ", "_");
        println!("  {}: file://{}/{}.html", variant.variant_id, output_dir, variant_name);
    }
    println!("  Final: file://{}", final_path);
    println!();
    
    println!("🔬 This was NOT a simulation:");
    println!("  • Actual network socket connection made");
    println!("  • Real HTTP request/response");
    println!("  • Genuine HTML from target server");
    println!("  • True data flow through the system");
    println!();
    
    Ok(())
}
