use browerai::GpuConfig;

fn main() {
    env_logger::init();
    
    println!("Testing GPU detection...");
    
    let config = GpuConfig::default();
    let providers = config.detect_available_providers();
    
    println!("Available providers: {} found", providers.len());
    for (i, provider) in providers.iter().enumerate() {
        println!("  {}. {:?}", i + 1, provider);
    }
    
    if providers.len() > 1 {
        println!("\n✓ GPU acceleration available!");
    } else {
        println!("\n⊘ No GPU detected, CPU-only mode");
    }
}
