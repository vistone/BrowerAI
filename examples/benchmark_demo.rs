/// Performance Benchmark Demo
///
/// This example demonstrates how to use the benchmark tool to measure
/// BrowerAI's parsing performance and compare baseline vs AI-enhanced modes.
use browerai::testing::{BenchmarkConfig, BenchmarkRunner};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("========================================");
    println!("BrowerAI Performance Benchmark");
    println!("========================================\n");

    // Create benchmark runner with default config
    let runner = BenchmarkRunner::new();

    println!("Running performance benchmarks...\n");

    // Run all benchmarks
    let results = runner.run_all_benchmarks()?;

    println!("Running baseline vs AI comparison...\n");

    // Compare baseline vs AI
    let comparisons = runner.compare_baseline_vs_ai()?;

    // Generate and print report
    let report = runner.generate_report(&results, &comparisons);
    println!("{}", report);

    // Save report to file
    std::fs::write("benchmark_report.txt", &report)?;
    println!("Report saved to: benchmark_report.txt");

    Ok(())
}
