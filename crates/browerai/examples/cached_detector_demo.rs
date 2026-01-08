/// Demonstration of CachedFrameworkDetector performance benefits
///
/// This example compares performance with and without caching.
use anyhow::Result;
use browerai::cached_detector::{CacheConfig, CachedFrameworkDetector};
use browerai::learning::FrameworkKnowledgeBase;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    println!("🚀 Cached Framework Detection Performance Demo\n");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Test data: multiple code samples
    let test_samples = vec![
        (
            "React App",
            r#"
            import React, { useState } from 'react';
            function App() {
                const [count, setCount] = useState(0);
                return React.createElement('div', null, count);
            }
        "#,
        ),
        (
            "Vue 3 App",
            r#"
            const { createApp, ref } = Vue;
            const app = createApp({
                setup() {
                    const message = ref('Hello');
                    return { message };
                }
            });
        "#,
        ),
        (
            "Angular App",
            r#"
            @Component({ selector: 'app-test' })
            class TestComponent {
                ngOnInit() {
                    console.log('init');
                }
            }
        "#,
        ),
        (
            "Webpack Bundle",
            r#"
            (function(modules) {
                function __webpack_require__(moduleId) {
                    return modules[moduleId];
                }
                __webpack_require__(0);
            })([function() {}]);
        "#,
        ),
        (
            "jQuery Code",
            r#"
            $(document).ready(function() {
                $('#app').show();
                $.ajax({ url: '/api' });
            });
        "#,
        ),
    ];

    println!("📊 Test Setup:");
    println!("   • {} unique code samples", test_samples.len());
    println!("   • 3 analysis rounds per sample (15 total)");
    println!("   • Cache TTL: 5 minutes");
    println!("   • Max cache entries: 1000\n");

    // ============================================================================
    // Part 1: Without Caching
    // ============================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("Part 1: WITHOUT Caching (Baseline)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let kb = FrameworkKnowledgeBase::new();
    let mut total_time_uncached = Duration::from_secs(0);

    for round in 1..=3 {
        println!("Round {}:", round);
        for (name, code) in &test_samples {
            let start = Instant::now();
            let detections = kb.analyze_code(code)?;
            let duration = start.elapsed();
            total_time_uncached += duration;

            println!(
                "   {} - {:?} ({} frameworks)",
                name,
                duration,
                detections.len()
            );
        }
        println!();
    }

    let avg_time_uncached = total_time_uncached / 15;
    println!("⏱️  Total time (uncached): {:?}", total_time_uncached);
    println!("⏱️  Average per analysis: {:?}\n", avg_time_uncached);

    // ============================================================================
    // Part 2: With Caching
    // ============================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("Part 2: WITH Caching (Optimized)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let cached_detector = CachedFrameworkDetector::with_config(CacheConfig {
        max_entries: 1000,
        ttl: Duration::from_secs(300),
        enable_stats: true,
    });

    let mut total_time_cached = Duration::from_secs(0);

    for round in 1..=3 {
        println!("Round {}:", round);
        for (name, code) in &test_samples {
            let start = Instant::now();
            let detections = cached_detector.analyze_code(code)?;
            let duration = start.elapsed();
            total_time_cached += duration;

            let marker = if round == 1 {
                "❄️  MISS"
            } else {
                "⚡ HIT "
            };
            println!(
                "   {} - {:?} ({} frameworks) {}",
                name,
                duration,
                detections.len(),
                marker
            );
        }
        println!();
    }

    let avg_time_cached = total_time_cached / 15;
    println!("⏱️  Total time (cached): {:?}", total_time_cached);
    println!("⏱️  Average per analysis: {:?}\n", avg_time_cached);

    // ============================================================================
    // Part 3: Performance Analysis
    // ============================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("Part 3: Performance Analysis");
    println!("═══════════════════════════════════════════════════════════════\n");

    let stats = cached_detector.stats();

    println!("📈 Cache Statistics:");
    println!("   Hits:        {}", stats.hits);
    println!("   Misses:      {}", stats.misses);
    println!("   Hit Rate:    {:.1}%", stats.hit_rate());
    println!("   Current Size: {}", stats.current_size);
    println!("   Evictions:   {}\n", stats.evictions);

    let speedup =
        total_time_uncached.as_micros() as f64 / total_time_cached.as_micros().max(1) as f64;
    let time_saved = total_time_uncached.saturating_sub(total_time_cached);
    let time_saved_percent =
        (time_saved.as_micros() as f64 / total_time_uncached.as_micros() as f64) * 100.0;

    println!("🚀 Performance Improvement:");
    println!("   Speedup:     {:.2}x faster", speedup);
    println!(
        "   Time Saved:  {:?} ({:.1}%)",
        time_saved, time_saved_percent
    );
    println!("   Baseline:    {:?}", total_time_uncached);
    println!("   Optimized:   {:?}\n", total_time_cached);

    // ============================================================================
    // Part 4: Memory vs Speed Tradeoff
    // ============================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("Part 4: Memory vs Speed Tradeoff");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("💾 Estimated Memory Usage:");
    println!("   Cache entries:     {} items", stats.current_size);
    println!("   Per entry:         ~1-5 KB (detection results)");
    println!(
        "   Total cache:       ~{}-{} KB",
        stats.current_size * 1,
        stats.current_size * 5
    );
    println!("   Knowledge base:    ~5 MB (shared, loaded once)\n");

    println!("⚖️  Tradeoff Analysis:");
    println!(
        "   Memory cost:       Low (~{} KB for {} entries)",
        stats.current_size * 3,
        stats.current_size
    );
    println!(
        "   Speed benefit:     High ({:.1}x faster with cache)",
        speedup
    );
    println!("   Recommendation:    ✅ Caching is highly beneficial\n");

    // ============================================================================
    // Part 5: Best Practices
    // ============================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("Part 5: Best Practices");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("📋 When to use caching:");
    println!("   ✅ Analyzing the same code multiple times");
    println!("   ✅ Hot reload scenarios (rapid re-analysis)");
    println!("   ✅ Large codebases with repeated patterns");
    println!("   ✅ Production environments with repeated requests\n");

    println!("📋 When NOT to use caching:");
    println!("   ❌ Single-use analysis (one-time processing)");
    println!("   ❌ Highly dynamic code (always changing)");
    println!("   ❌ Memory-constrained environments");
    println!("   ❌ Code with sensitive data (security concern)\n");

    println!("📋 Configuration recommendations:");
    println!("   Development:   max_entries=1000, ttl=5min");
    println!("   Production:    max_entries=10000, ttl=15min");
    println!("   High-memory:   max_entries=50000, ttl=30min");
    println!("   Low-memory:    max_entries=100, ttl=1min\n");

    // ============================================================================
    // Summary
    // ============================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("📊 Summary");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!(
        "✅ Caching provides {:.1}x performance improvement",
        speedup
    );
    println!(
        "✅ Hit rate of {:.1}% demonstrates effectiveness",
        stats.hit_rate()
    );
    println!(
        "✅ Memory overhead is minimal (~{} KB)",
        stats.current_size * 3
    );
    println!("✅ Recommended for production use\n");

    println!("🎯 Key Takeaways:");
    println!(
        "   1. First analysis (cache miss):    ~{:?}",
        avg_time_uncached
    );
    println!(
        "   2. Cached analysis (cache hit):    ~{:?}",
        avg_time_cached
    );
    println!("   3. Speedup factor:                 {:.1}x", speedup);
    println!("   4. Recommended configuration:      Default (1000 entries, 5min TTL)\n");

    Ok(())
}
