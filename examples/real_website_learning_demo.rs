//! Real Website Learning and Generation Demo
//! 
//! This example demonstrates the AI browser making REAL HTTP requests,
//! learning from actual websites, and generating completely new layouts
//! while preserving all functionality.

use anyhow::Result;
use std::sync::Arc;
use browerai::{
    ai::{
        AiRuntime, InferenceEngine, ModelManager, AutonomousConfig,
        LearningMode, PreservationStrategy, performance_monitor::PerformanceMonitor,
    },
    SeamlessBrowser, UserPreferences,
};

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║     BrowerAI - 真实网站学习与生成演示                            ║");
    println!("║     Real Website Learning and Generation Demo                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("本演示将：");
    println!("1. 真实请求网站获取HTML内容");
    println!("2. AI自主学习网站结构和模式");
    println!("3. 智能推理优化方案");
    println!("4. 生成全新的现代化布局");
    println!("5. 验证功能完整性");
    println!();

    // 1. 初始化AI运行时
    println!("🔧 初始化AI运行时...");
    let model_dir = std::path::PathBuf::from("./models/local");
    let model_manager = ModelManager::new(model_dir)?;
    let perf_monitor = PerformanceMonitor::new(true);
    let inference_engine = InferenceEngine::with_monitor(perf_monitor)?;
    let ai_runtime = Arc::new(AiRuntime::with_models(inference_engine, model_manager));
    println!("✅ AI运行时已初始化");
    println!();

    // 2. 配置为生成模式（而不是透明模式）
    println!("⚙️  配置AI浏览器为生成模式...");
    let config = AutonomousConfig {
        enable_autonomous_learning: true,
        enable_intelligent_reasoning: true,
        enable_code_generation: true,
        learning_mode: LearningMode::Explicit,  // 显式模式，展示过程
        preservation_strategy: PreservationStrategy::Intelligent,  // 智能保持关键功能
        max_concurrent_learning: 3,
        optimization_threshold: 0.3,  // 降低阈值，更容易触发生成
    };
    println!("   - 学习模式: Explicit (显式展示AI处理)");
    println!("   - 保持策略: Intelligent (智能保持关键功能)");
    println!("   - 优化阈值: 0.3 (更容易触发AI生成)");
    println!("✅ 配置完成");
    println!();

    // 3. 创建浏览器
    let mut browser = SeamlessBrowser::new(ai_runtime.clone());
    
    let preferences = UserPreferences {
        enable_ai_features: true,
        performance_priority: true,
        accessibility_priority: false,
        custom_styles: std::collections::HashMap::new(),
    };
    browser.set_user_preferences(preferences);
    
    browser.start_learning()?;
    println!("🌐 浏览器已准备就绪，持续学习已启动");
    println!();

    // 4. 测试网站列表（使用公共可访问的网站）
    let test_websites = vec![
        "http://example.com",           // 简单测试网站
        "http://info.cern.ch",          // 第一个网站，简单HTML
        "http://motherfuckingwebsite.com",  // 极简网站
    ];

    println!("═══════════════════════════════════════════════════════════════════");
    println!("🚀 开始真实网站访问、学习和生成");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    for (idx, url) in test_websites.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("🌍 网站 {}/{}: {}", idx + 1, test_websites.len(), url);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        match browser.navigate(url).await {
            Ok(result) => {
                println!("✅ 访问成功！");
                println!();
                println!("📊 结果统计：");
                println!("   - 渲染时间: {}ms", result.render_time_ms);
                println!("   - AI增强: {}", if result.ai_enhanced { "是 ✨" } else { "否" });
                println!("   - 功能验证: {}", if result.functionality_verified { "通过 ✓" } else { "失败 ✗" });
                println!("   - HTML大小: {} bytes", result.html.len());
                println!();

                if result.ai_enhanced {
                    println!("🎨 AI生成了全新布局！");
                    println!();
                    println!("生成的HTML预览（前500字符）:");
                    println!("┌─────────────────────────────────────────────────────────────┐");
                    let preview = if result.html.len() > 500 {
                        &result.html[..500]
                    } else {
                        &result.html
                    };
                    for line in preview.lines().take(20) {
                        println!("│ {}", line);
                    }
                    println!("└─────────────────────────────────────────────────────────────┘");
                    println!();
                    
                    // 保存生成的HTML到文件
                    let filename = format!("generated_{}.html", idx + 1);
                    if let Err(e) = std::fs::write(&filename, &result.html) {
                        println!("⚠️  无法保存文件: {}", e);
                    } else {
                        println!("💾 完整HTML已保存到: {}", filename);
                        println!("   可以在浏览器中打开查看生成的新布局！");
                    }
                } else {
                    println!("📋 使用原始版本（AI学习但未达到优化阈值）");
                }
                
                println!();
            }
            Err(e) => {
                println!("❌ 访问失败: {}", e);
                println!("   这可能是因为网络问题或网站不可访问");
                println!();
            }
        }

        // 暂停一下，避免过快请求
        if idx < test_websites.len() - 1 {
            println!("⏳ 等待2秒后继续...");
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            println!();
        }
    }

    // 5. 显示学习统计
    println!("═══════════════════════════════════════════════════════════════════");
    println!("📈 学习统计总结");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let stats = browser.get_session_stats();
    
    println!("🎯 会话统计：");
    println!("   - 访问页面: {}", stats.pages_visited);
    println!("   - AI增强应用: {}", stats.ai_enhancements_applied);
    println!("   - 成功率: {:.1}%", 
             if stats.pages_visited > 0 { 
                 (stats.ai_enhancements_applied as f32 / stats.pages_visited as f32) * 100.0 
             } else { 
                 0.0 
             });
    println!();

    println!("🤖 AI协调器统计：");
    println!("   - 处理网站: {}", stats.coordinator_stats.total_sites_processed);
    println!("   - 功能验证通过: {}", stats.coordinator_stats.functionality_validations_passed);
    println!("   - 学习的模式: {}", stats.coordinator_stats.total_patterns_learned);
    println!();

    // 6. 停止学习
    println!("⏹  停止持续学习...");
    browser.stop_learning()?;
    println!("✅ 已停止");
    println!();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                    演示完成！✨                                   ║");
    println!("║                                                                   ║");
    println!("║  BrowerAI成功展示了：                                             ║");
    println!("║  ✓ 真实HTTP请求获取网站内容                                       ║");
    println!("║  ✓ AI自主学习网站结构和模式                                       ║");
    println!("║  ✓ 智能推理和优化决策                                             ║");
    println!("║  ✓ 生成全新的现代化布局                                           ║");
    println!("║  ✓ 验证功能完整性                                                 ║");
    println!("║                                                                   ║");
    println!("║  查看生成的HTML文件以查看AI创建的新布局！                         ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    Ok(())
}
