//! 示例：自动探索 GitHub

use anyhow::Result;
use auto_observer::{AutoExplorer, ExplorationConfig, ExplorationReporter};

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志
    env_logger::init();

    println!("🚀 启动自动探索系统");
    println!("═══════════════════════════════════════\n");

    // 配置探索参数
    let config = ExplorationConfig {
        max_pages: 10,             // 最多探索10个页面
        max_time_seconds: 120,     // 最多运行2分钟
        max_depth: 2,              // 最大深度2层
        wait_after_action_ms: 500, // 操作后等待500ms
        wait_for_navigation_ms: 5000,
        respect_robots_txt: true,
        allowed_domains: vec!["github.com".to_string()],
        blocked_urls: vec![
            regex::Regex::new(r"logout").unwrap(),
            regex::Regex::new(r"delete").unwrap(),
            regex::Regex::new(r"settings").unwrap(),
        ],
        viewport: auto_observer::ViewportConfig {
            width: 1280,
            height: 720,
            device_scale_factor: 1.0,
        },
        user_agent: Some(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36".to_string(),
        ),
    };

    // 创建探索器
    let mut explorer = AutoExplorer::new(config);

    // 开始探索
    let target_url = "https://github.com";
    println!("🎯 目标网站: {}", target_url);
    println!("⏱️  开始探索...\n");

    match explorer.explore(target_url).await {
        Ok(report) => {
            println!("\n✅ 探索完成!\n");

            // 生成报告
            let reporter = ExplorationReporter::new();

            // 控制台报告
            let text_report = reporter.generate_detailed_report(&report);
            println!("{}", text_report);

            // JSON报告
            let json_report = reporter.generate_json_report(&report)?;
            std::fs::write("exploration_report.json", json_report)?;
            println!("📄 JSON报告已保存: exploration_report.json");

            // HTML报告
            let html_report = reporter.generate_html_report(&report);
            std::fs::write("exploration_report.html", html_report)?;
            println!("🌐 HTML报告已保存: exploration_report.html");

            // 打印关键发现
            println!("\n🔍 关键发现:");
            println!("───────────────────────────────────────");

            if !report.unique_behaviors.is_empty() {
                println!("\n识别的行为模式:");
                for (i, behavior) in report.unique_behaviors.iter().enumerate() {
                    println!(
                        "  {}. {:?} ({}次)",
                        i + 1,
                        behavior.pattern_type,
                        behavior.frequency
                    );
                }
            }

            println!("\n📊 覆盖率: {:.1}%", report.coverage.coverage_percentage);
            println!("📄 探索页面: {}个", report.pages_explored.len());
            println!("👁️  观察记录: {}条", report.total_observations);

            if !report.errors.is_empty() {
                println!("\n⚠️  错误数: {}", report.errors.len());
            }
        }
        Err(e) => {
            eprintln!("❌ 探索失败: {}", e);
            return Err(e);
        }
    }

    Ok(())
}
