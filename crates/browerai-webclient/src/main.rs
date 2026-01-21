//! BrowerAI WebClient - 主程序入口
//! 用法: cargo run --bin browerai-cli -- process <url>

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "BrowerAI")]
#[command(about = "AI驱动的个性化浏览器客户端", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// 用户ID
    #[arg(global = true, short, long, default_value = "default_user")]
    user: String,

    /// 输出目录
    #[arg(global = true, short, long, default_value = "./output")]
    output: PathBuf,

    /// 调试模式
    #[arg(global = true, short, long)]
    debug: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// 处理一个网站URL
    Process {
        /// 网站URL
        url: String,
    },
    /// 显示处理历史
    History,
    /// 配置偏好设置
    Config {
        /// 配置键名
        key: String,
        /// 配置值
        value: String,
    },
    /// 清空缓存
    ClearCache,
    /// 显示用户统计
    Stats,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // 初始化日志
    if cli.debug {
        std::env::set_var("RUST_LOG", "debug");
    } else {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // 创建客户端配置
    let config = browerai_webclient::ClientConfig {
        user_id: cli.user.clone(),
        output_dir: cli.output.clone(),
        cache_dir: PathBuf::from("./cache"),
        enable_cache: true,
        debug_mode: cli.debug,
    };

    let mut client = browerai_webclient::WebClient::new(config)?;

    // 执行命令
    let result = match cli.command {
        Commands::Process { url } => {
            println!("🌐 正在处理: {}", url);
            client
                .execute(browerai_webclient::Command::Process {
                    url,
                    user_id: Some(cli.user),
                })
                .await?
        }
        Commands::History => {
            client
                .execute(browerai_webclient::Command::History {
                    user_id: Some(cli.user),
                })
                .await?
        }
        Commands::Config { key, value } => {
            client
                .execute(browerai_webclient::Command::Config { key, value })
                .await?
        }
        Commands::ClearCache => {
            client
                .execute(browerai_webclient::Command::ClearCache)
                .await?
        }
        Commands::Stats => {
            client
                .execute(browerai_webclient::Command::Stats {
                    user_id: Some(cli.user),
                })
                .await?
        }
    };

    // 打印结果
    if result.success {
        println!("\n✅ 成功: {}", result.message);
        if let Some(path) = result.output_path {
            println!("📁 输出: {}", path);
        }
    } else {
        println!("\n❌ 失败: {}", result.message);
    }
    println!("⏱  耗时: {}ms", result.processing_time_ms);

    Ok(())
}
