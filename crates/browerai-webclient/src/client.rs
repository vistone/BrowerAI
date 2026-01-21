//! WebClient - 主客户端，整合所有功能

use crate::commands::{Command, ProcessResult};
use crate::config::ClientConfig;
use crate::session::UserSession;
use anyhow::Result;
use browerai_integrated_pipeline::{
    IntegratedPipeline, OutputFormat, OutputGenerator, PipelineConfig,
};
use std::time::Instant;

pub struct WebClient {
    config: ClientConfig,
    pipeline: IntegratedPipeline,
    session: UserSession,
}

impl WebClient {
    pub fn new(config: ClientConfig) -> Result<Self> {
        let user_id = config.user_id.clone();
        let mut pipeline_config = PipelineConfig::default();
        pipeline_config.user_id = config.user_id.clone();
        pipeline_config.enable_cache = config.enable_cache;
        pipeline_config.output_dir = config.output_dir.clone();
        pipeline_config.cache_dir = std::path::PathBuf::from("./cache");

        let pipeline = IntegratedPipeline::new(pipeline_config);

        let session = UserSession::new(user_id);

        Ok(Self {
            config,
            pipeline,
            session,
        })
    }

    /// 执行命令
    pub async fn execute(&mut self, command: Command) -> Result<ProcessResult> {
        match command {
            Command::Process { url, user_id } => {
                let user_id = user_id.unwrap_or_else(|| self.config.user_id.clone());
                self.process_url(&url, &user_id).await
            }
            Command::History { user_id: _ } => {
                self.show_history();
                Ok(ProcessResult::success(
                    "history".to_string(),
                    self.session.user_id.clone(),
                    "history".to_string(),
                    0,
                ))
            }
            Command::Config { key, value } => {
                self.update_config(&key, &value);
                Ok(ProcessResult::success(
                    "config".to_string(),
                    self.session.user_id.clone(),
                    "config updated".to_string(),
                    0,
                ))
            }
            Command::ClearCache => {
                self.clear_cache();
                Ok(ProcessResult::success(
                    "cache".to_string(),
                    self.session.user_id.clone(),
                    "cache cleared".to_string(),
                    0,
                ))
            }
            Command::Stats { user_id: _ } => {
                self.show_stats();
                Ok(ProcessResult::success(
                    "stats".to_string(),
                    self.session.user_id.clone(),
                    "stats".to_string(),
                    0,
                ))
            }
        }
    }

    /// 处理URL - 完整流程
    pub async fn process_url(&mut self, url: &str, user_id: &str) -> Result<ProcessResult> {
        let start = Instant::now();

        log::info!("▶ 开始处理: {} (用户: {})", url, user_id);

        match self.pipeline.execute(url).await {
            Ok(result) => {
                let output_dir = self
                    .config
                    .output_dir
                    .join(format!("{}", start.elapsed().as_secs()));
                let generator = OutputGenerator::new(output_dir.clone());

                match generator.generate(&result, OutputFormat::Package) {
                    Ok(_) => {
                        let output_path = output_dir.to_string_lossy().to_string();
                        self.session.add_entry(url.to_string(), output_path.clone());

                        let elapsed = start.elapsed().as_millis() as u64;
                        log::info!("✓ 处理完成，耗时: {}ms", elapsed);

                        Ok(ProcessResult::success(
                            url.to_string(),
                            user_id.to_string(),
                            output_path,
                            elapsed,
                        ))
                    }
                    Err(e) => {
                        let elapsed = start.elapsed().as_millis() as u64;
                        log::error!("✗ 输出生成失败: {}", e);
                        Ok(ProcessResult::error(
                            url.to_string(),
                            user_id.to_string(),
                            format!("输出生成失败: {}", e),
                            elapsed,
                        ))
                    }
                }
            }
            Err(e) => {
                let elapsed = start.elapsed().as_millis() as u64;
                log::error!("✗ 管道执行失败: {}", e);
                Ok(ProcessResult::error(
                    url.to_string(),
                    user_id.to_string(),
                    format!("管道执行失败: {}", e),
                    elapsed,
                ))
            }
        }
    }

    fn show_history(&self) {
        println!("\n📋 会话历史 (用户: {})", self.session.user_id);
        println!("---");
        for (i, entry) in self.session.history.iter().enumerate() {
            println!(
                "{}. [{}] {} → {}",
                i + 1,
                entry.timestamp,
                entry.url,
                entry.result_path
            );
        }
        if self.session.history.is_empty() {
            println!("暂无历史记录");
        }
    }

    fn show_stats(&self) {
        println!("\n📊 统计信息 (用户: {})", self.session.user_id);
        println!("---");
        println!("处理过的网站数: {}", self.session.history.len());
        println!(
            "首选颜色: {}",
            self.session.preferences.preferred_colors.join(", ")
        );
        println!("布局偏好: {}", self.session.preferences.layout_preference);
        println!(
            "字体大小倍数: {}",
            self.session.preferences.font_size_multiplier
        );
    }

    fn update_config(&mut self, key: &str, value: &str) {
        match key {
            "user_id" => {
                self.config.user_id = value.to_string();
                self.session.user_id = value.to_string();
            }
            "layout" => {
                self.session.preferences.layout_preference = value.to_string();
            }
            "font_size" => {
                if let Ok(size) = value.parse::<f32>() {
                    self.session.preferences.font_size_multiplier = size;
                }
            }
            _ => log::warn!("未知的配置项: {}", key),
        }
        log::info!("✓ 配置已更新: {} = {}", key, value);
    }

    fn clear_cache(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.config.cache_dir) {
            log::warn!("清空缓存失败: {}", e);
        } else {
            log::info!("✓ 缓存已清空");
        }
    }
}
