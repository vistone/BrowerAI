//! 完整的网站学习和个性化渲染集成管道
//!
//! 这个模块提供一个完整的、端到端的管道，将所有BrowerAI的功能集成在一起：
//! 1. 获取网页
//! 2. 学习网站技术
//! 3. 分析JavaScript
//! 4. 生成个性化布局
//! 5. 输出结果

#![allow(dead_code)]
#![allow(clippy::all)]

pub mod analysis;
pub mod learning;
pub mod network;
pub mod output;
pub mod pipeline;
pub mod rendering;

pub use output::{OutputFormat, OutputGenerator};
pub use pipeline::{IntegratedPipeline, PipelineConfig, PipelineResult};
