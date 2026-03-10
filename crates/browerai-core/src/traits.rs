//! 核心 Traits 定义
//!
//! 定义 BrowerAI 的核心抽象接口，包括：
//! - Parser: 解析器接口
//! - Renderer: 渲染器接口
//! - AiModel: AI 模型接口
//! - Deobfuscator: 反混淆器接口
//! - Learner: 学习器接口

use crate::error::Result;
use std::collections::HashMap;

/// 解析器 Trait
///
/// 所有解析器（HTML/CSS/JS）都实现此 trait
/// 设计原则：解析器应该是纯函数，无副作用
///
/// # 示例
/// ```
/// use browerai_core::traits::Parser;
/// use browerai_core::BrowserError;
///
/// struct HtmlDocument;
/// struct HtmlParser;
/// 
/// impl Parser for HtmlParser {
///     type Input = str;
///     type Output = HtmlDocument;
///     
///     fn parse(&self, _input: &Self::Input) -> Result<Self::Output, BrowserError> {
///         Ok(HtmlDocument)
///     }
/// }
/// ```
pub trait Parser {
    /// 输入类型
    type Input: ?Sized;
    /// 输出类型
    type Output;

    /// 解析输入
    ///
    /// # 参数
    /// - `input`: 要解析的输入
    ///
    /// # 返回
    /// - `Ok(Output)`: 解析成功
    /// - `Err(BrowserError)`: 解析失败
    fn parse(&self, input: &Self::Input) -> Result<Self::Output>;

    /// 检查输入是否有效（不实际解析）
    ///
    /// 默认实现调用 parse 并丢弃结果
    fn is_valid(&self, input: &Self::Input) -> bool {
        self.parse(input).is_ok()
    }
}

/// 渲染器 Trait
///
/// 所有渲染引擎都实现此 trait
/// 支持传统渲染和 AI 增强渲染
///
/// # 设计原则
/// - 渲染器应该是状态less的
/// - 支持渐进式渲染
/// - 支持取消操作
pub trait Renderer {
    /// 渲染输入
    ///
    /// # 参数
    /// - `input`: 要渲染的内容
    /// - `viewport`: 视口信息
    ///
    /// # 返回
    /// - `Ok(RenderOutput)`: 渲染成功
    fn render(&self, input: &Self::Input, viewport: &Viewport) -> Result<RenderOutput>
    where
        Self: Sized;

    /// 输入类型
    type Input: ?Sized;

    /// 检查是否支持特定输入
    fn supports(&self, input: &Self::Input) -> bool;

    /// 获取渲染器能力
    fn capabilities(&self) -> RenderCapabilities;
}

/// 视口信息
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    /// 宽度（像素）
    pub width: u32,
    /// 高度（像素）
    pub height: u32,
    /// 设备像素比
    pub device_pixel_ratio: f32,
}

impl Viewport {
    /// 创建新的视口
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            device_pixel_ratio: 1.0,
        }
    }

    /// 设置设备像素比
    pub fn with_dpr(mut self, dpr: f32) -> Self {
        self.device_pixel_ratio = dpr;
        self
    }

    /// 获取物理宽度
    pub fn physical_width(&self) -> u32 {
        (self.width as f32 * self.device_pixel_ratio) as u32
    }

    /// 获取物理高度
    pub fn physical_height(&self) -> u32 {
        (self.height as f32 * self.device_pixel_ratio) as u32
    }
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            device_pixel_ratio: 1.0,
        }
    }
}

/// 渲染输出
#[derive(Debug, Clone)]
pub struct RenderOutput {
    /// 渲染结果（可以是图像、文本等）
    pub data: Vec<u8>,
    /// 元数据
    pub metadata: RenderMetadata,
}

/// 渲染元数据
#[derive(Debug, Clone, Default)]
pub struct RenderMetadata {
    /// 渲染时间（毫秒）
    pub render_time_ms: u64,
    /// 使用的渲染模式
    pub render_mode: RenderMode,
    /// 是否使用了 AI 增强
    pub ai_enhanced: bool,
    /// 额外信息
    pub extra: HashMap<String, String>,
}

/// 渲染模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderMode {
    /// 传统渲染
    Traditional,
    /// AI 增强渲染
    AiEnhanced,
    /// 混合渲染
    Hybrid,
}

impl Default for RenderMode {
    fn default() -> Self {
        RenderMode::Traditional
    }
}

/// 渲染器能力
#[derive(Debug, Clone, Default)]
pub struct RenderCapabilities {
    /// 是否支持 AI 增强
    pub supports_ai: bool,
    /// 是否支持 GPU 加速
    pub supports_gpu: bool,
    /// 最大支持的视口尺寸
    pub max_viewport_size: (u32, u32),
    /// 支持的输出格式
    pub supported_formats: Vec<String>,
}

/// AI 模型 Trait
///
/// 所有 AI 模型（ONNX、自定义模型）都实现此 trait
/// 设计原则：
/// - 模型应该是可热重载的
/// - 支持异步推理
/// - 提供回退机制
///
/// # 示例
/// ```
/// use browerai_core::traits::{AiModel, ModelHealth, ModelMetadata};
/// use browerai_core::BrowserError;
///
/// struct OnnxModel;
///
/// impl AiModel for OnnxModel {
///     type Input = Vec<f32>;
///     type Output = Vec<f32>;
///     
///     fn infer(&self, _input: &Self::Input) -> Result<Self::Output, BrowserError> {
///         Ok(vec![1.0])
///     }
///     
///     fn health_check(&self) -> ModelHealth {
///         ModelHealth::Healthy
///     }
///     
///     fn metadata(&self) -> ModelMetadata {
///         ModelMetadata::default()
///     }
/// }
/// ```
pub trait AiModel: Send + Sync {
    /// 输入类型
    type Input;
    /// 输出类型
    type Output;

    /// 执行推理
    ///
    /// # 参数
    /// - `input`: 输入数据
    ///
    /// # 返回
    /// - `Ok(Output)`: 推理成功
    fn infer(&self, input: &Self::Input) -> Result<Self::Output>;

    /// 批量推理（默认实现逐个处理）
    fn batch_infer(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>>
    where
        Self::Input: Clone,
        Self::Output: Clone,
    {
        inputs.iter().map(|input| self.infer(input)).collect()
    }

    /// 健康检查
    fn health_check(&self) -> ModelHealth;

    /// 获取模型元数据
    fn metadata(&self) -> ModelMetadata;

    /// 是否可用
    fn is_available(&self) -> bool {
        matches!(self.health_check(), ModelHealth::Healthy)
    }
}

/// 模型健康状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelHealth {
    /// 健康
    Healthy,
    /// 降级（可用但性能下降）
    Degraded,
    /// 不可用
    Unhealthy,
}

/// 模型元数据
#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    /// 模型名称
    pub name: String,
    /// 模型版本
    pub version: String,
    /// 输入维度
    pub input_shape: Vec<usize>,
    /// 输出维度
    pub output_shape: Vec<usize>,
    /// 模型大小（字节）
    pub model_size: usize,
    /// 最后更新时间
    pub last_updated: Option<chrono::DateTime<chrono::Utc>>,
}

/// 反混淆器 Trait
///
/// 所有反混淆策略都实现此 trait
/// 设计原则：
/// - 策略应该是可组合的
/// - 支持迭代优化
/// - 提供转换统计
pub trait Deobfuscator: Send + Sync {
    /// 反混淆代码
    ///
    /// # 参数
    /// - `code`: 混淆的代码
    ///
    /// # 返回
    /// - `Ok(DeobfuscationResult)`: 反混淆成功
    fn deobfuscate(&self, code: &str) -> Result<DeobfuscationResult>;

    /// 检查是否能处理此代码
    fn can_handle(&self, code: &str) -> bool;

    /// 获取反混淆器名称
    fn name(&self) -> &str;

    /// 获取反混淆器描述
    fn description(&self) -> &str;
}

/// 反混淆结果
#[derive(Debug, Clone, Default)]
pub struct DeobfuscationResult {
    /// 反混淆后的代码
    pub code: String,
    /// 应用的转换
    pub transformations: Vec<String>,
    /// 统计信息
    pub stats: DeobfuscationStats,
}

/// 反混淆统计
#[derive(Debug, Clone, Default)]
pub struct DeobfuscationStats {
    /// 字符串数组展开数
    pub string_arrays_unpacked: usize,
    /// 代理函数移除数
    pub proxy_functions_removed: usize,
    /// 控制流简化数
    pub control_flow_simplified: usize,
    /// 常量折叠数
    pub constants_folded: usize,
    /// 代码大小变化（字节）
    pub size_change: i64,
}

/// 学习器 Trait
///
/// 所有学习系统（真实网站学习、反馈学习）都实现此 trait
/// 设计原则：
/// - 支持异步学习
/// - 可评估学习质量
/// - 支持增量学习
pub trait Learner: Send + Sync {
    /// 学习任务类型
    type Task;
    /// 学习结果类型
    type Result;

    /// 执行学习
    ///
    /// # 参数
    /// - `task`: 学习任务
    ///
    /// # 返回
    /// - `Ok(Self::Result)`: 学习成功
    fn learn(&self, task: &Self::Task) -> Result<Self::Result>;

    /// 评估学习质量
    fn evaluate(&self, result: &Self::Result) -> LearningQuality;

    /// 是否支持增量学习
    fn supports_incremental(&self) -> bool;

    /// 增量学习（如果支持）
    fn learn_incremental(&self, _previous: &Self::Result, _new_task: &Self::Task) -> Result<Self::Result> {
        Err(crate::error::BrowserError::learning("Incremental learning not supported"))
    }
}

/// 学习质量
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearningQuality {
    /// 总体评分（0.0 - 1.0）
    pub overall_score: f32,
    /// 覆盖率
    pub coverage: f32,
    /// 准确率
    pub accuracy: f32,
    /// 置信度
    pub confidence: f32,
}

impl LearningQuality {
    /// 创建新的质量评估
    pub fn new(score: f32) -> Self {
        Self {
            overall_score: score.clamp(0.0, 1.0),
            coverage: 0.0,
            accuracy: 0.0,
            confidence: 0.0,
        }
    }

    /// 设置覆盖率
    pub fn with_coverage(mut self, coverage: f32) -> Self {
        self.coverage = coverage.clamp(0.0, 1.0);
        self
    }

    /// 设置准确率
    pub fn with_accuracy(mut self, accuracy: f32) -> Self {
        self.accuracy = accuracy.clamp(0.0, 1.0);
        self
    }

    /// 是否合格（>0.7）
    pub fn is_acceptable(&self) -> bool {
        self.overall_score >= 0.7
    }

    /// 是否优秀（>0.9）
    pub fn is_excellent(&self) -> bool {
        self.overall_score >= 0.9
    }
}

impl Default for LearningQuality {
    fn default() -> Self {
        Self {
            overall_score: 0.0,
            coverage: 0.0,
            accuracy: 0.0,
            confidence: 0.0,
        }
    }
}

/// 分析器 Trait
///
/// 所有代码分析器都实现此 trait
/// 设计原则：
/// - 分析器应该是纯函数，无副作用
/// - 支持增量分析
/// - 提供详细的分析结果
pub trait Analyzer: Send + Sync {
    /// 输入类型
    type Input: ?Sized;
    /// 输出类型
    type Output;

    /// 分析输入
    ///
    /// # 参数
    /// - `input`: 要分析的输入
    ///
    /// # 返回
    /// - `Ok(Output)`: 分析成功
    /// - `Err(BrowserError)`: 分析失败
    fn analyze(&self, input: &Self::Input) -> Result<Self::Output>;

    /// 检查是否支持特定输入
    fn supports(&self, _input: &Self::Input) -> bool {
        true
    }
}

/// 缓存 Trait
///
/// 所有缓存实现（内存、Redis、磁盘）都实现此 trait
pub trait Cache: Send + Sync {
    /// 获取值
    fn get(&self, key: &str) -> Option<Vec<u8>>;

    /// 设置值
    fn set(&self, key: &str, value: Vec<u8>);

    /// 删除值
    fn delete(&self, key: &str);

    /// 清空缓存
    fn clear(&self);

    /// 获取统计信息
    fn stats(&self) -> CacheStats;
}

/// 缓存统计
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// 命中次数
    pub hits: u64,
    /// 未命中次数
    pub misses: u64,
    /// 缓存大小
    pub size: usize,
    /// 条目数
    pub entries: usize,
}

impl CacheStats {
    /// 命中率
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewport() {
        let vp = Viewport::new(1920, 1080).with_dpr(2.0);
        assert_eq!(vp.physical_width(), 3840);
        assert_eq!(vp.physical_height(), 2160);
    }

    #[test]
    fn test_learning_quality() {
        let quality = LearningQuality::new(0.85)
            .with_coverage(0.9)
            .with_accuracy(0.8);
        
        assert!(quality.is_acceptable());
        assert!(!quality.is_excellent());
    }

    #[test]
    fn test_cache_stats() {
        let stats = CacheStats {
            hits: 90,
            misses: 10,
            size: 1024,
            entries: 5,
        };
        
        assert_eq!(stats.hit_rate(), 0.9);
    }
}
