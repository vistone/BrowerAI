//! Learning Engine - 学习系统
//!
//! 实现持续学习机制，包括：
//! - 从真实网站学习
//! - 反馈学习
//! - 模型微调
//! - 知识库更新

use browerai_core::{BrowserError, Result};
use std::collections::HashMap;
use std::time::SystemTime;

/// 学习引擎
#[derive(Debug, Clone)]
pub struct LearningEngine {
    /// 配置
    #[allow(dead_code)]
    config: LearningConfig,
    /// 是否启用
    enabled: bool,
    /// 知识库
    knowledge_base: KnowledgeBase,
    /// 学习统计
    stats: LearningStats,
}

impl LearningEngine {
    /// 创建新的学习引擎
    pub fn new() -> Self {
        Self {
            config: LearningConfig::default(),
            enabled: true,
            knowledge_base: KnowledgeBase::new(),
            stats: LearningStats::default(),
        }
    }

    /// 使用配置创建学习引擎
    pub fn with_config(config: LearningConfig) -> Self {
        Self {
            enabled: config.enabled,
            knowledge_base: KnowledgeBase::new(),
            stats: LearningStats::default(),
            config,
        }
    }

    /// 学习样本
    pub fn learn(&mut self, sample: TrainingSample) -> Result<LearningResult> {
        if !self.enabled {
            return Ok(LearningResult::skipped());
        }

        // 验证样本
        self.validate_sample(&sample)?;

        // 添加到知识库
        self.knowledge_base.add_sample(sample.clone());

        // 更新统计
        self.stats.samples_learned += 1;
        self.stats.last_update = Some(SystemTime::now());

        Ok(LearningResult {
            success: true,
            confidence_delta: 0.01, // 简化实现
            message: "Sample learned successfully".to_string(),
        })
    }

    /// 批量学习
    pub fn learn_batch(&mut self, samples: Vec<TrainingSample>) -> Result<BatchLearningResult> {
        let mut results = Vec::new();
        let mut success_count = 0;

        for sample in samples {
            match self.learn(sample) {
                Ok(result) => {
                    if result.success {
                        success_count += 1;
                    }
                    results.push(result);
                }
                Err(e) => {
                    results.push(LearningResult {
                        success: false,
                        confidence_delta: 0.0,
                        message: e.to_string(),
                    });
                }
            }
        }

        Ok(BatchLearningResult {
            total: results.len(),
            successful: success_count,
            failed: results.len() - success_count,
            results,
        })
    }

    /// 提供反馈
    pub fn feedback(&mut self, prediction_id: &str, correct: bool, correction: Option<String>) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // 记录反馈
        self.knowledge_base.add_feedback(FeedbackRecord {
            prediction_id: prediction_id.to_string(),
            correct,
            correction,
            timestamp: SystemTime::now(),
        });

        self.stats.feedback_received += 1;

        Ok(())
    }

    /// 验证样本
    fn validate_sample(&self, sample: &TrainingSample) -> Result<()> {
        if sample.input.is_empty() {
            return Err(BrowserError::ai("Sample input is empty"));
        }

        if sample.expected_output.is_empty() {
            return Err(BrowserError::ai("Sample expected output is empty"));
        }

        Ok(())
    }

    /// 获取知识库
    pub fn knowledge_base(&self) -> &KnowledgeBase {
        &self.knowledge_base
    }

    /// 获取统计信息
    pub fn stats(&self) -> &LearningStats {
        &self.stats
    }

    /// 检查是否启用
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// 启用/禁用学习
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// 导出知识
    pub fn export_knowledge(&self) -> Result<String> {
        // 简化实现：返回JSON格式的知识
        Ok(format!(
            "{{\"samples\": {}, \"feedback\": {}}}",
            self.stats.samples_learned,
            self.stats.feedback_received
        ))
    }

    /// 重置学习状态
    pub fn reset(&mut self) {
        self.knowledge_base = KnowledgeBase::new();
        self.stats = LearningStats::default();
    }
}

impl Default for LearningEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// 训练样本
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// 样本ID
    pub id: String,
    /// 输入
    pub input: Vec<f32>,
    /// 期望输出
    pub expected_output: Vec<f32>,
    /// 样本类型
    pub sample_type: SampleType,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 创建时间
    pub created_at: SystemTime,
}

impl TrainingSample {
    /// 创建新的训练样本
    pub fn new(id: impl Into<String>, input: Vec<f32>, expected_output: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            input,
            expected_output,
            sample_type: SampleType::Generic,
            metadata: HashMap::new(),
            created_at: SystemTime::now(),
        }
    }

    /// 设置样本类型
    pub fn with_type(mut self, sample_type: SampleType) -> Self {
        self.sample_type = sample_type;
        self
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// 样本类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleType {
    /// 通用
    Generic,
    /// HTML结构
    HtmlStructure,
    /// CSS样式
    CssStyle,
    /// JavaScript代码
    JavaScript,
    /// 渲染输出
    RenderOutput,
    /// 用户反馈
    UserFeedback,
}

/// 学习结果
#[derive(Debug, Clone)]
pub struct LearningResult {
    /// 是否成功
    pub success: bool,
    /// 置信度变化
    pub confidence_delta: f32,
    /// 消息
    pub message: String,
}

impl LearningResult {
    /// 创建跳过的结果
    fn skipped() -> Self {
        Self {
            success: false,
            confidence_delta: 0.0,
            message: "Learning is disabled".to_string(),
        }
    }
}

/// 批量学习结果
#[derive(Debug, Clone)]
pub struct BatchLearningResult {
    /// 总数
    pub total: usize,
    /// 成功数
    pub successful: usize,
    /// 失败数
    pub failed: usize,
    /// 详细结果
    pub results: Vec<LearningResult>,
}

/// 知识库
#[derive(Debug, Clone)]
pub struct KnowledgeBase {
    /// 样本存储
    samples: Vec<TrainingSample>,
    /// 反馈记录
    feedback: Vec<FeedbackRecord>,
    /// 最大容量
    max_capacity: usize,
}

impl KnowledgeBase {
    /// 创建新的知识库
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            feedback: Vec::new(),
            max_capacity: 10000,
        }
    }

    /// 添加样本
    fn add_sample(&mut self, sample: TrainingSample) {
        if self.samples.len() >= self.max_capacity {
            // 移除最旧的样本
            self.samples.remove(0);
        }
        self.samples.push(sample);
    }

    /// 添加反馈
    fn add_feedback(&mut self, record: FeedbackRecord) {
        if self.feedback.len() >= self.max_capacity {
            self.feedback.remove(0);
        }
        self.feedback.push(record);
    }

    /// 获取样本数量
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// 获取反馈数量
    pub fn feedback_count(&self) -> usize {
        self.feedback.len()
    }

    /// 按类型获取样本
    pub fn get_samples_by_type(&self, sample_type: SampleType) -> Vec<&TrainingSample> {
        self.samples.iter()
            .filter(|s| s.sample_type == sample_type)
            .collect()
    }
}

/// 反馈记录
#[derive(Debug, Clone)]
pub struct FeedbackRecord {
    /// 预测ID
    pub prediction_id: String,
    /// 是否正确
    pub correct: bool,
    /// 修正
    pub correction: Option<String>,
    /// 时间戳
    pub timestamp: SystemTime,
}

/// 学习配置
#[derive(Debug, Clone)]
pub struct LearningConfig {
    /// 是否启用学习
    pub enabled: bool,
    /// 学习率
    pub learning_rate: f32,
    /// 批量大小
    pub batch_size: usize,
    /// 最大知识库大小
    pub max_knowledge_size: usize,
    /// 自动保存间隔（秒）
    pub auto_save_interval_secs: u64,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            learning_rate: 0.01,
            batch_size: 32,
            max_knowledge_size: 10000,
            auto_save_interval_secs: 3600,
        }
    }
}

/// 学习统计
#[derive(Debug, Clone, Default)]
pub struct LearningStats {
    /// 学习的样本数
    pub samples_learned: usize,
    /// 接收的反馈数
    pub feedback_received: usize,
    /// 模型更新次数
    pub model_updates: usize,
    /// 最后更新时间
    pub last_update: Option<SystemTime>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_engine_creation() {
        let engine = LearningEngine::new();
        assert!(engine.is_enabled());
    }

    #[test]
    fn test_training_sample() {
        let sample = TrainingSample::new("test-1", vec![1.0, 2.0], vec![0.5])
            .with_type(SampleType::HtmlStructure)
            .with_metadata("url", "https://example.com");
        
        assert_eq!(sample.id, "test-1");
        assert_eq!(sample.sample_type, SampleType::HtmlStructure);
        assert!(sample.metadata.contains_key("url"));
    }

    #[test]
    fn test_learn_sample() {
        let mut engine = LearningEngine::new();
        let sample = TrainingSample::new("test-1", vec![1.0, 2.0], vec![0.5]);
        
        let result = engine.learn(sample);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.success);
        
        assert_eq!(engine.stats().samples_learned, 1);
    }

    #[test]
    fn test_feedback() {
        let mut engine = LearningEngine::new();
        
        let result = engine.feedback("pred-1", false, Some("corrected".to_string()));
        assert!(result.is_ok());
        
        assert_eq!(engine.stats().feedback_received, 1);
    }

    #[test]
    fn test_batch_learn() {
        let mut engine = LearningEngine::new();
        let samples = vec![
            TrainingSample::new("test-1", vec![1.0], vec![0.1]),
            TrainingSample::new("test-2", vec![2.0], vec![0.2]),
            TrainingSample::new("test-3", vec![3.0], vec![0.3]),
        ];
        
        let result = engine.learn_batch(samples);
        assert!(result.is_ok());
        
        let batch_result = result.unwrap();
        assert_eq!(batch_result.total, 3);
        assert_eq!(batch_result.successful, 3);
    }

    #[test]
    fn test_disabled_learning() {
        let mut engine = LearningEngine::with_config(LearningConfig {
            enabled: false,
            ..Default::default()
        });
        
        let sample = TrainingSample::new("test-1", vec![1.0], vec![0.5]);
        let result = engine.learn(sample).unwrap();
        
        assert!(!result.success);
    }

    #[test]
    fn test_knowledge_base() {
        let mut kb = KnowledgeBase::new();
        
        let sample = TrainingSample::new("test-1", vec![1.0], vec![0.5]);
        kb.add_sample(sample);
        
        assert_eq!(kb.sample_count(), 1);
    }
}
