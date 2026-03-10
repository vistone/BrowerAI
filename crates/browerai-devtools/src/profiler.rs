//! Profiler - 性能分析器
//!
//! 提供性能分析和监控功能：
//! - 时间标记（Timing Marks）
//! - 性能指标收集
//! - 内存使用监控
//! - 火焰图生成

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 性能分析器
#[derive(Debug, Clone)]
pub struct Profiler {
    /// 配置
    config: ProfilerConfig,
    /// 时间标记
    marks: Vec<TimingMark>,
    /// 性能指标
    metrics: HashMap<String, f64>,
    /// 当前激活的测量
    active_measurements: Vec<String>,
    /// 样本计数
    sample_count: usize,
}

impl Profiler {
    /// 创建新的性能分析器
    pub fn new() -> Self {
        Self {
            config: ProfilerConfig::default(),
            marks: Vec::new(),
            metrics: HashMap::new(),
            active_measurements: Vec::new(),
            sample_count: 0,
        }
    }

    /// 使用配置创建分析器
    pub fn with_config(config: ProfilerConfig) -> Self {
        Self {
            config,
            marks: Vec::new(),
            metrics: HashMap::new(),
            active_measurements: Vec::new(),
            sample_count: 0,
        }
    }

    /// 开始测量
    pub fn start(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.active_measurements.push(name.clone());
        
        self.marks.push(TimingMark {
            name: name.clone(),
            start_time: Instant::now(),
            end_time: None,
            duration: None,
        });
    }

    /// 结束测量
    pub fn end(&mut self, name: &str) -> Option<Duration> {
        // 查找并更新标记
        if let Some(mark) = self.marks.iter_mut().rev().find(|m| m.name == name && m.end_time.is_none()) {
            let end_time = Instant::now();
            mark.end_time = Some(end_time);
            mark.duration = Some(end_time.duration_since(mark.start_time));
            
            self.active_measurements.retain(|n| n != name);
            self.sample_count += 1;
            
            return mark.duration;
        }
        
        None
    }

    /// 测量函数执行时间
    pub fn measure<T>(&mut self, name: impl Into<String>, f: impl FnOnce() -> T) -> T {
        let name = name.into();
        self.start(&name);
        let result = f();
        self.end(&name);
        result
    }

    /// 记录指标
    pub fn record_metric(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.insert(name.into(), value);
    }

    /// 获取指标
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }

    /// 获取所有标记
    pub fn marks(&self) -> &[TimingMark] {
        &self.marks
    }

    /// 获取特定名称的标记
    pub fn marks_by_name(&self, name: &str) -> Vec<&TimingMark> {
        self.marks.iter()
            .filter(|m| m.name == name)
            .collect()
    }

    /// 获取平均执行时间
    pub fn average_duration(&self, name: &str) -> Option<Duration> {
        let marks: Vec<_> = self.marks_by_name(name);
        if marks.is_empty() {
            return None;
        }
        
        let total: Duration = marks.iter()
            .filter_map(|m| m.duration)
            .sum();
        
        Some(total / marks.len() as u32)
    }

    /// 获取性能摘要
    pub fn summary(&self) -> ProfileSummary {
        ProfileSummary {
            total_marks: self.marks.len(),
            total_metrics: self.metrics.len(),
            sample_count: self.sample_count,
            average_times: self.calculate_average_times(),
        }
    }

    /// 计算所有平均时间
    fn calculate_average_times(&self) -> HashMap<String, f64> {
        let mut times = HashMap::new();
        let mut counts: HashMap<String, usize> = HashMap::new();
        
        for mark in &self.marks {
            if let Some(duration) = mark.duration {
                let millis = duration.as_secs_f64() * 1000.0;
                *times.entry(mark.name.clone()).or_insert(0.0) += millis;
                *counts.entry(mark.name.clone()).or_insert(0) += 1;
            }
        }
        
        for (name, total) in &mut times {
            if let Some(count) = counts.get(name) {
                *total /= *count as f64;
            }
        }
        
        times
    }

    /// 获取样本数量
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// 清空所有数据
    pub fn clear(&mut self) {
        self.marks.clear();
        self.metrics.clear();
        self.active_measurements.clear();
        self.sample_count = 0;
    }

    /// 获取配置
    pub fn config(&self) -> &ProfilerConfig {
        &self.config
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// 时间标记
#[derive(Debug, Clone)]
pub struct TimingMark {
    /// 标记名称
    pub name: String,
    /// 开始时间
    pub start_time: Instant,
    /// 结束时间
    pub end_time: Option<Instant>,
    /// 持续时间
    pub duration: Option<Duration>,
}

/// 性能摘要
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProfileSummary {
    /// 总标记数
    pub total_marks: usize,
    /// 总指标数
    pub total_metrics: usize,
    /// 样本数
    pub sample_count: usize,
    /// 平均时间（毫秒）
    pub average_times: HashMap<String, f64>,
}

/// 性能指标类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// 时间（毫秒）
    Time,
    /// 内存（字节）
    Memory,
    /// 计数
    Count,
    /// 百分比
    Percentage,
}

/// 性能指标定义
#[derive(Debug, Clone)]
pub struct MetricDefinition {
    /// 名称
    pub name: String,
    /// 类型
    pub metric_type: MetricType,
    /// 描述
    pub description: String,
    /// 单位
    pub unit: String,
}

/// 分析器配置
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// 最大标记数
    pub max_marks: usize,
    /// 启用内存分析
    pub enable_memory_profiling: bool,
    /// 采样间隔（毫秒）
    pub sample_interval_ms: u64,
    /// 自动记录FPS
    pub record_fps: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            max_marks: 10000,
            enable_memory_profiling: false,
            sample_interval_ms: 16,
            record_fps: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_creation() {
        let profiler = Profiler::new();
        assert_eq!(profiler.sample_count(), 0);
    }

    #[test]
    fn test_timing_mark() {
        let mut profiler = Profiler::new();
        
        profiler.start("test-operation");
        thread::sleep(Duration::from_millis(10));
        let duration = profiler.end("test-operation");
        
        assert!(duration.is_some());
        assert!(duration.unwrap() >= Duration::from_millis(10));
    }

    #[test]
    fn test_measure_function() {
        let mut profiler = Profiler::new();
        
        let result = profiler.measure("computation", || {
            thread::sleep(Duration::from_millis(5));
            42
        });
        
        assert_eq!(result, 42);
        assert_eq!(profiler.sample_count(), 1);
    }

    #[test]
    fn test_metrics() {
        let mut profiler = Profiler::new();
        
        profiler.record_metric("memory_usage", 1024.0);
        profiler.record_metric("cpu_usage", 50.5);
        
        assert_eq!(profiler.get_metric("memory_usage"), Some(1024.0));
        assert_eq!(profiler.get_metric("cpu_usage"), Some(50.5));
    }

    #[test]
    fn test_average_duration() {
        let mut profiler = Profiler::new();
        
        for _ in 0..3 {
            profiler.start("operation");
            thread::sleep(Duration::from_millis(5));
            profiler.end("operation");
        }
        
        let avg = profiler.average_duration("operation");
        assert!(avg.is_some());
        assert!(avg.unwrap() >= Duration::from_millis(5));
    }

    #[test]
    fn test_summary() {
        let mut profiler = Profiler::new();
        
        profiler.start("op1");
        profiler.end("op1");
        profiler.record_metric("metric1", 100.0);
        
        let summary = profiler.summary();
        assert_eq!(summary.total_marks, 1);
        assert_eq!(summary.total_metrics, 1);
    }

    #[test]
    fn test_clear() {
        let mut profiler = Profiler::new();
        
        profiler.start("test");
        profiler.end("test");
        profiler.record_metric("test", 1.0);
        
        profiler.clear();
        
        assert_eq!(profiler.sample_count(), 0);
        assert!(profiler.get_metric("test").is_none());
    }
}
