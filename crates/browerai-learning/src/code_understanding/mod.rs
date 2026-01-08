//! 代码理解系统 - 架构分析、模块图、数据流
//!
//! 核心功能：
//! 1. **架构检测** - 识别设计模式和代码组织方式
//! 2. **模块分析** - 提取模块边界和依赖关系
//! 3. **数据流追踪** - 分析数据如何在模块间流动
//! 4. **API 识别** - 找出导出和公共接口
//! 5. **文档生成** - 输出结构化分析报告

pub mod api_extractor;
pub mod architecture_detector;
pub mod dataflow_tracker;
pub mod module_analyzer;
pub mod visualization;

pub use api_extractor::ApiExtractor;
pub use architecture_detector::ArchitectureDetector;
pub use dataflow_tracker::DataflowTracker;
pub use module_analyzer::ModuleAnalyzer;
pub use visualization::{GraphVisualization, VisualizationFormat};

/// 代码理解系统 - 统一入口
pub struct CodeUnderstandingSystem {
    arch_detector: ArchitectureDetector,
    module_analyzer: ModuleAnalyzer,
    dataflow_tracker: DataflowTracker,
    api_extractor: ApiExtractor,
}

impl CodeUnderstandingSystem {
    pub fn new() -> Self {
        Self {
            arch_detector: ArchitectureDetector::new(),
            module_analyzer: ModuleAnalyzer::new(),
            dataflow_tracker: DataflowTracker::new(),
            api_extractor: ApiExtractor::new(),
        }
    }

    /// 完整分析流程
    ///
    /// # 输入
    /// - `code`: JavaScript 源代码或混淆代码
    /// - `name`: 项目名称（用于报告）
    ///
    /// # 输出
    /// - `UnderstandingReport`: 完整的架构分析报告
    pub fn analyze(&self, code: &str, name: &str) -> anyhow::Result<UnderstandingReport> {
        log::info!("开始分析代码：{}", name);

        // 1. 检测架构模式
        let architecture = self.arch_detector.detect(code)?;
        log::debug!("检测到架构: {:?}", architecture.pattern);

        // 2. 分析模块结构
        let modules = self.module_analyzer.extract_modules(code)?;
        log::debug!("识别到 {} 个模块", modules.len());

        // 3. 追踪数据流
        let dataflows = self.dataflow_tracker.analyze(code, &modules)?;
        log::debug!("识别到 {} 条数据流", dataflows.len());

        // 4. 提取 API
        let apis = self.api_extractor.extract(code)?;
        log::debug!("识别到 {} 个公共 API", apis.len());

        Ok(UnderstandingReport {
            project_name: name.to_string(),
            architecture,
            modules,
            dataflows,
            apis,
            statistics: ReportStatistics::compute(code),
        })
    }

    /// 生成可视化图表
    pub fn visualize(
        &self,
        report: &UnderstandingReport,
        format: VisualizationFormat,
    ) -> anyhow::Result<String> {
        let mut viz = GraphVisualization::new(&report.project_name);

        // 添加模块节点
        for module in &report.modules {
            viz.add_module_node(module);
        }

        // 添加模块依赖边
        for module in &report.modules {
            for dep in &module.dependencies {
                viz.add_dependency_edge(&module.name, dep);
            }
        }

        // 添加数据流
        for flow in &report.dataflows {
            viz.add_dataflow(&flow.source, &flow.target, &flow.description);
        }

        viz.render(format)
    }

    /// 生成文本报告
    pub fn generate_report(&self, report: &UnderstandingReport) -> String {
        let mut output = String::new();

        output.push_str(&format!("# 代码理解报告：{}\n\n", report.project_name));

        // 架构概览
        output.push_str("## 1. 架构概览\n");
        output.push_str(&format!("- **模式**: {:?}\n", report.architecture.pattern));
        output.push_str(&format!(
            "- **核心特征**: {}\n\n",
            report.architecture.characteristics.join(", ")
        ));

        // 模块统计
        output.push_str("## 2. 模块结构\n");
        output.push_str(&format!("发现 {} 个主要模块：\n\n", report.modules.len()));
        for module in &report.modules {
            output.push_str(&format!("### {}\n", module.name));
            output.push_str(&format!("- **职责**: {}\n", module.responsibility));
            output.push_str(&format!("- **导出**: {}\n", module.exports.join(", ")));
            if !module.dependencies.is_empty() {
                output.push_str(&format!("- **依赖**: {}\n", module.dependencies.join(", ")));
            }
            output.push_str("\n");
        }

        // 数据流
        output.push_str("## 3. 数据流分析\n");
        output.push_str(&format!(
            "识别到 {} 条主要数据流：\n\n",
            report.dataflows.len()
        ));
        for (i, flow) in report.dataflows.iter().take(10).enumerate() {
            output.push_str(&format!(
                "{}. {} → {}\n   {}\n",
                i + 1,
                flow.source,
                flow.target,
                flow.description
            ));
        }
        if report.dataflows.len() > 10 {
            output.push_str(&format!("... 还有 {} 条\n\n", report.dataflows.len() - 10));
        }

        // API 列表
        output.push_str("## 4. 公共 API\n");
        output.push_str(&format!("导出 {} 个公共接口：\n\n", report.apis.len()));
        for api in report.apis.iter().take(20) {
            output.push_str(&format!("- `{}`\n", api.signature));
        }
        if report.apis.len() > 20 {
            output.push_str(&format!("... 还有 {} 个\n\n", report.apis.len() - 20));
        }

        // 统计信息
        output.push_str("## 5. 统计数据\n");
        output.push_str(&format!(
            "- **代码行数**: {}\n",
            report.statistics.line_count
        ));
        output.push_str(&format!(
            "- **函数数量**: {}\n",
            report.statistics.function_count
        ));
        output.push_str(&format!(
            "- **变量数量**: {}\n",
            report.statistics.variable_count
        ));
        output.push_str(&format!(
            "- **复杂度等级**: {}\n",
            report.statistics.complexity_level
        ));

        output
    }
}

impl Default for CodeUnderstandingSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// 代码理解报告
#[derive(Debug, Clone)]
pub struct UnderstandingReport {
    pub project_name: String,
    pub architecture: ArchitectureInfo,
    pub modules: Vec<ModuleInfo>,
    pub dataflows: Vec<DataflowInfo>,
    pub apis: Vec<ApiInfo>,
    pub statistics: ReportStatistics,
}

/// 架构信息
#[derive(Debug, Clone)]
pub struct ArchitectureInfo {
    pub pattern: ArchitecturePattern,
    pub characteristics: Vec<String>,
    pub description: String,
}

/// 架构模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitecturePattern {
    /// 单文件单体
    Monolithic,
    /// 模块化（CommonJS/ES6 modules）
    Modular,
    /// MVC 架构
    MVC,
    /// MVVM 架构（如 Vue）
    MVVM,
    /// 插件架构
    Plugin,
    /// 库/工具集
    Library,
    /// 混合
    Hybrid,
    /// 未知
    Unknown,
}

/// 模块信息
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub name: String,
    pub responsibility: String,
    pub exports: Vec<String>,
    pub dependencies: Vec<String>,
    pub functions: Vec<String>,
    pub variables: Vec<String>,
    pub size: usize, // 行数
}

/// 数据流信息
#[derive(Debug, Clone)]
pub struct DataflowInfo {
    pub source: String,
    pub target: String,
    pub description: String,
    pub flow_type: DataflowType,
}

/// 数据流类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataflowType {
    /// 函数调用
    FunctionCall,
    /// 数据传递
    DataPassing,
    /// 事件传播
    EventPropagation,
    /// 状态管理
    StateManagement,
    /// 依赖注入
    DependencyInjection,
}

/// API 信息
#[derive(Debug, Clone)]
pub struct ApiInfo {
    pub name: String,
    pub signature: String,
    pub description: String,
    pub params: Vec<ParamInfo>,
    pub return_type: String,
    pub examples: Vec<String>,
}

/// 参数信息
#[derive(Debug, Clone)]
pub struct ParamInfo {
    pub name: String,
    pub typ: String,
    pub description: String,
}

/// 报告统计信息
#[derive(Debug, Clone)]
pub struct ReportStatistics {
    pub line_count: usize,
    pub function_count: usize,
    pub variable_count: usize,
    pub class_count: usize,
    pub module_count: usize,
    pub complexity_level: String, // "Low", "Medium", "High", "Very High"
}

impl ReportStatistics {
    pub fn compute(code: &str) -> Self {
        let line_count = code.lines().count();
        let function_count = code.matches("function ").count() + code.matches(" => ").count();
        let variable_count = code.matches("var ").count()
            + code.matches("let ").count()
            + code.matches("const ").count();
        let class_count = code.matches("class ").count();
        let module_count = code.matches("export ").count();

        let complexity_level = if line_count < 200 {
            "Low".to_string()
        } else if line_count < 1000 {
            "Medium".to_string()
        } else if line_count < 5000 {
            "High".to_string()
        } else {
            "Very High".to_string()
        };

        Self {
            line_count,
            function_count,
            variable_count,
            class_count,
            module_count,
            complexity_level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_creation() {
        let _system = CodeUnderstandingSystem::new();
        assert!(true);
    }
}
