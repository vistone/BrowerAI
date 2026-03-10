//! 探索报告生成器

use crate::*;
use anyhow::Result;
use std::collections::HashMap;

/// 报告生成器
pub struct ExplorationReporter;

impl ExplorationReporter {
    pub fn new() -> Self {
        Self
    }

    /// 生成详细报告
    pub fn generate_detailed_report(&self, report: &ExplorationReport) -> String {
        let mut output = String::new();

        // 报告头部
        output.push_str(&self.format_header(report));
        
        // 摘要
        output.push_str(&self.format_summary(report));
        
        // 覆盖率详情
        output.push_str(&self.format_coverage(&report.coverage));
        
        // 行为模式
        output.push_str(&self.format_behaviors(&report.unique_behaviors));
        
        // 页面详情
        output.push_str(&self.format_pages(&report.pages_explored));
        
        // 错误记录
        if !report.errors.is_empty() {
            output.push_str(&self.format_errors(&report.errors));
        }

        output
    }

    /// 生成JSON报告
    pub fn generate_json_report(&self, report: &ExplorationReport) -> Result<String> {
        Ok(serde_json::to_string_pretty(report)?)
    }

    /// 生成HTML报告
    pub fn generate_html_report(&self, report: &ExplorationReport) -> String {
        let duration = report.end_time.signed_duration_since(report.start_time);
        
        format!(r#"
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>探索报告 - {}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f6f8fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #586069;
            font-size: 14px;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #24292e;
            border-bottom: 1px solid #e1e4e8;
            padding-bottom: 10px;
        }}
        .coverage-bar {{
            height: 20px;
            background: #e1e4e8;
            border-radius: 10px;
            overflow: hidden;
        }}
        .coverage-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #34d058);
            transition: width 0.3s;
        }}
        .behavior-item {{
            padding: 15px;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            margin-bottom: 10px;
        }}
        .behavior-type {{
            font-weight: 600;
            color: #0366d6;
        }}
        .page-item {{
            padding: 15px;
            background: #f6f8fa;
            border-radius: 6px;
            margin-bottom: 10px;
        }}
        .page-url {{
            font-family: monospace;
            color: #0366d6;
        }}
        .error-item {{
            padding: 10px;
            background: #ffeef0;
            border-left: 3px solid #d73a49;
            margin-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e1e4e8;
        }}
        th {{
            background: #f6f8fa;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 网站探索报告</h1>
        <p>目标: {} | 探索时长: {}分钟</p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{}</div>
            <div class="stat-label">探索页面数</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{}</div>
            <div class="stat-label">观察记录数</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{}</div>
            <div class="stat-label">识别行为模式</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{:.1}%</div>
            <div class="stat-label">元素覆盖率</div>
        </div>
    </div>

    <div class="section">
        <h2>📊 覆盖率详情</h2>
        <div class="coverage-bar">
            <div class="coverage-fill" style="width: {:.1}%"></div>
        </div>
        <p>已探索 {} / {} 个元素</p>
        
        <table>
            <tr>
                <th>元素类型</th>
                <th>总数</th>
                <th>已探索</th>
                <th>覆盖率</th>
            </tr>
            {}
        </table>
    </div>

    <div class="section">
        <h2>🎯 识别的行为模式</h2>
        {}
    </div>

    <div class="section">
        <h2>📄 探索的页面</h2>
        {}
    </div>

    <div class="section">
        <h2>⚠️ 错误记录</h2>
        {}
    </div>

    <footer style="text-align: center; color: #586069; padding: 20px;">
        <p>生成时间: {}</p>
    </footer>
</body>
</html>
"#,
            report.target_url,
            report.target_url,
            duration.num_minutes(),
            report.pages_explored.len(),
            report.total_observations,
            report.unique_behaviors.len(),
            report.coverage.coverage_percentage,
            report.coverage.coverage_percentage,
            report.coverage.explored_elements,
            report.coverage.total_elements,
            self.format_coverage_table(&report.coverage.by_type),
            self.format_behaviors_html(&report.unique_behaviors),
            self.format_pages_html(&report.pages_explored),
            self.format_errors_html(&report.errors),
            report.end_time.format("%Y-%m-%d %H:%M:%S")
        )
    }

    fn format_header(&self, report: &ExplorationReport) -> String {
        let duration = report.end_time.signed_duration_since(report.start_time);
        
        format!(r#"
╔══════════════════════════════════════════════════════════════╗
║                    网站探索报告                               ║
╠══════════════════════════════════════════════════════════════╣
║ 目标URL: {:<50} ║
║ 开始时间: {:<49} ║
║ 结束时间: {:<49} ║
║ 探索时长: {:<49} ║
╚══════════════════════════════════════════════════════════════╝
"#,
            report.target_url,
            report.start_time.format("%Y-%m-%d %H:%M:%S"),
            report.end_time.format("%Y-%m-%d %H:%M:%S"),
            format!("{}分钟", duration.num_minutes())
        )
    }

    fn format_summary(&self, report: &ExplorationReport) -> String {
        format!(r#"
【摘要】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  探索页面数:     {}
  观察记录数:     {}
  识别行为模式:   {}
  元素覆盖率:     {:.1}%
  错误数:         {}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"#,
            report.pages_explored.len(),
            report.total_observations,
            report.unique_behaviors.len(),
            report.coverage.coverage_percentage,
            report.errors.len()
        )
    }

    fn format_coverage(&self, coverage: &CoverageReport) -> String {
        let mut output = String::from("\n【覆盖率详情】\n");
        output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        output.push_str(&format!("总元素数: {} | 已探索: {} | 覆盖率: {:.1}%\n",
            coverage.total_elements,
            coverage.explored_elements,
            coverage.coverage_percentage
        ));
        output.push_str("\n按类型统计:\n");

        let mut types: Vec<_> = coverage.by_type.iter().collect();
        types.sort_by(|a, b| b.1.percentage.partial_cmp(&a.1.percentage).unwrap());

        for (element_type, stats) in types {
            let bar = self.render_progress_bar(stats.percentage, 30);
            output.push_str(&format!("  {:12} {:30} {:3.0}% ({}/{})\n",
                element_type,
                bar,
                stats.percentage,
                stats.explored,
                stats.total
            ));
        }

        output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        output
    }

    fn format_behaviors(&self, behaviors: &[BehaviorPattern]) -> String {
        let mut output = String::from("\n【识别的行为模式】\n");
        output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for (idx, behavior) in behaviors.iter().enumerate() {
            output.push_str(&format!("\n[{}] {}\n", idx + 1, behavior.pattern_id));
            output.push_str(&format!("  类型: {:?}\n", behavior.pattern_type));
            output.push_str(&format!("  触发: {:?}\n", behavior.trigger));
            output.push_str(&format!("  频率: {}次\n", behavior.frequency));
            output.push_str(&format!("  置信度: {:.0}%\n", behavior.confidence * 100.0));
            
            if !behavior.typical_targets.is_empty() {
                output.push_str(&format!("  典型目标: {}\n", 
                    behavior.typical_targets.join(", ")));
            }
        }

        output.push_str("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        output
    }

    fn format_pages(&self, pages: &[PageExploration]) -> String {
        let mut output = String::from("\n【探索的页面】\n");
        output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for (idx, page) in pages.iter().enumerate() {
            output.push_str(&format!("\n[{}] {}\n", idx + 1, page.title));
            output.push_str(&format!("  URL: {}\n", page.url));
            output.push_str(&format!("  访问次数: {}\n", page.visit_count));
            output.push_str(&format!("  交互数: {}\n", page.interactions.len()));
            output.push_str(&format!("  发现元素: {}\n", page.elements_found.len()));
            output.push_str(&format!("  已探索: {}\n", page.explored_elements.len()));
        }

        output.push_str("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        output
    }

    fn format_errors(&self, errors: &[ExplorationError]) -> String {
        let mut output = String::from("\n【错误记录】\n");
        output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for error in errors {
            output.push_str(&format!("\n[{}] {}\n", 
                error.timestamp.format("%H:%M:%S"),
                if error.recoverable { "⚠️" } else { "❌" }
            ));
            output.push_str(&format!("  URL: {}\n", error.url));
            output.push_str(&format!("  操作: {}\n", error.action));
            output.push_str(&format!("  错误: {}\n", error.error_message));
        }

        output.push_str("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        output
    }

    fn render_progress_bar(&self, percentage: f64, width: usize) -> String {
        let filled = ((percentage / 100.0) * width as f64) as usize;
        let empty = width - filled;
        
        format!("[{}{}]",
            "█".repeat(filled),
            "░".repeat(empty)
        )
    }

    fn format_coverage_table(&self, by_type: &HashMap<String, TypeCoverage>) -> String {
        let mut rows = String::new();
        
        let mut types: Vec<_> = by_type.iter().collect();
        types.sort_by(|a, b| b.1.percentage.partial_cmp(&a.1.percentage).unwrap());

        for (element_type, stats) in types {
            rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.1}%</td></tr>",
                element_type, stats.total, stats.explored, stats.percentage
            ));
        }

        rows
    }

    fn format_behaviors_html(&self, behaviors: &[BehaviorPattern]) -> String {
        if behaviors.is_empty() {
            return "<p>未发现行为模式</p>".to_string();
        }

        behaviors.iter()
            .map(|b| format!(
                r#"<div class="behavior-item">
                    <div class="behavior-type">{:?}</div>
                    <p>触发: {:?} | 频率: {}次 | 置信度: {:.0}%</p>
                </div>"#,
                b.pattern_type, b.trigger, b.frequency, b.confidence * 100.0
            ))
            .collect::<Vec<_>>()
            .join("")
    }

    fn format_pages_html(&self, pages: &[PageExploration]) -> String {
        if pages.is_empty() {
            return "<p>未探索页面</p>".to_string();
        }

        pages.iter()
            .map(|p| format!(
                r#"<div class="page-item">
                    <div class="page-url">{}</div>
                    <p>{} | 交互: {} | 元素: {}/{}</p>
                </div>"#,
                p.url, p.title, p.interactions.len(), 
                p.explored_elements.len(), p.elements_found.len()
            ))
            .collect::<Vec<_>>()
            .join("")
    }

    fn format_errors_html(&self, errors: &[ExplorationError]) -> String {
        if errors.is_empty() {
            return "<p>无错误记录 ✓</p>".to_string();
        }

        errors.iter()
            .map(|e| format!(
                r#"<div class="error-item">
                    <strong>{}</strong> - {}
                    <p>{}</p>
                </div>"#,
                e.action, e.url, e.error_message
            ))
            .collect::<Vec<_>>()
            .join("")
    }
}

impl Default for ExplorationReporter {
    fn default() -> Self {
        Self::new()
    }
}
