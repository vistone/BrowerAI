//! WebView 面板模块 - 提供交互式 Web UI
//!
//! 通过 WebView 展示候选列表、性能指标、审计日志，
//! 并支持用户交互（应用候选、切换视图等）

use crate::style_switcher::{AuditEntry, CandidateSummary, StyleSwitcherBackend};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// WebView 面板的数据传输对象
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebViewState {
    /// 当前候选列表
    pub candidates: Vec<CandidateSummary>,
    /// 当前选中的候选 ID（如果有）
    pub selected_variant_id: Option<String>,
    /// 审计日志
    pub audit_log: Vec<AuditEntry>,
    /// 当前选卡（"candidates", "audit", "metrics"）
    pub current_tab: String,
    /// 性能指标（从 browerai-metrics 收集）
    pub metrics: Option<PerformanceMetrics>,
}

/// 性能指标数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// LCP（最大内容绘制）- 毫秒
    pub lcp_ms: f64,
    /// INP（交互到下次绘制）- 毫秒
    pub inp_ms: f64,
    /// CLS（累积布局偏移）- 0-1 之间
    pub cls: f64,
    /// 首字节时间 - 毫秒
    pub ttfb_ms: f64,
    /// 总加载时间 - 毫秒
    pub total_load_time_ms: f64,
    /// 渲染时间 - 毫秒
    pub render_time_ms: f64,
}

/// 来自 WebView 的用户操作
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebViewAction {
    /// 应用指定的候选
    ApplyCandidate { variant_id: String },
    /// 切换标签页
    SwitchTab { tab: String },
    /// 刷新数据
    Refresh,
    /// 导出数据（格式："json" 或 "csv"）
    Export { format: String },
}

/// WebView 面板事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebViewEvent {
    /// 成功应用候选
    CandidateApplied { variant_id: String },
    /// 数据已更新
    DataUpdated { state: WebViewState },
    /// 导出完成
    ExportCompleted { format: String, data: String },
    /// 错误发生
    Error { message: String },
}

/// WebView 面板管理器
pub struct WebViewPanel {
    backend: Box<dyn StyleSwitcherBackend>,
    metrics: Option<Arc<dyn MetricsProvider>>,
}

impl WebViewPanel {
    /// 创建新的 WebView 面板
    pub fn new(backend: Box<dyn StyleSwitcherBackend>) -> Self {
        Self {
            backend,
            metrics: None,
        }
    }

    /// 设置性能指标提供者
    pub fn with_metrics(mut self, metrics: Arc<dyn MetricsProvider>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// 获取当前的 WebView 状态
    pub fn get_state(&self, selected_variant_id: Option<String>) -> Result<WebViewState> {
        let candidates = self.backend.list_candidates()?;
        let audit_log = self.backend.audit_log()?;

        let metrics = if let Some(provider) = &self.metrics {
            Some(provider.collect_metrics()?)
        } else {
            None
        };

        Ok(WebViewState {
            candidates,
            selected_variant_id,
            audit_log,
            current_tab: "candidates".to_string(),
            metrics,
        })
    }

    /// 处理来自 WebView 的操作
    pub fn handle_action(&mut self, action: WebViewAction) -> Result<WebViewEvent> {
        match action {
            WebViewAction::ApplyCandidate { variant_id } => {
                self.backend.apply_candidate(&variant_id)?;
                Ok(WebViewEvent::CandidateApplied { variant_id })
            }
            WebViewAction::SwitchTab { tab: _ } => {
                let state = self.get_state(None)?;
                Ok(WebViewEvent::DataUpdated { state })
            }
            WebViewAction::Refresh => {
                let state = self.get_state(None)?;
                Ok(WebViewEvent::DataUpdated { state })
            }
            WebViewAction::Export { format } => {
                let candidates = self.backend.list_candidates()?;
                let data = if format.to_lowercase() == "json" {
                    serde_json::to_string_pretty(&candidates)?
                } else {
                    // CSV 格式导出
                    export_candidates_csv(&candidates)?
                };
                Ok(WebViewEvent::ExportCompleted { format, data })
            }
        }
    }

    /// 生成 HTML 面板内容
    pub fn render_html(&self) -> String {
        WEBVIEW_HTML.to_string()
    }

    /// 生成 CSS 样式
    pub fn render_css(&self) -> String {
        WEBVIEW_CSS.to_string()
    }

    /// 生成 JavaScript 初始化脚本
    pub fn render_js(&self) -> String {
        WEBVIEW_JS.to_string()
    }
}

/// 性能指标提供者 trait
pub trait MetricsProvider: Send + Sync {
    /// 收集当前的性能指标
    fn collect_metrics(&self) -> Result<PerformanceMetrics>;
}

/// 导出候选为 CSV 格式
fn export_candidates_csv(candidates: &[CandidateSummary]) -> Result<String> {
    let mut csv =
        String::from("variant_id,compatibility_score,accessibility_score,performance_score\n");
    for candidate in candidates {
        csv.push_str(&format!(
            "{},{:.2},{:.2},{:.2}\n",
            candidate.variant_id,
            candidate.compatibility_score,
            candidate.accessibility_score,
            candidate.performance_score
        ));
    }
    Ok(csv)
}

/// WebView HTML 模板
const WEBVIEW_HTML: &str = r#"<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BrowerAI DevTools - 样式切换面板</title>
    <style id="panel-styles"></style>
</head>
<body>
    <div class="panel-container">
        <!-- 头部 -->
        <div class="panel-header">
            <h1>🎨 BrowerAI 样式切换面板</h1>
            <div class="header-actions">
                <button id="btn-refresh" class="btn btn-primary" title="刷新数据">
                    <span>🔄 刷新</span>
                </button>
                <button id="btn-export-json" class="btn btn-secondary" title="导出为 JSON">
                    <span>📥 JSON</span>
                </button>
                <button id="btn-export-csv" class="btn btn-secondary" title="导出为 CSV">
                    <span>📥 CSV</span>
                </button>
            </div>
        </div>

        <!-- 标签导航 -->
        <div class="panel-tabs">
            <button class="tab-btn active" data-tab="candidates">候选列表</button>
            <button class="tab-btn" data-tab="metrics">性能指标</button>
            <button class="tab-btn" data-tab="audit">审计日志</button>
        </div>

        <!-- 候选列表面板 -->
        <div class="tab-content active" id="tab-candidates">
            <div class="candidates-container" id="candidates-list">
                <!-- 动态填充 -->
            </div>
        </div>

        <!-- 性能指标面板 -->
        <div class="tab-content" id="tab-metrics">
            <div class="metrics-grid" id="metrics-content">
                <!-- 动态填充 -->
            </div>
        </div>

        <!-- 审计日志面板 -->
        <div class="tab-content" id="tab-audit">
            <div class="audit-log" id="audit-log-content">
                <!-- 动态填充 -->
            </div>
        </div>

        <!-- 状态栏 -->
        <div class="panel-footer">
            <div class="status-indicator" id="status">就绪</div>
            <div class="info-text" id="info"></div>
        </div>
    </div>
</body>
</html>"#;

/// WebView CSS 样式
const WEBVIEW_CSS: &str = r#"
:root {
    --primary-color: #6366f1;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
    --gray-50: #f9fafb;
    --gray-100: #f3f4f6;
    --gray-200: #e5e7eb;
    --gray-300: #d1d5db;
    --gray-600: #4b5563;
    --gray-900: #111827;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: linear-gradient(135deg, var(--gray-50) 0%, var(--gray-100) 100%);
    color: var(--gray-900);
    font-size: 14px;
    line-height: 1.6;
}

.panel-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 16px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
}

/* 头部样式 */
.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 2px solid var(--gray-200);
}

.panel-header h1 {
    font-size: 24px;
    font-weight: 700;
    color: var(--gray-900);
}

.header-actions {
    display: flex;
    gap: 8px;
}

/* 按钮样式 */
.btn {
    padding: 8px 16px;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.btn-primary {
    background: var(--primary-color);
    color: white;
}

.btn-primary:hover {
    background: #4f46e5;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.btn-secondary {
    background: var(--gray-100);
    color: var(--gray-900);
    border: 1px solid var(--gray-300);
}

.btn-secondary:hover {
    background: var(--gray-200);
    border-color: var(--gray-400);
}

.btn-success {
    background: var(--success-color);
    color: white;
}

.btn-success:hover {
    background: #059669;
}

/* 标签导航 */
.panel-tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 24px;
    border-bottom: 1px solid var(--gray-200);
}

.tab-btn {
    padding: 12px 16px;
    background: none;
    border: none;
    border-bottom: 3px solid transparent;
    color: var(--gray-600);
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}

.tab-btn:hover {
    color: var(--primary-color);
}

.tab-btn.active {
    color: var(--primary-color);
    border-bottom-color: var(--primary-color);
}

/* 标签内容 */
.tab-content {
    display: none;
    animation: fadeIn 0.3s ease-in;
}

.tab-content.active {
    display: block;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* 候选卡片 */
.candidate-card {
    background: var(--gray-50);
    border: 1px solid var(--gray-200);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    transition: all 0.3s;
}

.candidate-card:hover {
    border-color: var(--primary-color);
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.15);
}

.candidate-card.selected {
    background: #f0f4ff;
    border-color: var(--primary-color);
}

.candidate-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.candidate-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--gray-900);
}

.candidate-badge {
    display: inline-block;
    padding: 4px 8px;
    background: var(--primary-color);
    color: white;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
}

.score-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin-bottom: 12px;
}

.score-item {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.score-label {
    font-size: 12px;
    color: var(--gray-600);
    font-weight: 600;
}

.score-bar {
    height: 8px;
    background: var(--gray-200);
    border-radius: 4px;
    overflow: hidden;
}

.score-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary-color), #06b6d4);
    border-radius: 4px;
    transition: width 0.3s ease;
}

.score-value {
    font-size: 14px;
    font-weight: 700;
    color: var(--primary-color);
}

.candidate-actions {
    display: flex;
    gap: 8px;
}

.candidate-actions .btn {
    flex: 1;
}

/* 性能指标 */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
}

.metric-card {
    background: linear-gradient(135deg, var(--gray-50), var(--gray-100));
    border: 1px solid var(--gray-200);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}

.metric-label {
    font-size: 12px;
    color: var(--gray-600);
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--primary-color);
    margin-bottom: 8px;
}

.metric-unit {
    font-size: 12px;
    color: var(--gray-600);
}

.metric-status {
    font-size: 12px;
    font-weight: 600;
    padding: 4px 8px;
    border-radius: 4px;
    margin-top: 8px;
}

.metric-status.good {
    background: #dbeafe;
    color: #0369a1;
}

.metric-status.warning {
    background: #fef3c7;
    color: #92400e;
}

.metric-status.critical {
    background: #fee2e2;
    color: #991b1b;
}

/* 审计日志 */
.audit-log {
    max-height: 400px;
    overflow-y: auto;
}

.audit-entry {
    background: var(--gray-50);
    border-left: 4px solid var(--primary-color);
    padding: 12px;
    margin-bottom: 8px;
    border-radius: 4px;
    font-size: 13px;
}

.audit-time {
    color: var(--gray-600);
    font-weight: 600;
}

.audit-action {
    color: var(--gray-900);
    margin-left: 8px;
}

/* 状态栏 */
.panel-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid var(--gray-200);
    font-size: 12px;
    color: var(--gray-600);
}

.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.status-indicator::before {
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    background: var(--success-color);
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* 响应式设计 */
@media (max-width: 768px) {
    .panel-container {
        padding: 12px;
    }

    .panel-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 12px;
    }

    .header-actions {
        width: 100%;
    }

    .header-actions .btn {
        flex: 1;
    }

    .score-grid {
        grid-template-columns: 1fr;
    }

    .metrics-grid {
        grid-template-columns: 1fr;
    }

    .candidate-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 8px;
    }
}
"#;

/// WebView JavaScript 交互脚本
const WEBVIEW_JS: &str = r#"
// WebView 面板的 JavaScript 控制器

class DevToolsPanel {
    constructor() {
        this.state = null;
        this.selectedVariantId = null;
        this.init();
    }

    async init() {
        this.attachEventListeners();
        await this.refresh();
        console.log('DevTools Panel initialized');
    }

    attachEventListeners() {
        // 标签页切换
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // 按钮事件
        document.getElementById('btn-refresh').addEventListener('click', () => this.refresh());
        document.getElementById('btn-export-json').addEventListener('click', () => this.export('json'));
        document.getElementById('btn-export-csv').addEventListener('click', () => this.export('csv'));
    }

    async refresh() {
        this.setStatus('刷新中...');
        try {
            // 发送刷新请求到 Rust 后端
            const event = {
                type: 'action',
                action: { Refresh: {} }
            };
            window.postMessage(event, '*');
            this.setStatus('就绪');
        } catch (error) {
            this.setError('刷新失败: ' + error.message);
        }
    }

    switchTab(tabName) {
        // 更新 UI
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === 'tab-' + tabName);
        });

        // 发送事件到后端
        const event = {
            type: 'action',
            action: { SwitchTab: { tab: tabName } }
        };
        window.postMessage(event, '*');
    }

    applyCandidate(variantId) {
        const event = {
            type: 'action',
            action: { ApplyCandidate: { variant_id: variantId } }
        };
        window.postMessage(event, '*');
        this.selectedVariantId = variantId;
        this.renderCandidates();
    }

    export(format) {
        this.setStatus('导出中...');
        const event = {
            type: 'action',
            action: { Export: { format: format } }
        };
        window.postMessage(event, '*');
    }

    // 从后端接收状态更新
    updateState(newState) {
        this.state = newState;
        this.renderCandidates();
        this.renderMetrics();
        this.renderAuditLog();
    }

    renderCandidates() {
        const container = document.getElementById('candidates-list');
        if (!this.state || !this.state.candidates) {
            container.innerHTML = '<p>无可用候选</p>';
            return;
        }

        container.innerHTML = this.state.candidates.map(candidate => `
            <div class="candidate-card ${candidate.variant_id === this.selectedVariantId ? 'selected' : ''}">
                <div class="candidate-header">
                    <span class="candidate-title">${this.escapeHtml(candidate.variant_id)}</span>
                    ${candidate.variant_id === this.selectedVariantId ? '<span class="candidate-badge">✓ 已应用</span>' : ''}
                </div>
                <div class="score-grid">
                    <div class="score-item">
                        <span class="score-label">兼容性</span>
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${candidate.compatibility_score * 100}%"></div>
                        </div>
                        <span class="score-value">${(candidate.compatibility_score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="score-item">
                        <span class="score-label">可访问性</span>
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${candidate.accessibility_score * 100}%"></div>
                        </div>
                        <span class="score-value">${(candidate.accessibility_score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="score-item">
                        <span class="score-label">性能</span>
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${candidate.performance_score * 100}%"></div>
                        </div>
                        <span class="score-value">${(candidate.performance_score * 100).toFixed(1)}%</span>
                    </div>
                </div>
                <div class="candidate-actions">
                    <button class="btn btn-success" onclick="panel.applyCandidate('${this.escapeHtml(candidate.variant_id)}')">
                        应用此候选
                    </button>
                </div>
            </div>
        `).join('');
    }

    renderMetrics() {
        const container = document.getElementById('metrics-content');
        if (!this.state || !this.state.metrics) {
            container.innerHTML = '<p>无可用指标数据</p>';
            return;
        }

        const metrics = this.state.metrics;
        container.innerHTML = `
            <div class="metric-card">
                <div class="metric-label">LCP（最大内容绘制）</div>
                <div class="metric-value">${metrics.lcp_ms.toFixed(1)}</div>
                <div class="metric-unit">毫秒</div>
                <div class="metric-status ${metrics.lcp_ms < 2500 ? 'good' : 'critical'}">
                    ${metrics.lcp_ms < 2500 ? '✓ 优' : '✗ 需优化'}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">INP（交互响应）</div>
                <div class="metric-value">${metrics.inp_ms.toFixed(1)}</div>
                <div class="metric-unit">毫秒</div>
                <div class="metric-status ${metrics.inp_ms < 200 ? 'good' : 'critical'}">
                    ${metrics.inp_ms < 200 ? '✓ 优' : '✗ 需优化'}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">CLS（布局稳定性）</div>
                <div class="metric-value">${metrics.cls.toFixed(3)}</div>
                <div class="metric-unit">得分</div>
                <div class="metric-status ${metrics.cls < 0.1 ? 'good' : 'critical'}">
                    ${metrics.cls < 0.1 ? '✓ 优' : '✗ 需优化'}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">TTFB（首字节时间）</div>
                <div class="metric-value">${metrics.ttfb_ms.toFixed(1)}</div>
                <div class="metric-unit">毫秒</div>
                <div class="metric-status ${metrics.ttfb_ms < 600 ? 'good' : 'warning'}">
                    ${metrics.ttfb_ms < 600 ? '✓ 良好' : '⚠ 尚可'}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">总加载时间</div>
                <div class="metric-value">${metrics.total_load_time_ms.toFixed(0)}</div>
                <div class="metric-unit">毫秒</div>
                <div class="metric-status good">即时更新</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">渲染时间</div>
                <div class="metric-value">${metrics.render_time_ms.toFixed(1)}</div>
                <div class="metric-unit">毫秒</div>
                <div class="metric-status ${metrics.render_time_ms < 100 ? 'good' : 'warning'}">
                    ${metrics.render_time_ms < 100 ? '✓ 快速' : '⚠ 适中'}
                </div>
            </div>
        `;
    }

    renderAuditLog() {
        const container = document.getElementById('audit-log-content');
        if (!this.state || !this.state.audit_log || this.state.audit_log.length === 0) {
            container.innerHTML = '<p>审计日志为空</p>';
            return;
        }

        container.innerHTML = this.state.audit_log.map((entry, idx) => `
            <div class="audit-entry">
                <span class="audit-time">[${idx + 1}]</span>
                <span class="audit-action">${this.escapeHtml(entry.action)} → ${this.escapeHtml(entry.variant_id)}</span>
            </div>
        `).join('');
    }

    setStatus(message) {
        const statusEl = document.getElementById('status');
        if (statusEl) {
            statusEl.textContent = message;
        }
    }

    setInfo(message) {
        const infoEl = document.getElementById('info');
        if (infoEl) {
            infoEl.textContent = message;
        }
    }

    setError(message) {
        this.setStatus('错误');
        this.setInfo(message);
        console.error(message);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// 初始化全局面板实例
const panel = new DevToolsPanel();

// 监听来自 Rust 后端的消息
window.addEventListener('message', (event) => {
    const data = event.data;
    if (data.type === 'state_update') {
        panel.updateState(data.state);
        panel.setStatus('就绪');
    } else if (data.type === 'export_complete') {
        panel.setInfo(`已导出 ${data.format}`);
        // 触发下载
        const blob = new Blob([data.data], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `browerai-candidates.${data.format === 'json' ? 'json' : 'csv'}`;
        a.click();
        URL.revokeObjectURL(url);
    } else if (data.type === 'error') {
        panel.setError(data.message);
    }
});
"#;
