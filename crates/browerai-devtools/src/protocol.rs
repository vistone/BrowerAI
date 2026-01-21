//! 前后端通信协议 - WebView 与 Rust 后端的交互
//!
//! 定义 WebView JavaScript 前端与 Rust 后端之间的消息协议

use serde::{Deserialize, Serialize};

/// 前端发送给后端的消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontendMessage {
    /// 消息类型（"action", "query", "heartbeat"）
    #[serde(rename = "type")]
    pub msg_type: String,
    /// 消息载荷
    pub payload: serde_json::Value,
}

/// 后端发送给前端的消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendMessage {
    /// 消息类型（"state_update", "event", "error"）
    #[serde(rename = "type")]
    pub msg_type: String,
    /// 消息载荷
    pub payload: serde_json::Value,
}

pub mod protocol_spec {
    //! WebView 通信协议说明
    //!
    //! ## 消息流向
    //!
    //! ```text
    //! 前端 (JavaScript)          <-->        后端 (Rust)
    //! |                                        |
    //! +-- 发送 Action             -->        处理 Action
    //! |   (apply_candidate,                   (StyleSwitcherBackend)
    //! |    refresh,                           |
    //! |    export)                            V
    //!                                    生成 Event
    //! |                                        |
    //! <-- 接收 State Update  <--        发送 State/Event
    //! |   (candidates,                        
    //! |    metrics,
    //! |    audit_log)
    //! ```
    //!
    //! ## 前端发送的消息格式
    //!
    //! ### 1. 应用候选
    //! ```json
    //! {
    //!   "type": "action",
    //!   "action": { "ApplyCandidate": { "variant_id": "minimal" } }
    //! }
    //! ```
    //!
    //! ### 2. 刷新数据
    //! ```json
    //! {
    //!   "type": "action",
    //!   "action": { "Refresh": {} }
    //! }
    //! ```
    //!
    //! ### 3. 切换标签页
    //! ```json
    //! {
    //!   "type": "action",
    //!   "action": { "SwitchTab": { "tab": "metrics" } }
    //! }
    //! ```
    //!
    //! ### 4. 导出数据
    //! ```json
    //! {
    //!   "type": "action",
    //!   "action": { "Export": { "format": "json" } }
    //! }
    //! ```
    //!
    //! ## 后端发送的消息格式
    //!
    //! ### 1. 状态更新
    //! ```json
    //! {
    //!   "type": "state_update",
    //!   "state": {
    //!     "candidates": [
    //!       { "variant_id": "minimal", "compatibility_score": 0.95, ... },
    //!       ...
    //!     ],
    //!     "metrics": {
    //!       "lcp_ms": 2000.0,
    //!       "inp_ms": 100.0,
    //!       ...
    //!     },
    //!     "audit_log": [
    //!       { "action": "switch", "variant_id": "minimal" },
    //!       ...
    //!     ],
    //!     "current_tab": "candidates"
    //!   }
    //! }
    //! ```
    //!
    //! ### 2. 事件通知
    //! ```json
    //! {
    //!   "type": "event",
    //!   "event": {
    //!     "CandidateApplied": { "variant_id": "minimal" }
    //!   }
    //! }
    //! ```
    //!
    //! ### 3. 导出完成
    //! ```json
    //! {
    //!   "type": "export_complete",
    //!   "format": "json",
    //!   "data": "[{...}]"
    //! }
    //! ```
    //!
    //! ### 4. 错误信息
    //! ```json
    //! {
    //!   "type": "error",
    //!   "message": "无法应用候选：variant_id 不存在"
    //! }
    //! ```
    //!
    //! ## 实现指南
    //!
    //! ### Rust 后端实现
    //! ```ignore
    //! use browerai_devtools::webview::WebViewPanel;
    //!
    //! // 创建面板
    //! let panel = WebViewPanel::new(backend).with_metrics(metrics);
    //!
    //! // 获取初始 HTML/CSS/JS
    //! let html = panel.render_html();
    //! let css = panel.render_css();
    //! let js = panel.render_js();
    //!
    //! // 处理来自前端的消息
    //! let action = serde_json::from_value(message)?;
    //! let event = panel.handle_action(action)?;
    //! let response = serde_json::to_value(event)?;
    //! ```
    //!
    //! ### JavaScript 前端实现
    //! ```javascript
    //! class DevToolsPanel {
    //!   async applyCandidate(variantId) {
    //!     const action = { ApplyCandidate: { variant_id: variantId } };
    //!     window.postMessage({ type: 'action', action }, '*');
    //!   }
    //!
    //!   updateState(newState) {
    //!     this.state = newState;
    //!     this.render();
    //!   }
    //! }
    //! ```
    //!
    //! ## 错误处理
    //!
    //! ### 网络错误
    //! - 后端无响应：前端显示"连接错误"
    //! - 消息格式错误：后端返回 400 Bad Request
    //!
    //! ### 业务错误
    //! - 候选不存在：返回 404 NotFound
    //! - 权限不足：返回 403 Forbidden
    //!
    //! ## 超时处理
    //! - 默认超时：5 秒
    //! - 导出操作：15 秒（可能较大的数据量）
    //!
    //! ## 会话管理
    //! - 无状态设计：每个请求都是独立的
    //! - WebSocket（可选）：用于实时性能指标推送
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frontend_message_serialization() {
        let msg = FrontendMessage {
            msg_type: "action".to_string(),
            payload: serde_json::json!({
                "ApplyCandidate": { "variant_id": "test" }
            }),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let restored: FrontendMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.msg_type, "action");
    }

    #[test]
    fn test_backend_message_serialization() {
        let msg = BackendMessage {
            msg_type: "state_update".to_string(),
            payload: serde_json::json!({
                "candidates": [],
                "metrics": null,
                "audit_log": [],
                "current_tab": "candidates"
            }),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let restored: BackendMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.msg_type, "state_update");
    }
}
