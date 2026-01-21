//! Behavior Record Data Models
//!
//! Defines data structures for capturing and recording browser behavior during page execution.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Behavior record capturing all interactions during page execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorRecord {
    /// Associated page URL
    pub page_url: String,

    /// Recording start timestamp
    pub start_timestamp: chrono::DateTime<chrono::Utc>,

    /// Recording end timestamp
    pub end_timestamp: chrono::DateTime<chrono::Utc>,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// API call records
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub api_calls: Vec<ApiCallRecord>,

    /// State change records
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub state_changes: Vec<StateChangeRecord>,

    /// Event flow records
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<EventFlowRecord>,

    /// Network request records
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub network_requests: Vec<NetworkRequestRecord>,

    /// Execution snapshots
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub snapshots: Vec<ExecutionSnapshot>,

    /// DOM mutations observed
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dom_mutations: Vec<DomMutation>,

    /// Console logs captured
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub console_logs: Vec<ConsoleLog>,
}

impl BehaviorRecord {
    /// Create a new empty behavior record
    pub fn new(page_url: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            page_url,
            start_timestamp: now,
            end_timestamp: now,
            duration_ms: 0,
            api_calls: Vec::new(),
            state_changes: Vec::new(),
            events: Vec::new(),
            network_requests: Vec::new(),
            snapshots: Vec::new(),
            dom_mutations: Vec::new(),
            console_logs: Vec::new(),
        }
    }

    /// Finalize the record with end timestamp
    pub fn finalize(&mut self) {
        self.end_timestamp = chrono::Utc::now();
        self.duration_ms = (self.end_timestamp - self.start_timestamp).num_milliseconds() as u64;
    }

    /// Add an API call record
    pub fn add_api_call(&mut self, call: ApiCallRecord) {
        self.api_calls.push(call);
    }

    /// Add a state change record
    pub fn add_state_change(&mut self, change: StateChangeRecord) {
        self.state_changes.push(change);
    }

    /// Add an event record
    pub fn add_event(&mut self, event: EventFlowRecord) {
        self.events.push(event);
    }

    /// Add a network request record
    pub fn add_network_request(&mut self, request: NetworkRequestRecord) {
        self.network_requests.push(request);
    }

    /// Add a console log
    pub fn add_console_log(&mut self, log: ConsoleLog) {
        self.console_logs.push(log);
    }

    /// Get summary statistics
    pub fn summary(&self) -> BehaviorSummary {
        let mut api_counts: HashMap<String, usize> = HashMap::new();
        let mut network_by_method: HashMap<String, usize> = HashMap::new();

        for call in &self.api_calls {
            *api_counts.entry(call.api_path.clone()).or_insert(0) += 1;
        }

        for request in &self.network_requests {
            *network_by_method.entry(request.method.clone()).or_insert(0) += 1;
        }

        BehaviorSummary {
            total_api_calls: self.api_calls.len(),
            total_state_changes: self.state_changes.len(),
            total_events: self.events.len(),
            total_network_requests: self.network_requests.len(),
            top_api_calls: api_counts,
            network_by_method,
            has_console_errors: self.console_logs.iter().any(|l| l.level == "error"),
        }
    }
}

/// Summary of recorded behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorSummary {
    pub total_api_calls: usize,
    pub total_state_changes: usize,
    pub total_events: usize,
    pub total_network_requests: usize,
    pub top_api_calls: HashMap<String, usize>,
    pub network_by_method: HashMap<String, usize>,
    pub has_console_errors: bool,
}

/// API call record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiCallRecord {
    /// Call timestamp (milliseconds from start)
    pub timestamp_ms: u64,

    /// Call location in code
    pub call_site: Option<CodeLocation>,

    /// API path (e.g., "fetch", "localStorage.setItem")
    pub api_path: String,

    /// Call arguments (serialized)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,

    /// Return value (for synchronous calls)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_value: Option<serde_json::Value>,

    /// Async operation ID (if async)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub async_id: Option<u64>,

    /// Call duration in milliseconds (if measured)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,

    /// Whether the call threw an error
    #[serde(default)]
    pub had_error: bool,
}

/// Code location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeLocation {
    /// File or script URL
    pub file: String,

    /// Line number
    pub line: u32,

    /// Column number
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<u32>,
}

/// State change record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateChangeRecord {
    /// Change timestamp
    pub timestamp_ms: u64,

    /// Target path (e.g., "localStorage.userName", "window.data")
    pub target: String,

    /// Previous value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_value: Option<serde_json::Value>,

    /// New value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_value: Option<serde_json::Value>,

    /// Source location
    pub source_location: Option<CodeLocation>,

    /// Change type
    pub change_type: ChangeType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    #[serde(rename = "assignment")]
    Assignment,
    #[serde(rename = "property_add")]
    PropertyAdd,
    #[serde(rename = "property_remove")]
    PropertyRemove,
    #[serde(rename = "array_push")]
    ArrayPush,
    #[serde(rename = "array_pop")]
    ArrayPop,
    #[serde(rename = "object_merge")]
    ObjectMerge,
    #[serde(rename = "other")]
    Other,
}

/// Event flow record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFlowRecord {
    /// Event timestamp
    pub timestamp_ms: u64,

    /// Event type (e.g., "click", "submit", "input")
    pub event_type: String,

    /// Target element selector
    pub target_element: String,

    /// Event handlers that were triggered
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub handlers: Vec<EventHandlerInfo>,

    /// Propagation path (bubbling)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub propagation_path: Vec<String>,

    /// Whether preventDefault was called
    #[serde(default)]
    pub default_prevented: bool,

    /// Whether stopPropagation was called
    #[serde(default)]
    pub propagation_stopped: bool,

    /// Event data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_data: Option<serde_json::Value>,
}

/// Information about an event handler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventHandlerInfo {
    /// Handler function name or ID
    pub handler_id: String,

    /// Handler location in code
    pub location: Option<CodeLocation>,

    /// Whether handler prevented default
    #[serde(default)]
    pub prevented_default: bool,
}

/// Network request record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRequestRecord {
    /// Request timestamp
    pub timestamp_ms: u64,

    /// Request ID
    pub request_id: String,

    /// HTTP method
    pub method: String,

    /// Request URL
    pub url: String,

    /// Request headers
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub request_headers: HashMap<String, String>,

    /// Request body
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_body: Option<String>,

    /// Response status code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_status: Option<u16>,

    /// Response headers
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub response_headers: HashMap<String, String>,

    /// Response body (truncated if too large)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_body: Option<String>,

    /// Request duration in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,

    /// Whether the request failed
    #[serde(default)]
    pub failed: bool,

    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

/// Execution snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSnapshot {
    /// Snapshot timestamp
    pub timestamp_ms: u64,

    /// Call stack at snapshot time
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub call_stack: Vec<StackFrame>,

    /// Variable values at snapshot time
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub variables: HashMap<String, serde_json::Value>,

    /// `this` binding
    #[serde(skip_serializing_if = "Option::is_none")]
    pub this_binding: Option<serde_json::Value>,
}

/// Stack frame information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    /// Function name
    pub function_name: String,

    /// Script URL
    pub script_url: String,

    /// Line number
    pub line: u32,

    /// Column number
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<u32>,
}

/// DOM mutation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomMutation {
    /// Mutation timestamp
    pub timestamp_ms: u64,

    /// Mutation type
    pub mutation_type: DomMutationType,

    /// Target element selector
    pub target: String,

    /// Added nodes (for childList mutations)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub added_nodes: Vec<String>,

    /// Removed nodes (for childList mutations)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub removed_nodes: Vec<String>,

    /// Previous sibling (for inserted mutations)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_sibling: Option<String>,

    /// Attribute changes (for attributes mutations)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attribute_changes: Vec<AttributeChange>,

    /// Character data changes (for characterData mutations)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomMutationType {
    #[serde(rename = "childList")]
    ChildList,
    #[serde(rename = "attributes")]
    Attributes,
    #[serde(rename = "characterData")]
    CharacterData,
}

/// Attribute change information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeChange {
    pub attribute_name: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
}

/// Console log record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsoleLog {
    /// Log timestamp
    pub timestamp_ms: u64,

    /// Log level
    pub level: String,

    /// Log message
    pub message: String,

    /// Additional arguments
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub arguments: Vec<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_behavior_record_creation() {
        let record = BehaviorRecord::new("https://example.com".to_string());
        assert_eq!(record.page_url, "https://example.com");
        assert!(record.api_calls.is_empty());
    }

    #[test]
    fn test_api_call_record() {
        let call = ApiCallRecord {
            timestamp_ms: 100,
            call_site: Some(CodeLocation {
                file: "script.js".to_string(),
                line: 10,
                column: Some(5),
            }),
            api_path: "fetch".to_string(),
            arguments: Some(serde_json::json!({"url": "https://api.example.com"})),
            return_value: None,
            async_id: Some(1),
            duration_ms: Some(50),
            had_error: false,
        };

        assert_eq!(call.api_path, "fetch");
        assert!(call.async_id.is_some());
    }

    #[test]
    fn test_network_request() {
        let request = NetworkRequestRecord {
            timestamp_ms: 50,
            request_id: "req-1".to_string(),
            method: "GET".to_string(),
            url: "https://api.example.com/data".to_string(),
            request_headers: HashMap::new(),
            request_body: None,
            response_status: Some(200),
            response_headers: HashMap::new(),
            response_body: Some(r#"{"success":true}"#.to_string()),
            duration_ms: Some(100),
            failed: false,
            error_message: None,
        };

        assert_eq!(request.method, "GET");
        assert_eq!(request.response_status, Some(200));
    }

    #[test]
    fn test_behavior_summary() {
        let mut record = BehaviorRecord::new("https://example.com".to_string());
        record.add_api_call(ApiCallRecord {
            timestamp_ms: 10,
            call_site: None,
            api_path: "fetch".to_string(),
            arguments: None,
            return_value: None,
            async_id: None,
            duration_ms: None,
            had_error: false,
        });
        record.add_api_call(ApiCallRecord {
            timestamp_ms: 20,
            call_site: None,
            api_path: "fetch".to_string(),
            arguments: None,
            return_value: None,
            async_id: None,
            duration_ms: None,
            had_error: false,
        });

        let summary = record.summary();
        assert_eq!(summary.total_api_calls, 2);
        assert_eq!(summary.top_api_calls["fetch"], 2);
    }
}
