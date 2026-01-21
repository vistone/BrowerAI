//! Behavior Recorder
//!
//! Records browser behavior during page execution, capturing API calls,
//! state changes, events, and network requests.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::data_models::{
    ApiCallRecord, BehaviorRecord, ChangeType, CodeLocation, ConsoleLog, EventFlowRecord,
    EventHandlerInfo, NetworkRequestRecord, StateChangeRecord,
};

use std::collections::HashMap;
use tokio::sync::Mutex;

/// Configuration for behavior recording
#[derive(Debug, Clone)]
pub struct RecorderConfig {
    /// Whether to record API calls
    pub record_api_calls: bool,

    /// Whether to record state changes
    pub record_state_changes: bool,

    /// Whether to record events
    pub record_events: bool,

    /// Whether to record network requests
    pub record_network_requests: bool,

    /// Maximum number of events to record (0 = unlimited)
    pub max_events: usize,

    /// Sample rate for high-frequency events (1.0 = record all)
    pub sample_rate: f32,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            record_api_calls: true,
            record_state_changes: true,
            record_events: true,
            record_network_requests: true,
            max_events: 10000,
            sample_rate: 1.0,
        }
    }
}

/// Behavior Recorder
///
/// Captures and records browser behavior during page execution.
/// Designed to work with the V8 JavaScript engine for instrumentation.
#[derive(Debug, Clone)]
pub struct BehaviorRecorder {
    config: RecorderConfig,
    record: Arc<Mutex<BehaviorRecord>>,
    start_time: Instant,
    event_counter: Arc<AtomicU64>,
    sampling_counter: Arc<AtomicU64>,
}

impl Default for BehaviorRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl BehaviorRecorder {
    /// Create a new behavior recorder
    pub fn new() -> Self {
        Self::with_config(RecorderConfig::default())
    }

    /// Create a behavior recorder with custom configuration
    pub fn with_config(config: RecorderConfig) -> Self {
        Self {
            config,
            record: Arc::new(Mutex::new(BehaviorRecord::new(String::new()))),
            start_time: Instant::now(),
            event_counter: Arc::new(AtomicU64::new(0)),
            sampling_counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Start recording for a URL
    pub async fn start_recording(&self, url: &str) {
        let mut record = self.record.lock().await;
        *record = BehaviorRecord::new(url.to_string());
        // start_time is not reset to avoid time drift
    }

    /// Stop recording and finalize
    pub async fn stop_recording(&mut self) -> BehaviorRecord {
        let mut record = self.record.lock().await;
        record.finalize();
        record.clone()
    }

    /// Get current recording state
    pub async fn get_record(&self) -> BehaviorRecord {
        self.record.lock().await.clone()
    }

    /// Check if we should sample this event
    fn should_sample(&self) -> bool {
        if self.config.sample_rate >= 1.0 {
            return true;
        }

        let count = self.sampling_counter.fetch_add(1, Ordering::SeqCst);
        let sample_index = (count as f32 / self.config.sample_rate) as u64;
        count / sample_index == 1
    }

    /// Get elapsed time in milliseconds
    fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Record an API call
    pub async fn record_api_call(
        &self,
        api_path: String,
        arguments: Option<serde_json::Value>,
        return_value: Option<serde_json::Value>,
        async_id: Option<u64>,
        location: Option<CodeLocation>,
    ) {
        if !self.config.record_api_calls {
            return;
        }

        let call = ApiCallRecord {
            timestamp_ms: self.elapsed_ms(),
            call_site: location,
            api_path,
            arguments,
            return_value,
            async_id,
            duration_ms: None,
            had_error: false,
        };

        let mut record = self.record.lock().await;

        // Check max events limit
        if self.config.max_events > 0 && record.api_calls.len() >= self.config.max_events {
            return;
        }

        record.add_api_call(call);
    }

    /// Record a state change
    pub async fn record_state_change(
        &self,
        target: String,
        old_value: Option<serde_json::Value>,
        new_value: Option<serde_json::Value>,
        change_type: ChangeType,
        location: Option<CodeLocation>,
    ) {
        if !self.config.record_state_changes {
            return;
        }

        let change = StateChangeRecord {
            timestamp_ms: self.elapsed_ms(),
            target,
            old_value,
            new_value,
            source_location: location,
            change_type,
        };

        let mut record = self.record.lock().await;

        if self.config.max_events > 0 && record.state_changes.len() >= self.config.max_events {
            return;
        }

        record.add_state_change(change);
    }

    /// Record an event
    pub async fn record_event(
        &self,
        event_type: String,
        target_element: String,
        handlers: Vec<EventHandlerInfo>,
        propagation_path: Vec<String>,
        default_prevented: bool,
        propagation_stopped: bool,
    ) {
        if !self.config.record_events {
            return;
        }

        if !self.should_sample() {
            return;
        }

        let event = EventFlowRecord {
            timestamp_ms: self.elapsed_ms(),
            event_type,
            target_element,
            handlers,
            propagation_path,
            default_prevented,
            propagation_stopped,
            event_data: None,
        };

        let mut record = self.record.lock().await;

        if self.config.max_events > 0 && record.events.len() >= self.config.max_events {
            return;
        }

        record.add_event(event);
    }

    /// Record a network request
    pub async fn record_network_request(
        &self,
        method: String,
        url: String,
        request_body: Option<String>,
    ) -> String {
        if !self.config.record_network_requests {
            return String::new();
        }

        let request_id = format!("req-{}", self.event_counter.fetch_add(1, Ordering::SeqCst));

        let request = NetworkRequestRecord {
            timestamp_ms: self.elapsed_ms(),
            request_id: request_id.clone(),
            method,
            url,
            request_headers: HashMap::new(),
            request_body,
            response_status: None,
            response_headers: HashMap::new(),
            response_body: None,
            duration_ms: None,
            failed: false,
            error_message: None,
        };

        let mut record = self.record.lock().await;
        record.add_network_request(request);

        request_id
    }

    /// Update network request with response
    pub async fn update_network_response(
        &self,
        request_id: &str,
        status: u16,
        response_body: Option<String>,
    ) {
        let mut record = self.record.lock().await;

        for request in &mut record.network_requests {
            if request.request_id == request_id {
                request.response_status = Some(status);
                request.response_body = response_body;
                break;
            }
        }
    }

    /// Record console log
    pub async fn record_console_log(&self, level: String, message: String) {
        let log = ConsoleLog {
            timestamp_ms: self.elapsed_ms(),
            level,
            message,
            arguments: Vec::new(),
        };

        let mut record = self.record.lock().await;
        record.add_console_log(log);
    }

    /// Get summary of recorded behavior
    pub async fn get_summary(&self) -> crate::data_models::BehaviorSummary {
        self.record.lock().await.summary()
    }
}

/// Instrumented JavaScript wrapper
///
/// Provides JavaScript code that instruments the browser environment
/// to capture behavior and report back to the recorder.
pub fn get_instrumentation_script() -> &'static str {
    r#"
(function() {
    'use strict';
    
    // Configuration
    const MAX_EVENTS = 10000;
    const SAMPLE_RATE = 1.0;
    
    // State
    let eventCount = 0;
    let sampleCount = 0;
    const startTime = Date.now();
    
    // Helper functions
    function elapsed() {
        return Date.now() - startTime;
    }
    
    function shouldSample() {
        if (SAMPLE_RATE >= 1.0) return true;
        sampleCount++;
        return Math.floor(sampleCount / SAMPLE_RATE) === 1;
    }
    
    function sendRecord(type, data) {
        if (typeof window.broweraiRecorder !== 'undefined') {
            window.broweraiRecorder(type, data);
        }
        // In browser context, this would send to the recorder
        // For now, we just log
        console.log('[BrowerAI Record]', type, data);
    }
    
    // Intercept fetch
    const originalFetch = window.fetch;
    window.fetch = function(...args) {
        const url = typeof args[0] === 'string' ? args[0] : args[0].url;
        const method = args[1]?.method || 'GET';
        
        const requestId = 'req-' + (++eventCount);
        
        sendRecord('network_request', {
            requestId,
            method,
            url,
            timestamp: elapsed()
        });
        
        return originalFetch.apply(this, args).then(response => {
            sendRecord('network_response', {
                requestId,
                status: response.status,
                timestamp: elapsed()
            });
            return response;
        }).catch(error => {
            sendRecord('network_error', {
                requestId,
                error: error.message,
                timestamp: elapsed()
            });
            throw error;
        });
    };
    
    // Intercept XMLHttpRequest
    const originalOpen = XMLHttpRequest.prototype.open;
    const originalSend = XMLHttpRequest.prototype.send;
    
    XMLHttpRequest.prototype.open = function(method, url) {
        this._browerai_xhr = {
            method,
            url,
            requestId: 'req-' + (++eventCount)
        };
        sendRecord('network_request', {
            requestId: this._browerai_xhr.requestId,
            method,
            url,
            timestamp: elapsed()
        });
        return originalOpen.apply(this, arguments);
    };
    
    XMLHttpRequest.prototype.send = function(body) {
        this._browerai_xhr.body = body;
        return originalSend.apply(this, arguments)
            .then(() => {
                sendRecord('network_response', {
                    requestId: this._browerai_xhr.requestId,
                    status: this.status,
                    timestamp: elapsed()
                });
            })
            .catch(error => {
                sendRecord('network_error', {
                    requestId: this._browerai_xhr.requestId,
                    error: error.message,
                    timestamp: elapsed()
                });
                throw error;
            });
    };
    
    // Intercept localStorage
    const originalSetItem = localStorage.setItem;
    localStorage.setItem = function(key, value) {
        sendRecord('state_change', {
            target: 'localStorage.' + key,
            oldValue: localStorage.getItem(key),
            newValue: value,
            changeType: 'assignment',
            timestamp: elapsed()
        });
        return originalSetItem.apply(this, arguments);
    };
    
    // Intercept sessionStorage
    const originalSessionSetItem = sessionStorage.setItem;
    sessionStorage.setItem = function(key, value) {
        sendRecord('state_change', {
            target: 'sessionStorage.' + key,
            oldValue: sessionStorage.getItem(key),
            newValue: value,
            changeType: 'assignment',
            timestamp: elapsed()
        });
        return originalSessionSetItem.apply(this, arguments);
    };
    
    // Intercept addEventListener
    const originalAddEventListener = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function(type, handler, options) {
        // Wrap handler to track invocations
        const wrapped = function(...args) {
            sendRecord('event', {
                eventType: type,
                target: this.tagName.toLowerCase() + (this.id ? '#' + this.id : '') + (this.className ? '.' + this.className.split(' ').join('.') : ''),
                timestamp: elapsed()
            });
            return handler.apply(this, args);
        };
        return originalAddEventListener.apply(this, [type, wrapped, options]);
    };
    
    // Intercept console.log
    const originalLog = console.log;
    console.log = function(...args) {
        sendRecord('console', {
            level: 'log',
            message: args.join(' '),
            timestamp: elapsed()
        });
        return originalLog.apply(this, args);
    };
    
    const originalError = console.error;
    console.error = function(...args) {
        sendRecord('console', {
            level: 'error',
            message: args.join(' '),
            timestamp: elapsed()
        });
        return originalError.apply(this, args);
    };
    
    // Expose recorder interface
    window.broweraiRecorder = function(type, data) {
        // This will be replaced by the actual recorder in Rust
    };
    
    console.log('[BrowerAI] Behavior recording initialized');
})();
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_recorder_creation() {
        let recorder = BehaviorRecorder::new();
        assert!(recorder.config.record_api_calls);
        assert!(recorder.config.record_events);
    }

    #[tokio::test]
    async fn test_start_stop_recording() {
        let mut recorder = BehaviorRecorder::new();
        recorder.start_recording("https://example.com").await;

        let record = recorder.stop_recording().await;
        assert_eq!(record.page_url, "https://example.com");
        assert!(record.duration_ms >= 0);
    }

    #[tokio::test]
    async fn test_record_api_call() {
        let recorder = BehaviorRecorder::new();
        recorder.start_recording("https://example.com").await;

        recorder
            .record_api_call(
                "fetch".to_string(),
                Some(serde_json::json!({"url": "https://api.example.com"})),
                None,
                Some(1),
                Some(CodeLocation {
                    file: "script.js".to_string(),
                    line: 10,
                    column: Some(5),
                }),
            )
            .await;

        let record = recorder.get_record().await;
        assert_eq!(record.api_calls.len(), 1);
        assert_eq!(record.api_calls[0].api_path, "fetch");
    }

    #[tokio::test]
    async fn test_record_state_change() {
        let recorder = BehaviorRecorder::new();
        recorder.start_recording("https://example.com").await;

        recorder
            .record_state_change(
                "localStorage.userToken".to_string(),
                Some(serde_json::json!("old-token")),
                Some(serde_json::json!("new-token")),
                ChangeType::Assignment,
                None,
            )
            .await;

        let record = recorder.get_record().await;
        assert_eq!(record.state_changes.len(), 1);
    }

    #[test]
    fn test_instrumentation_script() {
        let script = get_instrumentation_script();
        assert!(script.contains("fetch"));
        assert!(script.contains("localStorage"));
        assert!(script.contains("addEventListener"));
        assert!(script.contains("console.log"));
    }
}
