use anyhow::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use markup5ever_rcdom::{Handle, NodeData, RcDom};

/// DOM node information for inspection
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub node_type: String,
    pub tag_name: Option<String>,
    pub attributes: HashMap<String, String>,
    pub text_content: Option<String>,
    pub child_count: usize,
}

/// DOM inspector for debugging
pub struct DOMInspector {
    dom: Option<RcDom>,
}

impl DOMInspector {
    pub fn new() -> Self {
        Self { dom: None }
    }

    pub fn set_dom(&mut self, dom: RcDom) {
        self.dom = Some(dom);
    }

    fn get_node_info(&self, handle: &Handle) -> NodeInfo {
        match &handle.data {
            NodeData::Element { name, attrs, .. } => {
                let mut attributes = HashMap::new();
                for attr in attrs.borrow().iter() {
                    attributes.insert(attr.name.local.to_string(), attr.value.to_string());
                }
                NodeInfo {
                    node_type: "Element".to_string(),
                    tag_name: Some(name.local.to_string()),
                    attributes,
                    text_content: None,
                    child_count: handle.children.borrow().len(),
                }
            }
            NodeData::Text { contents } => NodeInfo {
                node_type: "Text".to_string(),
                tag_name: None,
                attributes: HashMap::new(),
                text_content: Some(contents.borrow().to_string()),
                child_count: 0,
            },
            NodeData::Document => NodeInfo {
                node_type: "Document".to_string(),
                tag_name: None,
                attributes: HashMap::new(),
                text_content: None,
                child_count: handle.children.borrow().len(),
            },
            _ => NodeInfo {
                node_type: "Other".to_string(),
                tag_name: None,
                attributes: HashMap::new(),
                text_content: None,
                child_count: 0,
            },
        }
    }

    pub fn tree_view(&self) -> Result<String> {
        let dom = self.dom.as_ref().ok_or_else(|| anyhow::anyhow!("No DOM loaded"))?;
        let mut output = String::new();
        self.build_tree_view(&dom.document, 0, &mut output);
        Ok(output)
    }

    fn build_tree_view(&self, handle: &Handle, depth: usize, output: &mut String) {
        let indent = "  ".repeat(depth);
        let info = self.get_node_info(handle);
        match info.node_type.as_str() {
            "Element" => {
                output.push_str(&format!("{}<{}> ({} children)\n", indent, info.tag_name.unwrap_or_default(), info.child_count));
            }
            "Text" => {
                if let Some(text) = info.text_content {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        output.push_str(&format!("{}[Text: \"{}\"]\n", indent, trimmed));
                    }
                }
            }
            _ => {}
        }
        for child in handle.children.borrow().iter() {
            self.build_tree_view(child, depth + 1, output);
        }
    }
}

impl Default for DOMInspector {
    fn default() -> Self {
        Self::new()
    }
}

/// Network request record
#[derive(Debug, Clone)]
pub struct NetworkRequest {
    pub url: String,
    pub method: String,
    pub status_code: Option<u16>,
    pub size: usize,
    pub duration: Duration,
    pub timestamp: Instant,
}

/// Network monitor for tracking requests
pub struct NetworkMonitor {
    requests: Vec<NetworkRequest>,
    total_bytes: usize,
}

impl NetworkMonitor {
    pub fn new() -> Self {
        Self {
            requests: Vec::new(),
            total_bytes: 0,
        }
    }

    pub fn record_request(&mut self, request: NetworkRequest) {
        self.total_bytes += request.size;
        self.requests.push(request);
    }

    pub fn requests(&self) -> &[NetworkRequest] {
        &self.requests
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    pub fn average_duration(&self) -> Duration {
        if self.requests.is_empty() {
            return Duration::from_secs(0);
        }
        let total: Duration = self.requests.iter().map(|r| r.duration).sum();
        total / self.requests.len() as u32
    }

    pub fn clear(&mut self) {
        self.requests.clear();
        self.total_bytes = 0;
    }

    pub fn report(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Network Monitor Report ===\n");
        output.push_str(&format!("Total Requests: {}\n", self.request_count()));
        output.push_str(&format!("Total Bytes: {} bytes\n", self.total_bytes()));
        output.push_str(&format!("Average Duration: {:?}\n", self.average_duration()));
        output
    }
}

impl Default for NetworkMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub parse_time: Duration,
    pub layout_time: Duration,
    pub paint_time: Duration,
    pub total_time: Duration,
}

/// Performance profiler
pub struct PerformanceProfiler {
    metrics: Option<PerformanceMetrics>,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self { metrics: None }
    }

    pub fn record(&mut self, metrics: PerformanceMetrics) {
        self.metrics = Some(metrics);
    }

    pub fn metrics(&self) -> Option<&PerformanceMetrics> {
        self.metrics.as_ref()
    }

    pub fn report(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Performance Profiler Report ===\n");
        if let Some(metrics) = &self.metrics {
            output.push_str(&format!("Parse Time: {:?}\n", metrics.parse_time));
            output.push_str(&format!("Layout Time: {:?}\n", metrics.layout_time));
            output.push_str(&format!("Paint Time: {:?}\n", metrics.paint_time));
            output.push_str(&format!("Total Time: {:?}\n", metrics.total_time));
        } else {
            output.push_str("No metrics recorded yet.\n");
        }
        output
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dom_inspector_creation() {
        let inspector = DOMInspector::new();
        assert!(inspector.dom.is_none());
    }

    #[test]
    fn test_network_monitor_creation() {
        let monitor = NetworkMonitor::new();
        assert_eq!(monitor.request_count(), 0);
    }

    #[test]
    fn test_network_monitor_record() {
        let mut monitor = NetworkMonitor::new();
        monitor.record_request(NetworkRequest {
            url: "https://example.com".to_string(),
            method: "GET".to_string(),
            status_code: Some(200),
            size: 1024,
            duration: Duration::from_millis(100),
            timestamp: Instant::now(),
        });
        assert_eq!(monitor.request_count(), 1);
        assert_eq!(monitor.total_bytes(), 1024);
    }

    #[test]
    fn test_performance_profiler_creation() {
        let profiler = PerformanceProfiler::new();
        assert!(profiler.metrics().is_none());
    }

    #[test]
    fn test_performance_profiler_record() {
        let mut profiler = PerformanceProfiler::new();
        profiler.record(PerformanceMetrics {
            parse_time: Duration::from_millis(10),
            layout_time: Duration::from_millis(20),
            paint_time: Duration::from_millis(15),
            total_time: Duration::from_millis(45),
        });
        assert!(profiler.metrics().is_some());
    }
}
