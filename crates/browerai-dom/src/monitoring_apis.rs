/// Modern Performance and Monitoring APIs for 2026 Standards
///
/// Implements cutting-edge monitoring and performance APIs including:
/// - Intersection Observer API
/// - Mutation Observer API
/// - Long Animation Frames (LoAF) API
/// - Navigation Timing API v2
/// - Resource Timing API
use std::time::SystemTime;

/// Intersection Observer for lazy-loading and visibility tracking
#[derive(Debug, Clone)]
pub struct IntersectionObserver {
    /// Callback ID for managing observers
    pub id: String,
    /// Root element (None = viewport)
    pub root: Option<String>,
    /// Root margin (e.g., "10px 20px 30px 40px")
    pub root_margin: String,
    /// Intersection threshold (0.0 to 1.0)
    pub threshold: Vec<f32>,
    /// Observed entries
    pub entries: Vec<IntersectionEntry>,
}

#[derive(Debug, Clone)]
pub struct IntersectionEntry {
    /// Target element ID or selector
    pub target: String,
    /// Whether the target is intersecting
    pub is_intersecting: bool,
    /// Intersection ratio (0.0 to 1.0)
    pub intersection_ratio: f32,
    /// Bounding rect of target
    pub bounding_rect: DOMRect,
    /// Bounding rect of root
    pub root_bounds: Option<DOMRect>,
    /// Time of observation (milliseconds)
    pub time: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DOMRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl DOMRect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            top: y,
            right: x + width,
            bottom: y + height,
            left: x,
        }
    }

    /// Check if two rects intersect
    pub fn intersects(&self, other: &DOMRect) -> bool {
        !(self.right < other.left
            || self.left > other.right
            || self.bottom < other.top
            || self.top > other.bottom)
    }

    /// Calculate intersection ratio
    pub fn intersection_ratio(&self, other: &DOMRect) -> f32 {
        if !self.intersects(other) {
            return 0.0;
        }

        let intersection_left = self.left.max(other.left);
        let intersection_top = self.top.max(other.top);
        let intersection_right = self.right.min(other.right);
        let intersection_bottom = self.bottom.min(other.bottom);

        let intersection_area =
            (intersection_right - intersection_left) * (intersection_bottom - intersection_top);
        let target_area = self.width * self.height;

        if target_area == 0.0 {
            0.0
        } else {
            intersection_area / target_area
        }
    }
}

impl IntersectionObserver {
    /// Create a new Intersection Observer
    pub fn new(id: String) -> Self {
        Self {
            id,
            root: None,
            root_margin: "0px".to_string(),
            threshold: vec![0.0],
            entries: Vec::new(),
        }
    }

    /// Set root element
    pub fn set_root(&mut self, root: Option<String>) {
        self.root = root;
    }

    /// Set root margin
    pub fn set_root_margin(&mut self, margin: String) {
        self.root_margin = margin;
    }

    /// Set thresholds
    pub fn set_threshold(&mut self, threshold: Vec<f32>) {
        self.threshold = threshold;
    }

    /// Observe an element
    pub fn observe(&mut self, target: String, target_rect: DOMRect, viewport_rect: DOMRect) {
        let is_intersecting = target_rect.intersects(&viewport_rect);
        let intersection_ratio = if is_intersecting {
            target_rect.intersection_ratio(&viewport_rect)
        } else {
            0.0
        };

        let entry = IntersectionEntry {
            target,
            is_intersecting,
            intersection_ratio,
            bounding_rect: target_rect,
            root_bounds: Some(viewport_rect),
            time: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as f64,
        };

        self.entries.push(entry);
    }

    /// Unobserve an element
    pub fn unobserve(&mut self, target: &str) {
        self.entries.retain(|e| e.target != target);
    }

    /// Get all observed entries
    pub fn take_records(&mut self) -> Vec<IntersectionEntry> {
        std::mem::take(&mut self.entries)
    }

    /// Disconnect observer
    pub fn disconnect(&mut self) {
        self.entries.clear();
    }
}

/// Mutation Observer for DOM change observation
#[derive(Debug, Clone)]
pub struct MutationObserver {
    /// Observer ID
    pub id: String,
    /// Configuration options
    pub config: MutationObserverConfig,
    /// Recorded mutations
    pub mutations: Vec<MutationRecord>,
}

#[derive(Debug, Clone)]
pub struct MutationObserverConfig {
    /// Observe attribute changes
    pub attributes: bool,
    /// Observe attribute old value
    pub attribute_old_value: bool,
    /// Filter specific attributes
    pub attribute_filter: Vec<String>,
    /// Observe character data changes
    pub character_data: bool,
    /// Observe character data old value
    pub character_data_old_value: bool,
    /// Observe child list changes
    pub child_list: bool,
    /// Observe subtree changes
    pub subtree: bool,
}

impl Default for MutationObserverConfig {
    fn default() -> Self {
        Self {
            attributes: false,
            attribute_old_value: false,
            attribute_filter: Vec::new(),
            character_data: false,
            character_data_old_value: false,
            child_list: true,
            subtree: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MutationRecord {
    /// Type of mutation
    pub mutation_type: MutationType,
    /// Target node
    pub target: String,
    /// Added nodes
    pub added_nodes: Vec<String>,
    /// Removed nodes
    pub removed_nodes: Vec<String>,
    /// Previous sibling
    pub previous_sibling: Option<String>,
    /// Next sibling
    pub next_sibling: Option<String>,
    /// Attribute name (for attribute mutations)
    pub attribute_name: Option<String>,
    /// Old value (for attribute or characterData mutations)
    pub old_value: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MutationType {
    Attributes,
    CharacterData,
    ChildList,
}

impl MutationObserver {
    /// Create a new Mutation Observer
    pub fn new(id: String, config: MutationObserverConfig) -> Self {
        Self {
            id,
            config,
            mutations: Vec::new(),
        }
    }

    /// Observe a target node
    pub fn observe(&mut self, _target: String) {
        // Observer is now active (in real implementation, would register with DOM)
    }

    /// Record a mutation
    pub fn record_mutation(&mut self, record: MutationRecord) {
        self.mutations.push(record);
    }

    /// Take recorded mutations
    pub fn take_records(&mut self) -> Vec<MutationRecord> {
        std::mem::take(&mut self.mutations)
    }

    /// Disconnect observer
    pub fn disconnect(&mut self) {
        self.mutations.clear();
    }
}

/// Long Animation Frames (LoAF) API for performance analysis
#[derive(Debug, Clone)]
pub struct LongAnimationFrame {
    /// Frame duration in milliseconds
    pub duration: f64,
    /// Start time
    pub start_time: f64,
    /// End time
    pub end_time: f64,
    /// Blocking duration
    pub blocking_duration: f64,
    /// Rendering duration
    pub render_duration: f64,
    /// Script entries
    pub scripts: Vec<ScriptEntry>,
}

#[derive(Debug, Clone)]
pub struct ScriptEntry {
    /// Script name or URL
    pub name: String,
    /// Entry type
    pub entry_type: String,
    /// Start time
    pub start_time: f64,
    /// Duration
    pub duration: f64,
    /// Invoker
    pub invoker: Option<String>,
}

impl LongAnimationFrame {
    /// Create a new long animation frame record
    pub fn new(start_time: f64, duration: f64) -> Self {
        Self {
            duration,
            start_time,
            end_time: start_time + duration,
            blocking_duration: 0.0,
            render_duration: 0.0,
            scripts: Vec::new(),
        }
    }

    /// Check if this is a long frame (> 50ms)
    pub fn is_long(&self) -> bool {
        self.duration > 50.0
    }

    /// Add a script entry
    pub fn add_script(&mut self, script: ScriptEntry) {
        self.scripts.push(script);
    }

    /// Calculate total script time
    pub fn total_script_time(&self) -> f64 {
        self.scripts.iter().map(|s| s.duration).sum()
    }
}

/// Navigation Timing API v2
#[derive(Debug, Clone)]
pub struct NavigationTiming {
    /// Navigation type
    pub navigation_type: NavigationType,
    /// Timestamps
    pub timestamps: NavigationTimestamps,
    /// Transfer size
    pub transfer_size: u64,
    /// Encoded body size
    pub encoded_body_size: u64,
    /// Decoded body size
    pub decoded_body_size: u64,
    /// Server timing
    pub server_timing: Vec<ServerTiming>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NavigationType {
    Navigate,
    Reload,
    BackForward,
    Prerender,
}

#[derive(Debug, Clone)]
pub struct NavigationTimestamps {
    /// Navigation start time
    pub navigation_start: f64,
    /// Unload event start
    pub unload_event_start: f64,
    /// Unload event end
    pub unload_event_end: f64,
    /// Redirect start
    pub redirect_start: f64,
    /// Redirect end
    pub redirect_end: f64,
    /// Fetch start
    pub fetch_start: f64,
    /// Domain lookup start
    pub domain_lookup_start: f64,
    /// Domain lookup end
    pub domain_lookup_end: f64,
    /// Connect start
    pub connect_start: f64,
    /// Connect end
    pub connect_end: f64,
    /// Request start
    pub request_start: f64,
    /// Response start
    pub response_start: f64,
    /// Response end
    pub response_end: f64,
    /// DOM interactive
    pub dom_interactive: f64,
    /// DOM content loaded event start
    pub dom_content_loaded_event_start: f64,
    /// DOM content loaded event end
    pub dom_content_loaded_event_end: f64,
    /// DOM complete
    pub dom_complete: f64,
    /// Load event start
    pub load_event_start: f64,
    /// Load event end
    pub load_event_end: f64,
}

impl Default for NavigationTimestamps {
    fn default() -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as f64;

        Self {
            navigation_start: now,
            unload_event_start: 0.0,
            unload_event_end: 0.0,
            redirect_start: 0.0,
            redirect_end: 0.0,
            fetch_start: now,
            domain_lookup_start: now,
            domain_lookup_end: now,
            connect_start: now,
            connect_end: now,
            request_start: now,
            response_start: now,
            response_end: now,
            dom_interactive: 0.0,
            dom_content_loaded_event_start: 0.0,
            dom_content_loaded_event_end: 0.0,
            dom_complete: 0.0,
            load_event_start: 0.0,
            load_event_end: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServerTiming {
    /// Metric name
    pub name: String,
    /// Duration in milliseconds
    pub duration: f64,
    /// Description
    pub description: String,
}

impl NavigationTiming {
    /// Create a new navigation timing record
    pub fn new(navigation_type: NavigationType) -> Self {
        Self {
            navigation_type,
            timestamps: NavigationTimestamps::default(),
            transfer_size: 0,
            encoded_body_size: 0,
            decoded_body_size: 0,
            server_timing: Vec::new(),
        }
    }

    /// Calculate time to first byte (TTFB)
    pub fn time_to_first_byte(&self) -> f64 {
        self.timestamps.response_start - self.timestamps.navigation_start
    }

    /// Calculate DOM content loaded time
    pub fn dom_content_loaded_time(&self) -> f64 {
        self.timestamps.dom_content_loaded_event_end - self.timestamps.navigation_start
    }

    /// Calculate page load time
    pub fn load_time(&self) -> f64 {
        self.timestamps.load_event_end - self.timestamps.navigation_start
    }

    /// Calculate DNS lookup time
    pub fn dns_lookup_time(&self) -> f64 {
        self.timestamps.domain_lookup_end - self.timestamps.domain_lookup_start
    }

    /// Calculate TCP connection time
    pub fn connection_time(&self) -> f64 {
        self.timestamps.connect_end - self.timestamps.connect_start
    }

    /// Add server timing
    pub fn add_server_timing(&mut self, timing: ServerTiming) {
        self.server_timing.push(timing);
    }
}

/// Resource Timing API for individual resource performance
#[derive(Debug, Clone)]
pub struct ResourceTiming {
    /// Resource name (URL)
    pub name: String,
    /// Entry type
    pub entry_type: String,
    /// Start time
    pub start_time: f64,
    /// Duration
    pub duration: f64,
    /// Initiator type
    pub initiator_type: String,
    /// Transfer size
    pub transfer_size: u64,
    /// Encoded body size
    pub encoded_body_size: u64,
    /// Decoded body size
    pub decoded_body_size: u64,
    /// Server timing
    pub server_timing: Vec<ServerTiming>,
}

impl ResourceTiming {
    /// Create a new resource timing entry
    pub fn new(name: String, initiator_type: String) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as f64;

        Self {
            name,
            entry_type: "resource".to_string(),
            start_time: now,
            duration: 0.0,
            initiator_type,
            transfer_size: 0,
            encoded_body_size: 0,
            decoded_body_size: 0,
            server_timing: Vec::new(),
        }
    }

    /// Complete the resource timing
    pub fn complete(&mut self, duration: f64, transfer_size: u64) {
        self.duration = duration;
        self.transfer_size = transfer_size;
    }

    /// Check if resource was cached
    pub fn is_cached(&self) -> bool {
        self.transfer_size == 0 && self.encoded_body_size > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dom_rect_intersects() {
        let rect1 = DOMRect::new(0.0, 0.0, 100.0, 100.0);
        let rect2 = DOMRect::new(50.0, 50.0, 100.0, 100.0);
        let rect3 = DOMRect::new(200.0, 200.0, 100.0, 100.0);

        assert!(rect1.intersects(&rect2));
        assert!(!rect1.intersects(&rect3));
    }

    #[test]
    fn test_dom_rect_intersection_ratio() {
        let rect1 = DOMRect::new(0.0, 0.0, 100.0, 100.0);
        let rect2 = DOMRect::new(50.0, 50.0, 100.0, 100.0);

        let ratio = rect1.intersection_ratio(&rect2);
        assert!(ratio > 0.0 && ratio <= 1.0);
        assert!((ratio - 0.25).abs() < 0.01); // 25% intersection
    }

    #[test]
    fn test_intersection_observer() {
        let mut observer = IntersectionObserver::new("test-observer".to_string());
        
        let target_rect = DOMRect::new(50.0, 50.0, 100.0, 100.0);
        let viewport_rect = DOMRect::new(0.0, 0.0, 200.0, 200.0);

        observer.observe("element-1".to_string(), target_rect, viewport_rect);

        let entries = observer.take_records();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].is_intersecting);
    }

    #[test]
    fn test_mutation_observer() {
        let config = MutationObserverConfig {
            child_list: true,
            subtree: true,
            ..Default::default()
        };

        let mut observer = MutationObserver::new("test-mutation".to_string(), config);

        let record = MutationRecord {
            mutation_type: MutationType::ChildList,
            target: "parent".to_string(),
            added_nodes: vec!["child1".to_string()],
            removed_nodes: vec![],
            previous_sibling: None,
            next_sibling: None,
            attribute_name: None,
            old_value: None,
        };

        observer.record_mutation(record);

        let mutations = observer.take_records();
        assert_eq!(mutations.len(), 1);
        assert_eq!(mutations[0].mutation_type, MutationType::ChildList);
    }

    #[test]
    fn test_long_animation_frame() {
        let mut frame = LongAnimationFrame::new(1000.0, 75.0);
        
        assert!(frame.is_long()); // > 50ms
        assert_eq!(frame.end_time, 1075.0);

        let script = ScriptEntry {
            name: "script.js".to_string(),
            entry_type: "script".to_string(),
            start_time: 1000.0,
            duration: 30.0,
            invoker: Some("setTimeout".to_string()),
        };

        frame.add_script(script);
        assert_eq!(frame.total_script_time(), 30.0);
    }

    #[test]
    fn test_navigation_timing() {
        let mut nav = NavigationTiming::new(NavigationType::Navigate);
        
        nav.timestamps.response_start = nav.timestamps.navigation_start + 100.0;
        nav.timestamps.dom_content_loaded_event_end = nav.timestamps.navigation_start + 500.0;
        nav.timestamps.load_event_end = nav.timestamps.navigation_start + 1000.0;

        assert_eq!(nav.time_to_first_byte(), 100.0);
        assert_eq!(nav.dom_content_loaded_time(), 500.0);
        assert_eq!(nav.load_time(), 1000.0);

        nav.add_server_timing(ServerTiming {
            name: "cache".to_string(),
            duration: 10.0,
            description: "Cache lookup".to_string(),
        });

        assert_eq!(nav.server_timing.len(), 1);
    }

    #[test]
    fn test_resource_timing() {
        let mut resource = ResourceTiming::new(
            "https://example.com/script.js".to_string(),
            "script".to_string(),
        );

        resource.complete(150.0, 50000);
        assert_eq!(resource.duration, 150.0);
        assert_eq!(resource.transfer_size, 50000);
        assert!(!resource.is_cached());

        // Cached resource
        let mut cached = ResourceTiming::new(
            "https://example.com/cached.css".to_string(),
            "link".to_string(),
        );
        cached.encoded_body_size = 10000;
        cached.transfer_size = 0;
        assert!(cached.is_cached());
    }
}
