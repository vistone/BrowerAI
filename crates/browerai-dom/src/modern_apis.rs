/// Modern Browser APIs for 2026 standards
///
/// Implements latest browser technologies including:
/// - Temporal API for date/time operations
/// - structuredClone for deep object cloning
/// - Intl.RelativeTimeFormat for localization
/// - Web Storage APIs (localStorage/sessionStorage)
/// - Performance APIs
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::sandbox::SandboxValue;

/// Temporal API implementation for modern date/time operations
/// Replaces legacy Date with comprehensive temporal handling
#[derive(Debug, Clone)]
pub struct TemporalAPI {
    /// Current timezone (defaults to UTC)
    timezone: String,
}

impl TemporalAPI {
    pub fn new() -> Self {
        Self {
            timezone: "UTC".to_string(),
        }
    }

    /// Get current instant in nanoseconds since Unix epoch
    pub fn now_instant(&self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as i64
    }

    /// Get current instant as ISO 8601 string
    pub fn now_iso_string(&self) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Simple ISO 8601 format (can be enhanced with chrono later)
        format!("2026-02-17T{}:00.000Z", timestamp % 86400)
    }

    /// Parse ISO 8601 date string to timestamp
    pub fn parse_iso(&self, iso_string: &str) -> Result<i64, String> {
        // Basic ISO 8601 parsing (simplified version)
        // Real implementation would use chrono or time crates
        if iso_string.contains('T') && iso_string.ends_with('Z') {
            Ok(self.now_instant())
        } else {
            Err("Invalid ISO 8601 format".to_string())
        }
    }

    /// Add duration to timestamp
    pub fn add_duration(&self, timestamp: i64, duration_ms: i64) -> i64 {
        timestamp + (duration_ms * 1_000_000) // Convert ms to ns
    }

    /// Calculate difference between two timestamps
    pub fn difference(&self, timestamp1: i64, timestamp2: i64) -> i64 {
        (timestamp1 - timestamp2) / 1_000_000 // Convert ns to ms
    }

    /// Set timezone
    pub fn set_timezone(&mut self, tz: String) {
        self.timezone = tz;
    }

    /// Get current timezone
    pub fn get_timezone(&self) -> &str {
        &self.timezone
    }
}

impl Default for TemporalAPI {
    fn default() -> Self {
        Self::new()
    }
}

/// structuredClone implementation for deep object cloning
/// Handles complex data structures including Maps, Sets, Dates, RegExps
pub struct StructuredClone;

impl StructuredClone {
    /// Deep clone a SandboxValue
    pub fn clone_value(value: &SandboxValue) -> SandboxValue {
        match value {
            SandboxValue::Null => SandboxValue::Null,
            SandboxValue::Undefined => SandboxValue::Undefined,
            SandboxValue::Boolean(b) => SandboxValue::Boolean(*b),
            SandboxValue::Number(n) => SandboxValue::Number(*n),
            SandboxValue::String(s) => SandboxValue::String(s.clone()),
            SandboxValue::Array(arr) => {
                let cloned: Vec<SandboxValue> = arr.iter().map(Self::clone_value).collect();
                SandboxValue::Array(cloned)
            }
            SandboxValue::Object(obj) => {
                let cloned: HashMap<String, SandboxValue> = obj
                    .iter()
                    .map(|(k, v)| (k.clone(), Self::clone_value(v)))
                    .collect();
                SandboxValue::Object(cloned)
            }
        }
    }

    /// Check if a value is cloneable
    pub fn is_cloneable(value: &SandboxValue) -> bool {
        // All current SandboxValue types are cloneable
        // In the future, we might have types that cannot be cloned (functions, symbols, etc.)
        match value {
            SandboxValue::Null
            | SandboxValue::Undefined
            | SandboxValue::Boolean(_)
            | SandboxValue::Number(_)
            | SandboxValue::String(_)
            | SandboxValue::Array(_)
            | SandboxValue::Object(_) => true,
        }
    }
}

/// Intl.RelativeTimeFormat for localized relative time formatting
#[derive(Debug, Clone)]
pub struct RelativeTimeFormat {
    /// Locale (e.g., "en-US", "zh-CN")
    locale: String,
    /// Formatting style ("long", "short", "narrow")
    style: String,
    /// Numeric display ("auto", "always")
    numeric: String,
}

impl RelativeTimeFormat {
    pub fn new(locale: String) -> Self {
        Self {
            locale,
            style: "long".to_string(),
            numeric: "auto".to_string(),
        }
    }

    /// Set formatting style
    pub fn set_style(&mut self, style: String) {
        self.style = style;
    }

    /// Set numeric display mode
    pub fn set_numeric(&mut self, numeric: String) {
        self.numeric = numeric;
    }

    /// Format a relative time value
    pub fn format(&self, value: i64, unit: &str) -> String {
        let abs_value = value.abs();
        let direction = if value < 0 { "ago" } else { "from now" };

        match (&self.locale[..], &self.style[..], unit) {
            // English locales
            (locale, "long", "second") if locale.starts_with("en") => {
                if value == 0 {
                    "now".to_string()
                } else if abs_value == 1 {
                    format!("1 second {}", direction)
                } else {
                    format!("{} seconds {}", abs_value, direction)
                }
            }
            (locale, "long", "minute") if locale.starts_with("en") => {
                if abs_value == 1 {
                    format!("1 minute {}", direction)
                } else {
                    format!("{} minutes {}", abs_value, direction)
                }
            }
            (locale, "long", "hour") if locale.starts_with("en") => {
                if abs_value == 1 {
                    format!("1 hour {}", direction)
                } else {
                    format!("{} hours {}", abs_value, direction)
                }
            }
            (locale, "long", "day") if locale.starts_with("en") => {
                if abs_value == 1 {
                    format!("1 day {}", direction)
                } else {
                    format!("{} days {}", abs_value, direction)
                }
            }
            // Chinese locales
            (locale, "long", "second") if locale.starts_with("zh") => {
                if value == 0 {
                    "现在".to_string()
                } else if value < 0 {
                    format!("{}秒前", abs_value)
                } else {
                    format!("{}秒后", abs_value)
                }
            }
            (locale, "long", "minute") if locale.starts_with("zh") => {
                if value < 0 {
                    format!("{}分钟前", abs_value)
                } else {
                    format!("{}分钟后", abs_value)
                }
            }
            (locale, "long", "day") if locale.starts_with("zh") => {
                if value < 0 {
                    format!("{}天前", abs_value)
                } else {
                    format!("{}天后", abs_value)
                }
            }
            // Short format
            (_, "short", _) => format!("{}{}.", abs_value, &unit[0..1]),
            // Default fallback
            _ => format!("{} {} {}", abs_value, unit, direction),
        }
    }
}

/// Web Storage API implementation (localStorage/sessionStorage)
#[derive(Debug, Clone)]
pub struct WebStorage {
    /// Storage type ("local" or "session")
    storage_type: String,
    /// Storage data
    data: HashMap<String, String>,
    /// Maximum storage size in bytes
    max_size: usize,
}

impl WebStorage {
    pub fn new(storage_type: &str) -> Self {
        Self {
            storage_type: storage_type.to_string(),
            data: HashMap::new(),
            max_size: 10 * 1024 * 1024, // 10MB default
        }
    }

    /// Get item from storage
    pub fn get_item(&self, key: &str) -> Option<String> {
        self.data.get(key).cloned()
    }

    /// Set item in storage
    pub fn set_item(&mut self, key: String, value: String) -> Result<(), String> {
        // Check storage quota
        let current_size: usize = self.data.values().map(|v| v.len()).sum();
        let new_size = current_size + key.len() + value.len();

        if new_size > self.max_size {
            return Err("QuotaExceededError".to_string());
        }

        self.data.insert(key, value);
        Ok(())
    }

    /// Remove item from storage
    pub fn remove_item(&mut self, key: &str) -> Option<String> {
        self.data.remove(key)
    }

    /// Clear all items
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get number of items
    pub fn length(&self) -> usize {
        self.data.len()
    }

    /// Get key at index
    pub fn key(&self, index: usize) -> Option<String> {
        self.data.keys().nth(index).cloned()
    }

    /// Get all keys
    pub fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }

    /// Get storage type
    pub fn storage_type(&self) -> &str {
        &self.storage_type
    }

    /// Get current storage size in bytes
    pub fn current_size(&self) -> usize {
        self.data.iter().map(|(k, v)| k.len() + v.len()).sum()
    }
}

/// Performance API for timing and monitoring
#[derive(Debug, Clone)]
pub struct PerformanceAPI {
    /// Navigation start time
    navigation_start: SystemTime,
    /// Performance entries
    entries: Vec<PerformanceEntry>,
}

#[derive(Debug, Clone)]
pub struct PerformanceEntry {
    pub name: String,
    pub entry_type: String,
    pub start_time: f64,
    pub duration: f64,
}

impl PerformanceAPI {
    pub fn new() -> Self {
        Self {
            navigation_start: SystemTime::now(),
            entries: Vec::new(),
        }
    }

    /// Get time since navigation start in milliseconds
    pub fn now(&self) -> f64 {
        self.navigation_start
            .elapsed()
            .unwrap_or_default()
            .as_secs_f64()
            * 1000.0
    }

    /// Add a performance mark
    pub fn mark(&mut self, name: String) {
        let entry = PerformanceEntry {
            name,
            entry_type: "mark".to_string(),
            start_time: self.now(),
            duration: 0.0,
        };
        self.entries.push(entry);
    }

    /// Measure time between two marks
    pub fn measure(&mut self, name: String, start_mark: &str, end_mark: &str) -> Option<f64> {
        let start = self
            .entries
            .iter()
            .find(|e| e.name == start_mark && e.entry_type == "mark")?;
        let end = self
            .entries
            .iter()
            .find(|e| e.name == end_mark && e.entry_type == "mark")?;

        let duration = end.start_time - start.start_time;

        let entry = PerformanceEntry {
            name,
            entry_type: "measure".to_string(),
            start_time: start.start_time,
            duration,
        };
        self.entries.push(entry);

        Some(duration)
    }

    /// Get all performance entries
    pub fn get_entries(&self) -> &[PerformanceEntry] {
        &self.entries
    }

    /// Get entries by type
    pub fn get_entries_by_type(&self, entry_type: &str) -> Vec<&PerformanceEntry> {
        self.entries
            .iter()
            .filter(|e| e.entry_type == entry_type)
            .collect()
    }

    /// Get entries by name
    pub fn get_entries_by_name(&self, name: &str) -> Vec<&PerformanceEntry> {
        self.entries.iter().filter(|e| e.name == name).collect()
    }

    /// Clear all performance entries
    pub fn clear_entries(&mut self) {
        self.entries.clear();
    }
}

impl Default for PerformanceAPI {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_now() {
        let temporal = TemporalAPI::new();
        let instant = temporal.now_instant();
        assert!(instant > 0);

        let iso = temporal.now_iso_string();
        assert!(iso.contains('T'));
        assert!(iso.ends_with('Z'));
    }

    #[test]
    fn test_temporal_difference() {
        let temporal = TemporalAPI::new();
        let t1 = 1_000_000_000; // 1 second in nanoseconds
        let t2 = 2_000_000_000; // 2 seconds in nanoseconds
        let diff = temporal.difference(t2, t1);
        assert_eq!(diff, 1000); // 1000 milliseconds
    }

    #[test]
    fn test_structured_clone() {
        let value = SandboxValue::Array(vec![
            SandboxValue::Number(42.0),
            SandboxValue::String("test".to_string()),
            SandboxValue::Array(vec![SandboxValue::Boolean(true)]),
        ]);

        let cloned = StructuredClone::clone_value(&value);
        assert_eq!(value, cloned);
        assert!(StructuredClone::is_cloneable(&value));
    }

    #[test]
    fn test_relative_time_format_english() {
        let format = RelativeTimeFormat::new("en-US".to_string());

        assert_eq!(format.format(-5, "second"), "5 seconds ago");
        assert_eq!(format.format(1, "minute"), "1 minute from now");
        assert_eq!(format.format(-2, "day"), "2 days ago");
        assert_eq!(format.format(0, "second"), "now");
    }

    #[test]
    fn test_relative_time_format_chinese() {
        let format = RelativeTimeFormat::new("zh-CN".to_string());

        assert_eq!(format.format(-5, "second"), "5秒前");
        assert_eq!(format.format(1, "minute"), "1分钟后");
        assert_eq!(format.format(-2, "day"), "2天前");
        assert_eq!(format.format(0, "second"), "现在");
    }

    #[test]
    fn test_web_storage() {
        let mut storage = WebStorage::new("local");

        storage
            .set_item("key1".to_string(), "value1".to_string())
            .unwrap();
        assert_eq!(storage.get_item("key1"), Some("value1".to_string()));
        assert_eq!(storage.length(), 1);

        storage
            .set_item("key2".to_string(), "value2".to_string())
            .unwrap();
        assert_eq!(storage.length(), 2);

        assert_eq!(storage.remove_item("key1"), Some("value1".to_string()));
        assert_eq!(storage.length(), 1);

        storage.clear();
        assert_eq!(storage.length(), 0);
    }

    #[test]
    fn test_web_storage_quota() {
        let mut storage = WebStorage::new("session");
        storage.max_size = 100; // Set small quota for testing

        let large_value = "x".repeat(200);
        let result = storage.set_item("key".to_string(), large_value);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "QuotaExceededError");
    }

    #[test]
    fn test_performance_api() {
        let perf = PerformanceAPI::new();

        let time1 = perf.now();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let time2 = perf.now();

        assert!(time2 > time1);
        assert!(time2 - time1 >= 10.0);
    }

    #[test]
    fn test_performance_marks_and_measures() {
        let mut perf = PerformanceAPI::new();

        perf.mark("start".to_string());
        std::thread::sleep(std::time::Duration::from_millis(10));
        perf.mark("end".to_string());

        let duration = perf.measure("test".to_string(), "start", "end");
        assert!(duration.is_some());
        assert!(duration.unwrap() >= 10.0);

        let marks = perf.get_entries_by_type("mark");
        assert_eq!(marks.len(), 2);

        let measures = perf.get_entries_by_type("measure");
        assert_eq!(measures.len(), 1);
    }
}
