/// A/B testing framework for comparing model performance
///
/// Enables controlled experiments to compare different models or configurations
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Test variant in an A/B test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestVariant {
    /// Variant identifier
    pub id: String,
    /// Variant name
    pub name: String,
    /// Model or configuration to use
    pub config: HashMap<String, String>,
    /// Traffic allocation (0.0 to 1.0)
    pub traffic_percentage: f32,
    /// Number of samples seen
    pub sample_count: usize,
    /// Performance metrics
    pub metrics: HashMap<String, Vec<f32>>,
}

impl TestVariant {
    pub fn new(id: impl Into<String>, name: impl Into<String>, traffic_percentage: f32) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            config: HashMap::new(),
            traffic_percentage: traffic_percentage.clamp(0.0, 1.0),
            sample_count: 0,
            metrics: HashMap::new(),
        }
    }

    /// Add a configuration parameter
    pub fn with_config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.insert(key.into(), value.into());
        self
    }

    /// Record a metric value
    pub fn record_metric(&mut self, metric_name: impl Into<String>, value: f32) {
        self.metrics
            .entry(metric_name.into())
            .or_insert_with(Vec::new)
            .push(value);
        self.sample_count += 1;
    }

    /// Get average for a metric
    pub fn get_metric_average(&self, metric_name: &str) -> Option<f32> {
        let values = self.metrics.get(metric_name)?;
        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f32>() / values.len() as f32)
        }
    }
}

/// A/B test comparing multiple variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTest {
    /// Test identifier
    pub id: String,
    /// Test name
    pub name: String,
    /// Description of what is being tested
    pub description: String,
    /// Test variants
    pub variants: Vec<TestVariant>,
    /// Test start time
    pub start_time: u64,
    /// Test end time (if completed)
    pub end_time: Option<u64>,
    /// Whether test is active
    pub is_active: bool,
}

impl ABTest {
    /// Create a new A/B test
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs();

        Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            variants: Vec::new(),
            start_time,
            end_time: None,
            is_active: true,
        }
    }

    /// Add a variant to the test
    pub fn add_variant(&mut self, variant: TestVariant) {
        self.variants.push(variant);
    }

    /// Select a variant based on traffic allocation (simple random selection)
    pub fn select_variant(&self) -> Option<&TestVariant> {
        if self.variants.is_empty() || !self.is_active {
            return None;
        }

        // Simple round-robin for testing
        // In production, this would use proper weighted random selection
        self.variants.first()
    }

    /// Get a variant by ID
    pub fn get_variant_mut(&mut self, variant_id: &str) -> Option<&mut TestVariant> {
        self.variants.iter_mut().find(|v| v.id == variant_id)
    }

    /// Complete the test
    pub fn complete(&mut self) {
        self.is_active = false;
        self.end_time = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_else(|_| std::time::Duration::from_secs(0))
                .as_secs(),
        );
    }

    /// Get test results
    pub fn get_results(&self) -> HashMap<String, HashMap<String, f32>> {
        let mut results = HashMap::new();

        for variant in &self.variants {
            let mut variant_results = HashMap::new();
            for (metric_name, values) in &variant.metrics {
                if !values.is_empty() {
                    let avg = values.iter().sum::<f32>() / values.len() as f32;
                    variant_results.insert(metric_name.clone(), avg);
                }
            }
            results.insert(variant.id.clone(), variant_results);
        }

        results
    }

    /// Get winner based on a specific metric (higher is better)
    pub fn get_winner(&self, metric_name: &str) -> Option<&TestVariant> {
        self.variants
            .iter()
            .filter(|v| v.sample_count > 0)
            .max_by(|a, b| {
                let a_val = a.get_metric_average(metric_name).unwrap_or(0.0);
                let b_val = b.get_metric_average(metric_name).unwrap_or(0.0);
                // Use total_cmp to handle NaN values safely
                a_val.total_cmp(&b_val)
            })
    }
}

/// Manager for multiple A/B tests
pub struct ABTestManager {
    /// Active tests
    tests: HashMap<String, ABTest>,
}

impl ABTestManager {
    /// Create a new A/B test manager
    pub fn new() -> Self {
        Self {
            tests: HashMap::new(),
        }
    }

    /// Register a new test
    pub fn register_test(&mut self, test: ABTest) {
        self.tests.insert(test.id.clone(), test);
    }

    /// Get a test by ID
    pub fn get_test(&self, test_id: &str) -> Option<&ABTest> {
        self.tests.get(test_id)
    }

    /// Get a mutable test by ID
    pub fn get_test_mut(&mut self, test_id: &str) -> Option<&mut ABTest> {
        self.tests.get_mut(test_id)
    }

    /// List all active tests
    pub fn list_active_tests(&self) -> Vec<&ABTest> {
        self.tests.values().filter(|t| t.is_active).collect()
    }

    /// Complete a test
    pub fn complete_test(&mut self, test_id: &str) -> Result<(), String> {
        let test = self
            .tests
            .get_mut(test_id)
            .ok_or_else(|| format!("Test not found: {}", test_id))?;

        test.complete();
        Ok(())
    }

    /// Get count of tests
    pub fn test_count(&self) -> usize {
        self.tests.len()
    }

    /// Get count of active tests
    pub fn active_test_count(&self) -> usize {
        self.tests.values().filter(|t| t.is_active).count()
    }
}

impl Default for ABTestManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_creation() {
        let variant = TestVariant::new("var_a", "Variant A", 0.5);
        assert_eq!(variant.id, "var_a");
        assert_eq!(variant.name, "Variant A");
        assert_eq!(variant.traffic_percentage, 0.5);
    }

    #[test]
    fn test_variant_with_config() {
        let variant = TestVariant::new("var_a", "Variant A", 0.5)
            .with_config("model_id", "model_v1")
            .with_config("threshold", "0.8");

        assert_eq!(
            variant.config.get("model_id"),
            Some(&"model_v1".to_string())
        );
    }

    #[test]
    fn test_variant_record_metric() {
        let mut variant = TestVariant::new("var_a", "Variant A", 0.5);
        variant.record_metric("accuracy", 0.95);
        variant.record_metric("accuracy", 0.97);

        assert_eq!(variant.sample_count, 2);
        let avg = variant.get_metric_average("accuracy").unwrap();
        assert!((avg - 0.96).abs() < 0.001);
    }

    #[test]
    fn test_ab_test_creation() {
        let test = ABTest::new("test_1", "Parser Test", "Testing HTML parsers");
        assert_eq!(test.id, "test_1");
        assert_eq!(test.name, "Parser Test");
        assert!(test.is_active);
    }

    #[test]
    fn test_ab_test_add_variant() {
        let mut test = ABTest::new("test_1", "Test", "Description");
        test.add_variant(TestVariant::new("var_a", "A", 0.5));
        test.add_variant(TestVariant::new("var_b", "B", 0.5));

        assert_eq!(test.variants.len(), 2);
    }

    #[test]
    fn test_ab_test_select_variant() {
        let mut test = ABTest::new("test_1", "Test", "Description");
        test.add_variant(TestVariant::new("var_a", "A", 0.5));

        let selected = test.select_variant();
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().id, "var_a");
    }

    #[test]
    fn test_ab_test_complete() {
        let mut test = ABTest::new("test_1", "Test", "Description");
        test.complete();

        assert!(!test.is_active);
        assert!(test.end_time.is_some());
    }

    #[test]
    fn test_ab_test_get_results() {
        let mut test = ABTest::new("test_1", "Test", "Description");

        let mut var_a = TestVariant::new("var_a", "A", 0.5);
        var_a.record_metric("accuracy", 0.95);
        var_a.record_metric("accuracy", 0.97);

        test.add_variant(var_a);

        let results = test.get_results();
        assert!(results.contains_key("var_a"));
        let accuracy = results["var_a"]["accuracy"];
        assert!((accuracy - 0.96).abs() < 0.001);
    }

    #[test]
    fn test_ab_test_get_winner() {
        let mut test = ABTest::new("test_1", "Test", "Description");

        let mut var_a = TestVariant::new("var_a", "A", 0.5);
        var_a.record_metric("accuracy", 0.90);

        let mut var_b = TestVariant::new("var_b", "B", 0.5);
        var_b.record_metric("accuracy", 0.95);

        test.add_variant(var_a);
        test.add_variant(var_b);

        let winner = test.get_winner("accuracy");
        assert!(winner.is_some());
        assert_eq!(winner.unwrap().id, "var_b");
    }

    #[test]
    fn test_ab_test_manager_register() {
        let mut manager = ABTestManager::new();
        let test = ABTest::new("test_1", "Test", "Description");

        manager.register_test(test);
        assert_eq!(manager.test_count(), 1);
    }

    #[test]
    fn test_ab_test_manager_get_test() {
        let mut manager = ABTestManager::new();
        let test = ABTest::new("test_1", "Test", "Description");

        manager.register_test(test);

        let retrieved = manager.get_test("test_1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Test");
    }

    #[test]
    fn test_ab_test_manager_complete() {
        let mut manager = ABTestManager::new();
        let test = ABTest::new("test_1", "Test", "Description");

        manager.register_test(test);
        manager.complete_test("test_1").unwrap();

        let test = manager.get_test("test_1").unwrap();
        assert!(!test.is_active);
    }

    #[test]
    fn test_ab_test_manager_active_count() {
        let mut manager = ABTestManager::new();

        manager.register_test(ABTest::new("test_1", "Test 1", "Desc"));
        manager.register_test(ABTest::new("test_2", "Test 2", "Desc"));

        assert_eq!(manager.active_test_count(), 2);

        manager.complete_test("test_1").unwrap();
        assert_eq!(manager.active_test_count(), 1);
    }
}
