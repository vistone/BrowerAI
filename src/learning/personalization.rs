/// User personalization system for customized browsing experience
/// 
/// Learns user preferences and adapts rendering/parsing strategies

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// User preference categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PreferenceCategory {
    /// Performance vs quality tradeoff
    Performance,
    /// Rendering style preferences
    Rendering,
    /// Content preferences
    Content,
    /// Privacy settings
    Privacy,
    /// Accessibility settings
    Accessibility,
    /// Custom preference
    Custom(String),
}

/// User preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// User identifier
    pub user_id: String,
    /// Preference values by category
    pub preferences: HashMap<PreferenceCategory, HashMap<String, f32>>,
    /// Learned patterns from user behavior
    pub learned_patterns: HashMap<String, Vec<String>>,
    /// Privacy-preserving: only store aggregated data
    pub privacy_mode: bool,
}

impl UserPreferences {
    /// Create new user preferences
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            preferences: HashMap::new(),
            learned_patterns: HashMap::new(),
            privacy_mode: true,
        }
    }

    /// Set a preference value
    pub fn set_preference(
        &mut self,
        category: PreferenceCategory,
        key: impl Into<String>,
        value: f32,
    ) {
        self.preferences
            .entry(category)
            .or_insert_with(HashMap::new)
            .insert(key.into(), value);
    }

    /// Get a preference value
    pub fn get_preference(&self, category: &PreferenceCategory, key: &str) -> Option<f32> {
        self.preferences.get(category)?.get(key).copied()
    }

    /// Learn a pattern from user behavior
    pub fn learn_pattern(&mut self, pattern_type: impl Into<String>, value: impl Into<String>) {
        if !self.privacy_mode {
            self.learned_patterns
                .entry(pattern_type.into())
                .or_insert_with(Vec::new)
                .push(value.into());
        }
    }

    /// Get learned patterns
    pub fn get_patterns(&self, pattern_type: &str) -> Option<&[String]> {
        self.learned_patterns.get(pattern_type).map(|v| v.as_slice())
    }

    /// Enable/disable privacy mode
    pub fn set_privacy_mode(&mut self, enabled: bool) {
        self.privacy_mode = enabled;
        if enabled {
            // Clear learned patterns when enabling privacy mode
            self.learned_patterns.clear();
        }
    }

    /// Export preferences to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Import preferences from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Personalization engine that adapts system behavior
pub struct PersonalizationEngine {
    /// User preferences by user ID
    users: HashMap<String, UserPreferences>,
}

impl PersonalizationEngine {
    /// Create a new personalization engine
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
        }
    }

    /// Register a new user
    pub fn register_user(&mut self, user_id: impl Into<String>) -> &mut UserPreferences {
        let user_id_str = user_id.into();
        self.users
            .entry(user_id_str.clone())
            .or_insert_with(|| UserPreferences::new(&user_id_str))
    }

    /// Get user preferences
    pub fn get_user(&self, user_id: &str) -> Option<&UserPreferences> {
        self.users.get(user_id)
    }

    /// Get mutable user preferences
    pub fn get_user_mut(&mut self, user_id: &str) -> Option<&mut UserPreferences> {
        self.users.get_mut(user_id)
    }

    /// Get or create user preferences
    pub fn get_or_create_user(&mut self, user_id: impl Into<String>) -> &mut UserPreferences {
        let user_id_str = user_id.into();
        if !self.users.contains_key(&user_id_str) {
            self.register_user(&user_id_str);
        }
        self.users.get_mut(&user_id_str).expect("User should exist after registration")
    }

    /// Apply personalization to rendering config
    pub fn personalize_rendering(&self, user_id: &str, base_quality: f32) -> f32 {
        if let Some(user) = self.get_user(user_id) {
            // Check if user prefers quality over speed
            let quality_pref = user
                .get_preference(&PreferenceCategory::Performance, "quality_weight")
                .unwrap_or(0.5);
            
            // Adjust quality based on preference (0.0 = speed, 1.0 = quality)
            base_quality * (0.5 + quality_pref * 0.5)
        } else {
            base_quality
        }
    }

    /// Apply personalization to cache strategy
    pub fn personalize_cache_strategy(&self, user_id: &str) -> String {
        if let Some(user) = self.get_user(user_id) {
            // Check cache preference
            let cache_aggressiveness = user
                .get_preference(&PreferenceCategory::Performance, "cache_aggressiveness")
                .unwrap_or(0.5);
            
            if cache_aggressiveness > 0.7 {
                "aggressive".to_string()
            } else if cache_aggressiveness < 0.3 {
                "minimal".to_string()
            } else {
                "balanced".to_string()
            }
        } else {
            "balanced".to_string()
        }
    }

    /// Get personalization recommendations
    pub fn get_recommendations(&self, user_id: &str) -> Vec<String> {
        let mut recommendations = Vec::new();

        if let Some(user) = self.get_user(user_id) {
            // Analyze preferences and suggest optimizations
            if let Some(quality_pref) = user.get_preference(&PreferenceCategory::Performance, "quality_weight") {
                if quality_pref > 0.8 {
                    recommendations.push("Consider enabling high-quality rendering mode".to_string());
                } else if quality_pref < 0.2 {
                    recommendations.push("Consider enabling fast rendering mode".to_string());
                }
            }

            if user.learned_patterns.len() > 10 {
                recommendations.push("Sufficient browsing data collected for advanced personalization".to_string());
            }

            if user.privacy_mode {
                recommendations.push("Privacy mode is enabled - some personalization features are limited".to_string());
            }
        }

        recommendations
    }

    /// Get user count
    pub fn user_count(&self) -> usize {
        self.users.len()
    }

    /// Export user preferences
    pub fn export_user(&self, user_id: &str) -> Option<String> {
        self.get_user(user_id)?.to_json().ok()
    }

    /// Import user preferences
    pub fn import_user(&mut self, json: &str) -> Result<(), String> {
        let prefs = UserPreferences::from_json(json)
            .map_err(|e| format!("Failed to parse JSON: {}", e))?;
        let user_id = prefs.user_id.clone();
        self.users.insert(user_id, prefs);
        Ok(())
    }
}

impl Default for PersonalizationEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_preferences_creation() {
        let prefs = UserPreferences::new("user123");
        assert_eq!(prefs.user_id, "user123");
        assert!(prefs.privacy_mode);
    }

    #[test]
    fn test_user_preferences_set_get() {
        let mut prefs = UserPreferences::new("user123");
        prefs.set_preference(PreferenceCategory::Performance, "quality_weight", 0.8);

        let value = prefs.get_preference(&PreferenceCategory::Performance, "quality_weight");
        assert_eq!(value, Some(0.8));
    }

    #[test]
    fn test_user_preferences_learn_pattern() {
        let mut prefs = UserPreferences::new("user123");
        prefs.set_privacy_mode(false); // Disable privacy mode to allow pattern learning
        
        prefs.learn_pattern("visited_sites", "example.com");
        prefs.learn_pattern("visited_sites", "test.com");

        let patterns = prefs.get_patterns("visited_sites");
        assert!(patterns.is_some());
        assert_eq!(patterns.unwrap().len(), 2);
    }

    #[test]
    fn test_user_preferences_privacy_mode() {
        let mut prefs = UserPreferences::new("user123");
        prefs.set_privacy_mode(false);
        prefs.learn_pattern("test", "value");

        assert!(!prefs.learned_patterns.is_empty());

        prefs.set_privacy_mode(true);
        assert!(prefs.learned_patterns.is_empty());
    }

    #[test]
    fn test_user_preferences_json() {
        let mut prefs = UserPreferences::new("user123");
        prefs.set_preference(PreferenceCategory::Performance, "quality_weight", 0.8);

        let json = prefs.to_json().unwrap();
        let restored = UserPreferences::from_json(&json).unwrap();

        assert_eq!(restored.user_id, "user123");
        assert_eq!(
            restored.get_preference(&PreferenceCategory::Performance, "quality_weight"),
            Some(0.8)
        );
    }

    #[test]
    fn test_personalization_engine_creation() {
        let engine = PersonalizationEngine::new();
        assert_eq!(engine.user_count(), 0);
    }

    #[test]
    fn test_personalization_engine_register_user() {
        let mut engine = PersonalizationEngine::new();
        engine.register_user("user123");

        assert_eq!(engine.user_count(), 1);
        assert!(engine.get_user("user123").is_some());
    }

    #[test]
    fn test_personalization_engine_get_or_create() {
        let mut engine = PersonalizationEngine::new();
        
        let user = engine.get_or_create_user("user123");
        user.set_preference(PreferenceCategory::Performance, "quality_weight", 0.9);

        assert_eq!(engine.user_count(), 1);
        
        let user = engine.get_or_create_user("user123");
        assert_eq!(
            user.get_preference(&PreferenceCategory::Performance, "quality_weight"),
            Some(0.9)
        );
    }

    #[test]
    fn test_personalization_engine_personalize_rendering() {
        let mut engine = PersonalizationEngine::new();
        {
            let user = engine.register_user("user123");
            user.set_preference(PreferenceCategory::Performance, "quality_weight", 1.0);
        }

        let quality = engine.personalize_rendering("user123", 0.8);
        assert_eq!(quality, 0.8); // quality_weight=1.0 => 0.8 * (0.5 + 1.0*0.5) = 0.8
        
        // Test with lower quality preference
        {
            let user = engine.get_user_mut("user123").unwrap();
            user.set_preference(PreferenceCategory::Performance, "quality_weight", 0.0);
        }
        let quality_low = engine.personalize_rendering("user123", 0.8);
        assert_eq!(quality_low, 0.4); // quality_weight=0.0 => 0.8 * 0.5 = 0.4
    }

    #[test]
    fn test_personalization_engine_personalize_cache() {
        let mut engine = PersonalizationEngine::new();
        let user = engine.register_user("user123");
        user.set_preference(PreferenceCategory::Performance, "cache_aggressiveness", 0.9);

        let strategy = engine.personalize_cache_strategy("user123");
        assert_eq!(strategy, "aggressive");
    }

    #[test]
    fn test_personalization_engine_recommendations() {
        let mut engine = PersonalizationEngine::new();
        let user = engine.register_user("user123");
        user.set_preference(PreferenceCategory::Performance, "quality_weight", 0.9);

        let recommendations = engine.get_recommendations("user123");
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_personalization_engine_export_import() {
        let mut engine = PersonalizationEngine::new();
        let user = engine.register_user("user123");
        user.set_preference(PreferenceCategory::Performance, "quality_weight", 0.85);

        let json = engine.export_user("user123").unwrap();

        let mut new_engine = PersonalizationEngine::new();
        new_engine.import_user(&json).unwrap();

        let imported_user = new_engine.get_user("user123").unwrap();
        assert_eq!(
            imported_user.get_preference(&PreferenceCategory::Performance, "quality_weight"),
            Some(0.85)
        );
    }
}
