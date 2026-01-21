//! JavaScript Generator
//!
//! Generates JavaScript code based on website features and intent.

use crate::learning_sandbox::intent_analyzer::WebsiteIntent;
use anyhow::Result;

fn escape_js_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// JavaScript Generator Configuration
#[derive(Debug, Clone)]
pub struct JsGeneratorConfig {
    /// Use ES6+ features
    pub modern_js: bool,

    /// Include error handling
    pub error_handling: bool,

    /// Include console logging
    pub logging: bool,
}

impl Default for JsGeneratorConfig {
    fn default() -> Self {
        Self {
            modern_js: true,
            error_handling: true,
            logging: true,
        }
    }
}

/// JavaScript Generator
///
/// Generates JavaScript code based on website intent.
#[derive(Debug, Clone)]
pub struct JsGenerator {
    config: JsGeneratorConfig,
}

impl JsGenerator {
    /// Create a new JS generator
    pub fn new() -> Self {
        Self::with_config(JsGeneratorConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: JsGeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate JavaScript based on intent
    pub async fn generate(&self, intent: &WebsiteIntent) -> Result<String> {
        let mut js = String::new();

        // IIFE wrapper
        js.push_str("(function() {\n");
        js.push_str("    'use strict';\n\n");

        // Configuration
        js.push_str("    // Configuration\n");
        js.push_str("    const CONFIG = {{\n");
        js.push_str(&format!(
            "        siteType: '{}',\n",
            escape_js_string(&intent.website_type)
        ));
        js.push_str("        version: '1.0.0',\n");
        js.push_str("        debug: false\n");
        js.push_str("    };\n\n");

        // State management
        js.push_str("    // State\n");
        js.push_str("    const state = {\n");
        js.push_str("        initialized: false,\n");
        js.push_str("        user: null,\n");
        js.push_str("        cart: [],\n");
        js.push_str("        preferences: {}\n");
        js.push_str("    };\n\n");

        // Initialization
        js.push_str("    // Initialization\n");
        js.push_str("    function init() {\n");
        js.push_str("        if (state.initialized) return;\n");
        js.push_str("        \n");
        js.push_str("        bindEvents();\n");
        js.push_str("        loadState();\n");
        js.push_str("        state.initialized = true;\n");
        js.push_str("        \n");
        if self.config.logging {
            js.push_str("        console.log('Site initialized');\n");
        }
        js.push_str("    }\n\n");

        // Event binding
        js.push_str("    // Event Binding\n");
        js.push_str("    function bindEvents() {\n");
        js.push_str("        // Navigation\n");
        js.push_str("        const navLinks = document.querySelectorAll('.nav-list a');\n");
        js.push_str("        navLinks.forEach(link => {\n");
        js.push_str("            link.addEventListener('click', handleNavClick);\n");
        js.push_str("        });\n\n");

        // Feature-specific event bindings
        if intent
            .core_features
            .iter()
            .any(|f| f.contains("cart") || f.contains("checkout"))
        {
            js.push_str("        // Cart events\n");
            js.push_str(
                "        const addToCartBtns = document.querySelectorAll('.btn-add-to-cart');\n",
            );
            js.push_str("        addToCartBtns.forEach(btn => {\n");
            js.push_str("            btn.addEventListener('click', handleAddToCart);\n");
            js.push_str("        });\n\n");
        }

        if intent
            .core_features
            .iter()
            .any(|f| f.contains("authentication") || f.contains("login"))
        {
            js.push_str("        // Login form\n");
            js.push_str("        const loginForm = document.querySelector('.login-form');\n");
            js.push_str("        if (loginForm) {\n");
            js.push_str("            loginForm.addEventListener('submit', handleLogin);\n");
            js.push_str("        }\n\n");
        }

        if intent.core_features.iter().any(|f| f.contains("search")) {
            js.push_str("        // Search form\n");
            js.push_str("        const searchForm = document.querySelector('.search-form');\n");
            js.push_str("        if (searchForm) {\n");
            js.push_str("            searchForm.addEventListener('submit', handleSearch);\n");
            js.push_str("        }\n\n");
        }

        js.push_str("    }\n\n");

        // Event handlers
        js.push_str("    // Event Handlers\n");
        js.push_str("    function handleNavClick(e) {\n");
        js.push_str("        const href = this.getAttribute('href');\n");
        js.push_str("        if (href && href !== '#') {\n");
        js.push_str("            e.preventDefault();\n");
        js.push_str("            navigateTo(href);\n");
        js.push_str("        }\n");
        js.push_str("    }\n\n");

        // Feature-specific handlers
        if intent
            .core_features
            .iter()
            .any(|f| f.contains("cart") || f.contains("checkout"))
        {
            js.push_str("    function handleAddToCart(e) {\n");
            js.push_str("        const productId = this.dataset.productId;\n");
            js.push_str("        addToCart(productId);\n");
            js.push_str("    }\n\n");

            js.push_str("    function addToCart(productId) {\n");
            js.push_str("        state.cart.push(productId);\n");
            js.push_str("        saveState();\n");
            js.push_str("        updateCartUI();\n");
            if self.config.logging {
                js.push_str("        console.log('Added to cart:', productId);\n");
            }
            js.push_str("    }\n\n");

            js.push_str("    function updateCartUI() {\n");
            js.push_str("        const cartCount = document.querySelector('.cart-count');\n");
            js.push_str("        if (cartCount) {\n");
            js.push_str("            cartCount.textContent = state.cart.length;\n");
            js.push_str("        }\n");
            js.push_str("    }\n\n");
        }

        if intent
            .core_features
            .iter()
            .any(|f| f.contains("authentication") || f.contains("login"))
        {
            js.push_str("    function handleLogin(e) {\n");
            js.push_str("        e.preventDefault();\n");
            js.push_str("        const form = e.target;\n");
            js.push_str("        const email = form.querySelector('[name=\"email\"]').value;\n");
            js.push_str("        login(email);\n");
            js.push_str("    }\n\n");

            js.push_str("    function login(email) {\n");
            js.push_str("        state.user = { email };\n");
            js.push_str("        saveState();\n");
            if self.config.logging {
                js.push_str("        console.log('User logged in:', email);\n");
            }
            js.push_str("    }\n\n");
        }

        if intent.core_features.iter().any(|f| f.contains("search")) {
            js.push_str("    function handleSearch(e) {\n");
            js.push_str("        e.preventDefault();\n");
            js.push_str("        const query = this.querySelector('input').value;\n");
            js.push_str("        performSearch(query);\n");
            js.push_str("    }\n\n");

            js.push_str("    function performSearch(query) {\n");
            js.push_str("        if (self.config.logging) {\n");
            js.push_str("            console.log('Searching for:', query);\n");
            js.push_str("        }\n");
            js.push_str("        // Implement search logic\n");
            js.push_str("    }\n\n");
        }

        // Navigation
        js.push_str("    // Navigation\n");
        js.push_str("    function navigateTo(url) {\n");
        js.push_str("        if (self.config.logging) {\n");
        js.push_str("            console.log('Navigating to:', url);\n");
        js.push_str("        }\n");
        js.push_str("        window.location.href = url;\n");
        js.push_str("    }\n\n");

        // State management
        js.push_str("    // State Management\n");
        js.push_str("    function saveState() {\n");
        js.push_str("        try {\n");
        js.push_str("            localStorage.setItem('siteState', JSON.stringify(state));\n");
        js.push_str("        } catch (e) {\n");
        if self.config.error_handling {
            js.push_str("            console.error('Failed to save state:', e);\n");
        }
        js.push_str("        }\n");
        js.push_str("    }\n\n");

        js.push_str("    function loadState() {\n");
        js.push_str("        try {\n");
        js.push_str("            const saved = localStorage.getItem('siteState');\n");
        js.push_str("            if (saved) {\n");
        js.push_str("                const loaded = JSON.parse(saved);\n");
        js.push_str("                Object.assign(state, loaded);\n");
        js.push_str("            }\n");
        js.push_str("        } catch (e) {\n");
        if self.config.error_handling {
            js.push_str("            console.error('Failed to load state:', e);\n");
        }
        js.push_str("        }\n");
        js.push_str("    }\n\n");

        // Utility functions
        js.push_str("    // Utilities\n");
        js.push_str("    function $(selector) {\n");
        js.push_str("        return document.querySelector(selector);\n");
        js.push_str("    }\n\n");

        js.push_str("    function $$(selector) {\n");
        js.push_str("        return Array.from(document.querySelectorAll(selector));\n");
        js.push_str("    }\n\n");

        js.push_str("    function showNotification(message, type = 'info') {\n");
        js.push_str("        const notification = document.createElement('div');\n");
        js.push_str("        notification.className = `notification ${type}`;\n");
        js.push_str("        notification.textContent = message;\n");
        js.push_str("        document.body.appendChild(notification);\n");
        js.push_str("        setTimeout(() => {\n");
        js.push_str("            notification.remove();\n");
        js.push_str("        }, 3000);\n");
        js.push_str("    }\n\n");

        // Initialization call
        js.push_str("    // Start\n");
        js.push_str("    if (document.readyState === 'loading') {\n");
        js.push_str("        document.addEventListener('DOMContentLoaded', init);\n");
        js.push_str("    } else {\n");
        js.push_str("        init();\n");
        js.push_str("    }\n\n");

        // Expose public API
        js.push_str("    // Public API\n");
        js.push_str("    window.siteAPI = {\n");
        js.push_str("        init,\n");
        js.push_str("        addToCart: addToCart,\n");
        js.push_str("        login,\n");
        js.push_str("        logout,\n");
        js.push_str("        search: performSearch\n");
        js.push_str("    };\n\n");

        // Close IIFE
        js.push_str("})();\n");

        Ok(js)
    }
}

impl Default for JsGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning_sandbox::intent_analyzer::{
        ComplexityLevel, DesignStyle, LayoutType, PageStructure, WebsiteIntent,
    };
    use std::collections::HashMap;

    #[test]
    fn test_js_generator_creation() {
        let generator = JsGenerator::new();
        assert!(generator.config.modern_js);
        assert!(generator.config.error_handling);
    }

    #[test]
    fn test_generate_js() {
        let generator = JsGenerator::new();

        let intent = WebsiteIntent {
            website_type: "e-commerce".to_string(),
            confidence: 0.9,
            core_features: vec![
                "shopping_cart".to_string(),
                "checkout".to_string(),
                "user_account".to_string(),
            ],
            target_audience: "consumers".to_string(),
            design_style: DesignStyle::default(),
            structure: PageStructure {
                has_header: true,
                has_navigation: true,
                has_sidebar: false,
                has_main_content: true,
                has_footer: true,
                layout_type: LayoutType::SingleColumn,
                section_count: 5,
                complexity: ComplexityLevel::Moderate,
            },
            business_model: "B2C".to_string(),
            type_scores: HashMap::new(),
        };

        let js = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(generator.generate(&intent))
            .unwrap();

        assert!(js.contains("(function()"));
        assert!(js.contains("'use strict'"));
        assert!(js.contains("addToCart"));
        assert!(js.contains("handleNavClick"));
        assert!(js.contains("localStorage"));
        assert!(js.contains("siteAPI"));
    }

    #[test]
    fn test_generate_simple_js() {
        let generator = JsGenerator::new();

        let intent = WebsiteIntent {
            website_type: "blog".to_string(),
            confidence: 0.8,
            core_features: vec!["article_reader".to_string()],
            target_audience: "readers".to_string(),
            design_style: DesignStyle::default(),
            structure: PageStructure {
                has_header: true,
                has_navigation: true,
                has_sidebar: false,
                has_main_content: true,
                has_footer: true,
                layout_type: LayoutType::SingleColumn,
                section_count: 3,
                complexity: ComplexityLevel::Simple,
            },
            business_model: "content".to_string(),
            type_scores: HashMap::new(),
        };

        let js = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(generator.generate(&intent))
            .unwrap();

        assert!(js.contains("(function()"));
        assert!(js.contains("init"));
        assert!(js.contains("bindEvents"));
    }
}
