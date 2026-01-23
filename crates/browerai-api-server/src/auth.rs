use axum::{
    extract::{FromRequestParts, Request},
    http::request::Parts,
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// API key information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyInfo {
    pub key: String,
    pub name: String,
    pub created_at: String,
    pub last_used: Option<String>,
    pub enabled: bool,
}

/// API key store
pub struct ApiKeyStore {
    keys: Arc<DashMap<String, ApiKeyInfo>>,
}

impl ApiKeyStore {
    /// Create a new API key store
    pub fn new() -> Self {
        Self {
            keys: Arc::new(DashMap::new()),
        }
    }

    /// Generate a new API key
    pub fn generate(&self, name: &str) -> String {
        let key = format!("browerai_{}", Uuid::new_v4().to_string());
        let info = ApiKeyInfo {
            key: key.clone(),
            name: name.to_string(),
            created_at: chrono::Local::now().to_rfc3339(),
            last_used: None,
            enabled: true,
        };
        self.keys.insert(key.clone(), info);
        key
    }

    /// Validate an API key
    pub fn validate(&self, key: &str) -> bool {
        self.keys
            .get(key)
            .map(|info| info.value().enabled)
            .unwrap_or(false)
    }

    /// Validate and update last used time
    pub fn validate_and_update(&self, key: &str) -> bool {
        if let Some(mut info) = self.keys.get_mut(key) {
            if info.value().enabled {
                info.last_used = Some(chrono::Local::now().to_rfc3339());
                return true;
            }
        }
        false
    }

    /// List all API keys
    pub fn list_keys(&self) -> Vec<ApiKeyInfo> {
        self.keys
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Revoke an API key
    pub fn revoke(&self, key: &str) -> bool {
        if let Some(mut info) = self.keys.get_mut(key) {
            info.enabled = false;
            return true;
        }
        false
    }
}

impl Default for ApiKeyStore {
    fn default() -> Self {
        Self::new()
    }
}

/// API key extractor for request parts
pub struct ApiKey(pub String);

#[async_trait::async_trait]
impl<S> FromRequestParts<S> for ApiKey
where
    S: Send + Sync,
{
    type Rejection = StatusCode;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Check Authorization header
        if let Some(auth_header) = parts.headers.get("Authorization") {
            if let Ok(auth_str) = auth_header.to_str() {
                if let Some(key) = auth_str.strip_prefix("Bearer ") {
                    return Ok(ApiKey(key.to_string()));
                }
            }
        }

        // Check X-API-Key header
        if let Some(key_header) = parts.headers.get("X-API-Key") {
            if let Ok(key_str) = key_header.to_str() {
                return Ok(ApiKey(key_str.to_string()));
            }
        }

        Err(StatusCode::UNAUTHORIZED)
    }
}

/// Authentication middleware factory
pub fn auth_middleware(
    api_key_store: Arc<ApiKeyStore>,
    allow_unauthenticated: Vec<String>,
) -> impl Fn(Request, Next) -> futures::future::BoxFuture<'static, Result<Response, StatusCode>> {
    move |req: Request, next: Next| {
        let store = api_key_store.clone();
        let allow_list = allow_unauthenticated.clone();
        let path = req.uri().path().to_string();

        Box::pin(async move {
            // Check if path is in allow list
            if allow_list.iter().any(|p| path.starts_with(p)) {
                return Ok(next.run(req).await);
            }

            // Extract API key from request
            let key = if let Some(auth_header) = req.headers().get("Authorization") {
                if let Ok(auth_str) = auth_header.to_str() {
                    auth_str.strip_prefix("Bearer ").map(|k| k.to_string())
                } else {
                    None
                }
            } else if let Some(key_header) = req.headers().get("X-API-Key") {
                key_header.to_str().ok().map(|k| k.to_string())
            } else {
                None
            };

            // Validate key
            match key {
                Some(k) if store.validate_and_update(&k) => Ok(next.run(req).await),
                _ => Err(StatusCode::UNAUTHORIZED),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_api_key() {
        let store = ApiKeyStore::new();
        let key = store.generate("test-key");

        assert!(key.starts_with("browerai_"));
        assert!(store.validate(&key));
    }

    #[test]
    fn test_validate_api_key() {
        let store = ApiKeyStore::new();
        let key = store.generate("test-key");

        assert!(store.validate(&key));
        assert!(!store.validate("invalid-key"));
    }

    #[test]
    fn test_revoke_api_key() {
        let store = ApiKeyStore::new();
        let key = store.generate("test-key");

        assert!(store.validate(&key));
        assert!(store.revoke(&key));
        assert!(!store.validate(&key));
    }

    #[test]
    fn test_list_keys() {
        let store = ApiKeyStore::new();
        store.generate("key1");
        store.generate("key2");

        let keys = store.list_keys();
        assert_eq!(keys.len(), 2);
    }
}
