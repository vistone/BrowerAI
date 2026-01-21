// 认证处理模块 - 自动化认证流程
use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use reqwest::{header::HeaderMap, Client};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use url::Url;

/// 认证配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    pub auth_type: AuthenticationType,
    pub credentials: HashMap<String, String>,
    pub auto_refresh: bool,
    pub token_storage: TokenStorage,
}

/// 认证类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuthenticationType {
    None,
    BearerToken,
    OAuth2 {
        client_id: String,
        client_secret: String,
        auth_url: String,
        token_url: String,
        redirect_uri: String,
        scope: Option<String>,
    },
    ApiKey {
        key_name: String,
        key_location: ApiKeyLocation,
    },
    BasicAuth,
    JwtToken,
    Custom(String),
}

/// API Key 位置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ApiKeyLocation {
    Header,
    QueryParam,
    Cookie,
}

/// Token 存储方式
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TokenStorage {
    Memory,
    LocalStorage,
    SessionStorage,
    Cookie,
    File(String),
}

/// 认证管理器
pub struct AuthManager {
    config: AuthConfig,
    tokens: HashMap<String, AuthToken>,
    http_client: Option<Client>,
}

/// 认证令牌
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    pub token_type: String,
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_in: Option<u64>,
    pub expires_at: Option<u64>,
    pub scope: Option<String>,
    pub token_response: Option<serde_json::Value>,
}

impl AuthManager {
    pub fn new(config: AuthConfig) -> Self {
        Self {
            config,
            tokens: HashMap::new(),
            http_client: None,
        }
    }

    pub fn with_http_client(mut self, client: Client) -> Self {
        self.http_client = Some(client);
        self
    }

    fn get_http_client(&self) -> Result<&Client> {
        self.http_client
            .as_ref()
            .context("HTTP client not configured. Use AuthManager::with_http_client()")
    }

    /// 执行认证流程
    pub async fn authenticate(&mut self) -> Result<AuthToken> {
        let auth_type = self.config.auth_type.clone();
        match &auth_type {
            AuthenticationType::None => Err(anyhow::anyhow!("No authentication configured")),
            AuthenticationType::BearerToken => self.handle_bearer_token().await,
            AuthenticationType::OAuth2 { .. } => self.handle_oauth2().await,
            AuthenticationType::ApiKey {
                key_name,
                key_location,
            } => self.handle_api_key(key_name, key_location).await,
            AuthenticationType::BasicAuth => self.handle_basic_auth().await,
            AuthenticationType::JwtToken => self.handle_jwt_token().await,
            AuthenticationType::Custom(method) => self.handle_custom_auth(method).await,
        }
    }

    /// Bearer Token 认证
    async fn handle_bearer_token(&mut self) -> Result<AuthToken> {
        let token_value = self
            .config
            .credentials
            .get("token")
            .context("Bearer token not found in credentials")?;

        let token = AuthToken {
            token_type: "Bearer".to_string(),
            access_token: token_value.clone(),
            refresh_token: None,
            expires_in: None,
            expires_at: None,
            scope: None,
            token_response: None,
        };

        self.store_token("bearer", token.clone()).await?;
        Ok(token)
    }

    /// OAuth 2.0 认证 - 完整实现
    async fn handle_oauth2(&mut self) -> Result<AuthToken> {
        let (client_id, client_secret, auth_url, token_url, redirect_uri, scope) =
            match &self.config.auth_type {
                AuthenticationType::OAuth2 {
                    client_id,
                    client_secret,
                    auth_url,
                    token_url,
                    redirect_uri,
                    scope,
                } => (
                    client_id.clone(),
                    client_secret.clone(),
                    auth_url.clone(),
                    token_url.clone(),
                    redirect_uri.clone(),
                    scope.clone(),
                ),
                _ => return Err(anyhow::anyhow!("Invalid OAuth2 configuration")),
            };

        log::info!("OAuth2 认证: client_id={}", client_id);

        if let Some(existing_token) = self.tokens.get("oauth2").cloned() {
            if !self.is_token_expired("oauth2") {
                return Ok(existing_token);
            }
            if let Some(refresh_token) = existing_token.refresh_token.as_ref() {
                match self
                    .perform_token_refresh(&token_url, &client_id, &client_secret, refresh_token)
                    .await
                {
                    Ok(new_token) => return Ok(new_token),
                    Err(e) => log::warn!("令牌刷新失败: {}", e),
                }
            }
        }

        // 检查是否已有授权码
        let auth_code = self.config.credentials.get("auth_code").cloned();

        if let Some(code) = auth_code {
            // 使用授权码交换令牌
            let token = self
                .exchange_code_for_token(
                    &token_url,
                    &client_id,
                    &client_secret,
                    &code,
                    &redirect_uri,
                    &scope,
                )
                .await?;
            self.store_token("oauth2", token.clone()).await?;
            return Ok(token);
        }

        // 生成授权 URL
        let auth_url_result =
            Self::build_authorization_url(&auth_url, &client_id, &redirect_uri, &scope);

        match auth_url_result {
            Ok(auth_url) => {
                // 在实际应用中，这里会打开浏览器让用户授权
                // 由于是自动化工具，我们返回授权 URL 供用户手动完成
                log::info!("请访问以下 URL 完成授权: {}", auth_url);

                // 返回一个特殊的令牌，表示需要用户交互
                let pending_token = AuthToken {
                    token_type: "Bearer".to_string(),
                    access_token: format!("PENDING:{}", auth_url),
                    refresh_token: None,
                    expires_in: None,
                    expires_at: None,
                    scope: scope.clone(),
                    token_response: Some(serde_json::json!({
                        "pending_authorization": true,
                        "authorization_url": auth_url,
                        "message": "请访问授权 URL 完成认证"
                    })),
                };

                self.store_token("oauth2", pending_token.clone()).await?;
                Ok(pending_token)
            }
            Err(e) => Err(e),
        }
    }

    /// 构建 OAuth 2.0 授权 URL
    fn build_authorization_url(
        auth_url: &str,
        client_id: &str,
        redirect_uri: &str,
        scope: &Option<String>,
    ) -> Result<String> {
        let mut url = Url::parse(auth_url).context("Invalid auth URL")?;

        let mut params = vec![
            ("client_id", client_id),
            ("redirect_uri", redirect_uri),
            ("response_type", "code"),
        ];

        if let Some(s) = scope {
            params.push(("scope", s.as_str()));
        }

        url.query_pairs_mut().extend_pairs(params);

        Ok(url.to_string())
    }

    /// 使用授权码交换访问令牌
    async fn exchange_code_for_token(
        &mut self,
        token_url: &str,
        client_id: &str,
        client_secret: &str,
        code: &str,
        redirect_uri: &str,
        scope: &Option<String>,
    ) -> Result<AuthToken> {
        let client = self.get_http_client()?;

        let mut params = vec![
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("client_id", client_id),
        ];

        if !client_secret.is_empty() {
            let creds = format!("{}:{}", client_id, client_secret);
            let encoded = STANDARD.encode(creds.as_bytes());
            let mut headers = HeaderMap::new();
            headers.insert("Authorization", format!("Basic {}", encoded).parse()?);
        }

        if let Some(s) = scope {
            params.push(("scope", s.as_str()));
        }

        let response = client
            .post(token_url)
            .form(&params)
            .send()
            .await
            .context("Failed to send token request")?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Token exchange failed: {}", error_text));
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse token response")?;

        self.parse_token_response(response_json)
    }

    /// 执行令牌刷新
    async fn perform_token_refresh(
        &mut self,
        token_url: &str,
        client_id: &str,
        client_secret: &str,
        refresh_token: &str,
    ) -> Result<AuthToken> {
        let client = self.get_http_client()?;

        let params = vec![
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", client_id),
        ];

        if !client_secret.is_empty() {
            let creds = format!("{}:{}", client_id, client_secret);
            let encoded = STANDARD.encode(creds.as_bytes());
            let mut headers = HeaderMap::new();
            headers.insert("Authorization", format!("Basic {}", encoded).parse()?);
        }

        let response = client
            .post(token_url)
            .form(&params)
            .send()
            .await
            .context("Failed to send token refresh request")?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Token refresh failed: {}", error_text));
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse token refresh response")?;

        self.parse_token_response(response_json)
    }

    /// 解析令牌响应
    fn parse_token_response(&self, response: serde_json::Value) -> Result<AuthToken> {
        let access_token = response
            .get("access_token")
            .context("No access_token in response")?
            .as_str()
            .context("access_token is not a string")?
            .to_string();

        let token_type = response
            .get("token_type")
            .and_then(|v| v.as_str())
            .unwrap_or("Bearer")
            .to_string();

        let refresh_token = response
            .get("refresh_token")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let expires_in = response.get("expires_in").and_then(|v| v.as_u64());

        let scope = response
            .get("scope")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // 计算过期时间
        let expires_at = expires_in.map(|seconds| {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs();
            now + seconds
        });

        Ok(AuthToken {
            token_type,
            access_token,
            refresh_token,
            expires_in,
            expires_at,
            scope,
            token_response: Some(response),
        })
    }

    /// API Key 认证
    async fn handle_api_key(
        &mut self,
        key_name: &str,
        location: &ApiKeyLocation,
    ) -> Result<AuthToken> {
        let api_key = self
            .config
            .credentials
            .get("api_key")
            .context("API key not found in credentials")?;

        log::info!(
            "API Key 认证: {}={}, location={:?}",
            key_name,
            api_key,
            location
        );

        let token = AuthToken {
            token_type: "ApiKey".to_string(),
            access_token: api_key.clone(),
            refresh_token: None,
            expires_in: None,
            expires_at: None,
            scope: None,
            token_response: None,
        };

        self.store_token("api_key", token.clone()).await?;
        Ok(token)
    }

    /// Basic Auth 认证
    async fn handle_basic_auth(&mut self) -> Result<AuthToken> {
        let username = self
            .config
            .credentials
            .get("username")
            .context("Username not found")?;
        let password = self
            .config
            .credentials
            .get("password")
            .context("Password not found")?;

        let credentials = format!("{}:{}", username, password);
        let encoded = STANDARD.encode(credentials.as_bytes());

        let token = AuthToken {
            token_type: "Basic".to_string(),
            access_token: encoded,
            refresh_token: None,
            expires_in: None,
            expires_at: None,
            scope: None,
            token_response: None,
        };

        self.store_token("basic", token.clone()).await?;
        Ok(token)
    }

    /// JWT Token 认证
    async fn handle_jwt_token(&mut self) -> Result<AuthToken> {
        let jwt = self
            .config
            .credentials
            .get("jwt")
            .context("JWT token not found")?;

        // 解析 JWT
        let parts: Vec<&str> = jwt.split('.').collect();
        if parts.len() != 3 {
            return Err(anyhow::anyhow!("Invalid JWT format: expected 3 parts"));
        }

        // 解析 payload (第二部分)
        let payload_base64 = parts[1];
        let payload = match Self::base64_url_decode(payload_base64) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(json_str) => serde_json::from_str::<serde_json::Value>(&json_str).ok(),
                _ => None,
            },
            _ => None,
        }
        .context("Failed to parse JWT payload")?;

        let expires_at = Self::extract_jwt_expiry_from_payload(&payload);
        let scope = payload
            .get("scope")
            .or(payload.get("scp"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let token = AuthToken {
            token_type: "JWT".to_string(),
            access_token: jwt.clone(),
            refresh_token: None,
            expires_in: None,
            expires_at,
            scope,
            token_response: Some(serde_json::json!({
                "header": Self::base64_url_decode(parts[0]).ok().and_then(|b| String::from_utf8(b).ok()),
                "payload": payload
            })),
        };

        self.store_token("jwt", token.clone()).await?;
        Ok(token)
    }

    /// Base64 URL 解码
    fn base64_url_decode(input: &str) -> Result<Vec<u8>> {
        let mut decoded = input.replace('-', "+").replace('_', "/");

        let padding = (4 - decoded.len() % 4) % 4;
        decoded.push_str(&"=".repeat(padding));

        STANDARD.decode(decoded).context("Base64 decode failed")
    }

    /// 从 JWT payload 提取过期时间
    fn extract_jwt_expiry_from_payload(payload: &serde_json::Value) -> Option<u64> {
        // 尝试常见的过期时间字段
        let exp_field = payload
            .get("exp")
            .or(payload.get("expires_at"))
            .or(payload.get("expire_at"))?;

        exp_field.as_u64().or(exp_field.as_i64().map(|i| i as u64))
    }

    /// 自定义认证
    async fn handle_custom_auth(&mut self, method: &str) -> Result<AuthToken> {
        log::info!("自定义认证方法: {}", method);

        let token = AuthToken {
            token_type: "Custom".to_string(),
            access_token: "custom_token".to_string(),
            refresh_token: None,
            expires_in: None,
            expires_at: None,
            scope: None,
            token_response: None,
        };

        self.store_token("custom", token.clone()).await?;
        Ok(token)
    }

    /// 存储令牌
    async fn store_token(&mut self, key: &str, token: AuthToken) -> Result<()> {
        self.tokens.insert(key.to_string(), token.clone());

        match &self.config.token_storage {
            TokenStorage::Memory => Ok(()),
            TokenStorage::File(path) => {
                let json = serde_json::to_string_pretty(&token)?;
                tokio::fs::write(path, json).await?;
                Ok(())
            }
            _ => {
                log::warn!(
                    "Token storage type {:?} not fully implemented, using memory only",
                    self.config.token_storage
                );
                Ok(())
            }
        }
    }

    /// 获取令牌
    pub fn get_token(&self, key: &str) -> Option<&AuthToken> {
        self.tokens.get(key)
    }

    /// 刷新令牌 - 完整实现
    pub async fn refresh_token(&mut self, key: &str) -> Result<AuthToken> {
        let token = self.tokens.get(key).context("Token not found")?.clone();

        if let Some(refresh_token) = &token.refresh_token {
            log::info!("刷新令牌: {}", refresh_token);

            let (client_id, client_secret, token_url) = match &self.config.auth_type {
                AuthenticationType::OAuth2 {
                    client_id,
                    client_secret,
                    token_url,
                    ..
                } => (client_id.clone(), client_secret.clone(), token_url.clone()),
                _ => return Err(anyhow::anyhow!("Invalid auth type for token refresh")),
            };

            let new_token = self
                .perform_token_refresh(&token_url, &client_id, &client_secret, refresh_token)
                .await?;
            self.store_token(key, new_token.clone()).await?;
            Ok(new_token)
        } else {
            Err(anyhow::anyhow!(
                "No refresh token available for key: {}",
                key
            ))
        }
    }

    /// 检查令牌是否过期
    pub fn is_token_expired(&self, key: &str) -> bool {
        if let Some(token) = self.tokens.get(key) {
            if let Some(expires_at) = token.expires_at {
                let now = chrono::Utc::now().timestamp() as u64;
                return now >= expires_at;
            }
            // 如果没有过期时间，假设永不过期
            return false;
        }
        true
    }

    /// 检查令牌是否即将过期（5分钟内）
    pub fn is_token_expiring_soon(&self, key: &str) -> bool {
        if let Some(token) = self.tokens.get(key) {
            if let Some(expires_at) = token.expires_at {
                let now = chrono::Utc::now().timestamp() as u64;
                let five_minutes = 300;
                return expires_at - now < five_minutes;
            }
        }
        false
    }

    /// 构建认证头
    pub fn build_auth_header(&self, key: &str) -> Result<(String, String)> {
        let token = self.tokens.get(key).context("Token not found")?;

        let header_value = match token.token_type.as_str() {
            "Bearer" | "JWT" => format!("Bearer {}", token.access_token),
            "Basic" => format!("Basic {}", token.access_token),
            "ApiKey" => token.access_token.clone(),
            _ => token.access_token.clone(),
        };

        Ok(("Authorization".to_string(), header_value))
    }

    /// 构建带认证的请求头
    pub fn build_auth_headers(&self, key: &str) -> Result<HeaderMap> {
        let (name, value) = self.build_auth_header(key)?;
        let mut headers = HeaderMap::new();
        if let Ok(name) = name.parse::<reqwest::header::HeaderName>() {
            if let Ok(value) = value.parse::<reqwest::header::HeaderValue>() {
                headers.insert(name, value);
            }
        }
        Ok(headers)
    }

    /// 清除令牌
    pub fn clear_token(&mut self, key: &str) {
        self.tokens.remove(key);
    }

    /// 获取所有令牌密钥
    pub fn get_token_keys(&self) -> Vec<&String> {
        self.tokens.keys().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jwt_parsing() {
        // 测试 JWT 解析
        let header = r#"{"alg":"HS256","typ":"JWT"}"#;
        let payload = r#"{"sub":"1234567890","name":"John Doe","exp":1893456000}"#;
        let signature = "dummy_signature";

        let encoded_header = STANDARD
            .encode(header.as_bytes())
            .replace('=', "")
            .replace('+', "-")
            .replace('/', "_");
        let encoded_payload = STANDARD
            .encode(payload.as_bytes())
            .replace('=', "")
            .replace('+', "-")
            .replace('/', "_");

        let jwt = format!("{}.{}.{}", encoded_header, encoded_payload, signature);

        let mut credentials = HashMap::new();
        credentials.insert("jwt".to_string(), jwt);

        let config = AuthConfig {
            auth_type: AuthenticationType::JwtToken,
            credentials,
            auto_refresh: false,
            token_storage: TokenStorage::Memory,
        };

        let mut manager = AuthManager::new(config);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(manager.authenticate());

        assert!(result.is_ok());
        let token = result.unwrap();
        assert_eq!(token.token_type, "JWT");
        assert!(token.expires_at.is_some());
    }

    #[tokio::test]
    async fn test_oauth2_pending_auth() {
        let credentials = HashMap::new();
        let config = AuthConfig {
            auth_type: AuthenticationType::OAuth2 {
                client_id: "test_client".to_string(),
                client_secret: "test_secret".to_string(),
                auth_url: "https://example.com/oauth/authorize".to_string(),
                token_url: "https://example.com/oauth/token".to_string(),
                redirect_uri: "http://localhost:8080/callback".to_string(),
                scope: Some("read write".to_string()),
            },
            credentials,
            auto_refresh: false,
            token_storage: TokenStorage::Memory,
        };

        let mut manager = AuthManager::new(config);
        let result = manager.authenticate().await;

        assert!(result.is_ok());
        let token = result.unwrap();
        assert!(token.access_token.starts_with("PENDING:"));
        assert!(token
            .token_response
            .as_ref()
            .unwrap()
            .get("pending_authorization")
            .unwrap()
            .as_bool()
            .unwrap());
    }

    #[tokio::test]
    async fn test_token_expiry_check() {
        let mut credentials = HashMap::new();
        credentials.insert("token".to_string(), "test_token".to_string());

        let config = AuthConfig {
            auth_type: AuthenticationType::BearerToken,
            credentials,
            auto_refresh: false,
            token_storage: TokenStorage::Memory,
        };

        let mut manager = AuthManager::new(config);

        // 认证
        let token = manager.authenticate().await.unwrap();
        assert!(!manager.is_token_expired("bearer"));

        // 清除令牌后检查
        manager.clear_token("bearer");
        assert!(manager.is_token_expired("bearer"));
    }

    #[test]
    fn test_auth_header_building() {
        let mut credentials = HashMap::new();
        credentials.insert("token".to_string(), "my_bearer_token".to_string());

        let config = AuthConfig {
            auth_type: AuthenticationType::BearerToken,
            credentials,
            auto_refresh: false,
            token_storage: TokenStorage::Memory,
        };

        let manager = AuthManager::new(config);

        let (name, value) = manager.build_auth_header("nonexistent");
        assert!(name.is_empty());

        // 手动设置令牌
        let token = AuthToken {
            token_type: "Bearer".to_string(),
            access_token: "test_token".to_string(),
            refresh_token: None,
            expires_in: None,
            expires_at: None,
            scope: None,
            token_response: None,
        };
        let manager = AuthManager {
            config,
            tokens: vec![("test".to_string(), token)].into_iter().collect(),
            http_client: None,
        };

        let (name, value) = manager.build_auth_header("test").unwrap();
        assert_eq!(name, "Authorization");
        assert_eq!(value, "Bearer test_token");
    }
}
