use axum::{
    extract::{Json, State},
    http::StatusCode,
    middleware,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::Serialize;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

pub mod auth;
pub mod handlers;
pub mod metrics;
mod metrics_middleware;
pub mod rate_limit;
pub mod state;

pub use auth::{ApiKeyInfo, ApiKeyStore};
pub use metrics_middleware::metrics_middleware;
pub use rate_limit::{RateLimitConfig, RequestRateLimiter};
pub use state::AppState;

/// API version information
#[derive(Debug, Serialize)]
pub struct ApiVersion {
    pub version: String,
    pub phase: String,
    pub features: Vec<String>,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub ai_enabled: bool,
}

/// Create the main application router
pub fn create_app(state: Arc<AppState>) -> Router {
    // API routes
    let api_routes = Router::new()
        .route("/health", get(health_handler))
        .route("/version", get(version_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/render", post(handlers::render_handler))
        .route("/v1/parse/css", post(handlers::parse_css_handler))
        .route("/v1/parse/html", post(handlers::parse_html_handler));

    // Main router with CORS and metrics middleware
    Router::new()
        .nest("/api", api_routes)
        .layer(middleware::from_fn(metrics_middleware))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Health check endpoint
async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let response = HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        ai_enabled: state.is_ai_enabled(),
    };

    (StatusCode::OK, Json(response))
}

/// Version endpoint
async fn version_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut features = vec!["css-parser".to_string(), "html-parser".to_string()];

    if state.is_ai_enabled() {
        features.push("ai".to_string());
    }

    #[cfg(feature = "onnx")]
    features.push("onnx".to_string());

    let response = ApiVersion {
        version: env!("CARGO_PKG_VERSION").to_string(),
        phase: "Phase 3 Week 3".to_string(),
        features,
    };

    (StatusCode::OK, Json(response))
}

/// Metrics endpoint (Prometheus format)
async fn metrics_handler() -> impl IntoResponse {
    match metrics::export_metrics() {
        Ok(metrics_text) => (
            StatusCode::OK,
            [("content-type", "text/plain; version=0.0.4")],
            metrics_text,
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            [("content-type", "text/plain; version=0.0.4")],
            format!("Error exporting metrics: {}", e),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_app_creation() {
        let state = Arc::new(AppState::new());
        let app = create_app(state);
        // Should compile and create router
        assert!(true);
    }
}
