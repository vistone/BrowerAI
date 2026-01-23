use axum::{body::Body, extract::Request, middleware::Next, response::Response};
use std::time::Instant;

use crate::metrics;

/// Middleware to record HTTP request metrics
pub async fn metrics_middleware(req: Request<Body>, next: Next) -> Response {
    let start = Instant::now();
    let method = req.method().to_string();
    let path = req.uri().path().to_string();

    // Extract endpoint for metrics (strip /api prefix)
    let endpoint = path.strip_prefix("/api").unwrap_or(&path).to_string();

    // Process the request
    let response = next.run(req).await;

    // Record metrics
    let duration = start.elapsed();
    let status = response.status().as_u16();

    metrics::record_http_request(&endpoint, &method, status);
    metrics::record_http_duration(&endpoint, &method, duration.as_secs_f64());

    response
}
