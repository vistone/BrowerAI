use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
};
use browerai_html_parser::HtmlParser;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info};
use validator::Validate;

use crate::metrics;
use crate::state::AppState;

/// Request to render HTML/CSS
#[derive(Debug, Deserialize, Validate)]
pub struct RenderRequest {
    #[validate(length(min = 1, max = 1000000))]
    pub html: String,

    pub css: Option<String>,

    #[serde(default)]
    pub use_ai: bool,
}

/// Render response
#[derive(Debug, Serialize)]
pub struct RenderResponse {
    pub success: bool,
    pub message: String,
    pub rules_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ai_enhanced: Option<bool>,
}

/// CSS parsing request
#[derive(Debug, Deserialize, Validate)]
pub struct ParseCssRequest {
    #[validate(length(min = 1, max = 500000))]
    pub css: String,

    #[serde(default)]
    pub use_ai: bool,
}

/// CSS parsing response
#[derive(Debug, Serialize)]
pub struct ParseCssResponse {
    pub success: bool,
    pub rules_count: usize,
    pub ai_enhanced: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_properties: Option<Vec<PredictedPropertyResponse>>,
}

/// Predicted property in response
#[derive(Debug, Serialize)]
pub struct PredictedPropertyResponse {
    pub name: String,
    pub confidence: f32,
}

/// HTML parsing request
#[derive(Debug, Deserialize, Validate)]
pub struct ParseHtmlRequest {
    #[validate(length(min = 1, max = 1000000))]
    pub html: String,
}

/// HTML parsing response
#[derive(Debug, Serialize)]
pub struct ParseHtmlResponse {
    pub success: bool,
    pub node_count: usize,
    pub depth: usize,
    pub message: String,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub details: Option<String>,
}

/// Render HTML/CSS endpoint
pub async fn render_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RenderRequest>,
) -> impl IntoResponse {
    info!("Received render request (use_ai: {})", req.use_ai);

    // Validate request
    if let Err(e) = req.validate() {
        error!("Validation error: {}", e);
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Validation failed".to_string(),
                details: Some(e.to_string()),
            }),
        )
            .into_response();
    }

    // Parse CSS if provided
    let rules_count = if let Some(css) = &req.css {
        match state.css_parser().parse(css) {
            Ok(rules) => {
                info!("Parsed {} CSS rules", rules.len());
                rules.len()
            }
            Err(e) => {
                error!("Failed to parse CSS: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Failed to parse CSS".to_string(),
                        details: Some(e.to_string()),
                    }),
                )
                    .into_response();
            }
        }
    } else {
        0
    };

    let response = RenderResponse {
        success: true,
        message: "Rendering completed".to_string(),
        rules_count,
        ai_enhanced: Some(req.use_ai && state.is_ai_enabled()),
    };

    (StatusCode::OK, Json(response)).into_response()
}

/// Parse CSS endpoint
pub async fn parse_css_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ParseCssRequest>,
) -> impl IntoResponse {
    info!("Received CSS parse request (use_ai: {})", req.use_ai);

    // Validate request
    if let Err(e) = req.validate() {
        error!("Validation error: {}", e);
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Validation failed".to_string(),
                details: Some(e.to_string()),
            }),
        )
            .into_response();
    }

    // Parse CSS
    match state.css_parser().parse(&req.css) {
        Ok(rules) => {
            info!("Parsed {} CSS rules", rules.len());

            // Record CSS parsing metrics
            metrics::record_css_rules_parsed(rules.len(), false);

            // Try AI enhancement if requested
            #[cfg(feature = "onnx")]
            let (ai_enhanced, predicted_properties) = if req.use_ai && state.is_ai_enabled() {
                // Get first selector for demo
                if let Some(rule) = rules.first() {
                    let inference_start = Instant::now();
                    match state.css_parser().predict_properties(&rule.selector) {
                        Ok(props) => {
                            let inference_duration = inference_start.elapsed().as_secs_f64();
                            metrics::record_ai_inference(
                                "property_predictor",
                                true,
                                inference_duration,
                            );
                            metrics::record_css_rules_parsed(rules.len(), true);

                            let response_props: Vec<_> = props
                                .into_iter()
                                .map(|p| PredictedPropertyResponse {
                                    name: p.name,
                                    confidence: p.confidence,
                                })
                                .collect();
                            (true, Some(response_props))
                        }
                        Err(e) => {
                            let inference_duration = inference_start.elapsed().as_secs_f64();
                            metrics::record_ai_inference(
                                "property_predictor",
                                false,
                                inference_duration,
                            );
                            error!("Failed to predict properties: {}", e);
                            (false, None)
                        }
                    }
                } else {
                    (false, None)
                }
            } else {
                (false, None)
            };

            #[cfg(not(feature = "onnx"))]
            let (ai_enhanced, predicted_properties) = (false, None);

            let response = ParseCssResponse {
                success: true,
                rules_count: rules.len(),
                ai_enhanced,
                predicted_properties,
            };

            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            error!("Failed to parse CSS: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Failed to parse CSS".to_string(),
                    details: Some(e.to_string()),
                }),
            )
                .into_response()
        }
    }
}

/// Parse HTML endpoint
pub async fn parse_html_handler(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<ParseHtmlRequest>,
) -> impl IntoResponse {
    info!("Received HTML parse request");

    // Validate request
    if let Err(e) = req.validate() {
        error!("Validation error: {}", e);
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Validation failed".to_string(),
                details: Some(e.to_string()),
            }),
        )
            .into_response();
    }

    let parser = HtmlParser::new();

    if let Err(e) = parser.parse(&req.html) {
        error!("Failed to parse HTML: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to parse HTML".to_string(),
                details: Some(e.to_string()),
            }),
        )
            .into_response();
    }

    let stats = parser.get_stats(&req.html);

    let response = ParseHtmlResponse {
        success: true,
        node_count: stats.tag_count,
        depth: stats.max_depth,
        message: format!("Parsed HTML ({} bytes)", req.html.len()),
    };

    (StatusCode::OK, Json(response)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_request_validation() {
        let req = RenderRequest {
            html: "".to_string(),
            css: None,
            use_ai: false,
        };
        assert!(req.validate().is_err());

        let req = RenderRequest {
            html: "<div>Test</div>".to_string(),
            css: None,
            use_ai: false,
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_parse_css_request_validation() {
        let req = ParseCssRequest {
            css: "".to_string(),
            use_ai: false,
        };
        assert!(req.validate().is_err());

        let req = ParseCssRequest {
            css: ".test { color: red; }".to_string(),
            use_ai: false,
        };
        assert!(req.validate().is_ok());
    }
}
