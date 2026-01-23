use anyhow::Result;
use browerai_api_server::{create_app, AppState};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    info!("🚀 BrowerAI API Server - Phase 3");
    info!("Version: {}", env!("CARGO_PKG_VERSION"));

    // Create application state
    #[cfg(feature = "onnx")]
    let state = {
        let model_dir =
            std::env::var("MODEL_DIR").unwrap_or_else(|_| "models/onnx_exports".to_string());
        info!("Loading AI models from: {}", model_dir);

        match AppState::with_models(&model_dir) {
            Ok(state) => {
                info!("✅ AI models loaded successfully");
                Arc::new(state)
            }
            Err(e) => {
                tracing::warn!("⚠️  Failed to load AI models: {}", e);
                tracing::warn!("   Falling back to basic mode");
                Arc::new(AppState::new())
            }
        }
    };

    #[cfg(not(feature = "onnx"))]
    let state = {
        info!("Running in basic mode (no AI features)");
        Arc::new(AppState::new())
    };

    // Create application
    let app = create_app(state.clone());

    // Bind to address
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    info!("🌐 Listening on http://{}", addr);
    info!("📚 API Documentation:");
    info!("   GET  /api/health       - Health check");
    info!("   GET  /api/version      - Version info");
    info!("   POST /api/v1/render    - Render HTML/CSS");
    info!("   POST /api/v1/parse/css - Parse CSS");
    info!("   POST /api/v1/parse/html - Parse HTML");

    // Run server
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
