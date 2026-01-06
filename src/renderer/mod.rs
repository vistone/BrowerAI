#[cfg(feature = "ai")]
pub mod ai_regeneration;
pub mod engine;
pub mod layout;
pub mod paint;
pub mod predictive;
pub mod validation;

#[cfg(feature = "ai")]
pub use ai_regeneration::{RegeneratedWebsite, WebsiteRegenerator};
pub use engine::RenderEngine;
pub use predictive::PredictiveRenderer;
pub use validation::{AiLayoutHint, LayoutValidator, ValidationReport};
