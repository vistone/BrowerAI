use crate::error::Result;

/// Trait for parsers (HTML, CSS, JS)
pub trait Parser {
    type Input;
    type Output;
    
    fn parse(&self, input: Self::Input) -> Result<Self::Output>;
}

/// Trait for renderers
pub trait Renderer {
    type Input;
    type Output;
    
    fn render(&self, input: Self::Input) -> Result<Self::Output>;
}

/// Trait for AI models
pub trait AiModel {
    type Input;
    type Output;
    
    fn infer(&self, input: Self::Input) -> Result<Self::Output>;
    fn is_loaded(&self) -> bool;
}
