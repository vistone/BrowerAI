// 错误处理

use thiserror::Error;

#[derive(Error, Debug)]
pub enum RenderError {
    #[error("Layout error: {0}")]
    LayoutError(String),

    #[error("Paint error: {0}")]
    PaintError(String),

    #[error("Invalid style: {0}")]
    InvalidStyle(String),
}
