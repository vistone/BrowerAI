//! 数据库错误类型定义

use thiserror::Error;

#[derive(Error, Debug)]
pub enum DbError {
    #[error("Database connection error: {0}")]
    ConnectionError(String),

    #[error("Query error: {0}")]
    QueryError(String),

    #[error("Record not found")]
    NotFound,

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Database error: {0}")]
    DatabaseError(String),
}

pub type DbResult<T> = Result<T, DbError>;
