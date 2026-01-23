//! 数据库连接管理

use crate::errors::{DbError, DbResult};
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use std::sync::Arc;

/// 数据库连接池管理器
pub struct DbConnection {
    pool: Arc<PgPool>,
}

impl DbConnection {
    /// 创建新的数据库连接
    pub async fn new(database_url: &str, max_connections: u32) -> DbResult<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(max_connections)
            .connect(database_url)
            .await
            .map_err(|e| DbError::ConnectionError(e.to_string()))?;

        Ok(Self {
            pool: Arc::new(pool),
        })
    }

    /// 获取连接池引用
    pub fn pool(&self) -> Arc<PgPool> {
        self.pool.clone()
    }

    /// 测试数据库连接
    pub async fn health_check(&self) -> DbResult<()> {
        sqlx::query("SELECT 1")
            .fetch_one(self.pool.as_ref())
            .await
            .map_err(|e| DbError::ConnectionError(format!("Health check failed: {}", e)))?;

        Ok(())
    }

    /// 初始化数据库表
    pub async fn init_tables(&self) -> DbResult<()> {
        use crate::schema::*;

        // 创建表
        sqlx::query(SQL_CREATE_CACHE_ENTRIES)
            .execute(self.pool.as_ref())
            .await
            .map_err(|e| {
                DbError::DatabaseError(format!("Failed to create cache_entries table: {}", e))
            })?;

        sqlx::query(SQL_CREATE_CACHE_STATS)
            .execute(self.pool.as_ref())
            .await
            .map_err(|e| {
                DbError::DatabaseError(format!("Failed to create cache_stats table: {}", e))
            })?;

        sqlx::query(SQL_CREATE_CACHE_EVENTS)
            .execute(self.pool.as_ref())
            .await
            .map_err(|e| {
                DbError::DatabaseError(format!("Failed to create cache_events table: {}", e))
            })?;

        // 创建索引
        sqlx::query(SQL_CREATE_INDEXES)
            .execute(self.pool.as_ref())
            .await
            .map_err(|e| DbError::DatabaseError(format!("Failed to create indexes: {}", e)))?;

        Ok(())
    }
}

impl Clone for DbConnection {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
        }
    }
}
