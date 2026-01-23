//! 数据库 CRUD 操作

use crate::connection::DbConnection;
use crate::errors::{DbError, DbResult};
use crate::schema::CacheStats;
use sqlx::{QueryBuilder, Row};

/// 数据库操作接口
pub struct DbOperations {
    conn: DbConnection,
}

impl DbOperations {
    pub fn new(conn: DbConnection) -> Self {
        Self { conn }
    }

    // ==================== CACHE ENTRY OPERATIONS ====================

    /// 获取缓存值
    pub async fn get(&self, key: &str) -> DbResult<Option<Vec<u8>>> {
        let pool = self.conn.pool();

        let row = sqlx::query_scalar::<_, Option<Vec<u8>>>(
            "SELECT value FROM cache_entries WHERE key = $1 AND deleted_at IS NULL",
        )
        .bind(key)
        .fetch_optional(pool.as_ref())
        .await
        .map_err(|e| DbError::QueryError(e.to_string()))?;

        Ok(row.flatten())
    }

    /// 设置缓存值
    pub async fn set(&self, key: &str, value: Vec<u8>, ttl_seconds: i32) -> DbResult<()> {
        let pool = self.conn.pool();

        sqlx::query(
            "INSERT INTO cache_entries (key, value, ttl_seconds, created_at, updated_at, hits)
             VALUES ($1, $2, $3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0)
             ON CONFLICT (key) DO UPDATE SET 
                 value = $2, 
                 ttl_seconds = $3, 
                 updated_at = CURRENT_TIMESTAMP",
        )
        .bind(key)
        .bind(&value)
        .bind(ttl_seconds)
        .execute(pool.as_ref())
        .await
        .map_err(|e| DbError::QueryError(e.to_string()))?;

        Ok(())
    }

    /// 删除缓存
    pub async fn delete(&self, key: &str) -> DbResult<()> {
        let pool = self.conn.pool();

        sqlx::query("UPDATE cache_entries SET deleted_at = CURRENT_TIMESTAMP WHERE key = $1 AND deleted_at IS NULL")
            .bind(key)
            .execute(pool.as_ref())
            .await
            .map_err(|e| DbError::QueryError(e.to_string()))?;

        Ok(())
    }

    /// 清空所有缓存
    pub async fn clear(&self) -> DbResult<u64> {
        let pool = self.conn.pool();

        let result = sqlx::query(
            "UPDATE cache_entries SET deleted_at = CURRENT_TIMESTAMP WHERE deleted_at IS NULL",
        )
        .execute(pool.as_ref())
        .await
        .map_err(|e| DbError::QueryError(e.to_string()))?;

        Ok(result.rows_affected())
    }

    /// 批量设置缓存值（单条 UPSERT）
    pub async fn set_batch(&self, entries: &[(String, Vec<u8>, i32)]) -> DbResult<u64> {
        if entries.is_empty() {
            return Ok(0);
        }

        let pool = self.conn.pool();

        let mut qb = QueryBuilder::new(
            "INSERT INTO cache_entries (key, value, ttl_seconds, created_at, updated_at, hits) ",
        );
        qb.push_values(entries, |mut b, (k, v, ttl)| {
            b.push_bind(k)
                .push_bind(v)
                .push_bind(ttl)
                .push("CURRENT_TIMESTAMP")
                .push("CURRENT_TIMESTAMP")
                .push_bind(0_i64);
        });
        qb.push(
            " ON CONFLICT (key) DO UPDATE SET \
                value = EXCLUDED.value, \
                ttl_seconds = EXCLUDED.ttl_seconds, \
                updated_at = CURRENT_TIMESTAMP",
        );

        let rows = qb
            .build()
            .execute(pool.as_ref())
            .await
            .map_err(|e| DbError::QueryError(e.to_string()))?;

        Ok(rows.rows_affected())
    }

    /// 批量获取缓存值
    pub async fn get_batch(&self, keys: &[String]) -> DbResult<Vec<(String, Vec<u8>)>> {
        if keys.is_empty() {
            return Ok(vec![]);
        }

        let pool = self.conn.pool();

        let rows = sqlx::query(
            "SELECT key, value FROM cache_entries WHERE deleted_at IS NULL AND key = ANY($1)",
        )
        .bind(keys)
        .fetch_all(pool.as_ref())
        .await
        .map_err(|e| DbError::QueryError(e.to_string()))?;

        Ok(rows
            .into_iter()
            .filter_map(|row| {
                let key: String = row.try_get("key").ok()?;
                let value: Vec<u8> = row.try_get("value").ok()?;
                Some((key, value))
            })
            .collect())
    }

    /// 清理过期的缓存
    pub async fn cleanup_expired(&self) -> DbResult<u64> {
        let pool = self.conn.pool();

        let result = sqlx::query(
            "UPDATE cache_entries 
             SET deleted_at = CURRENT_TIMESTAMP 
             WHERE deleted_at IS NULL 
             AND created_at + INTERVAL '1 second' * ttl_seconds < CURRENT_TIMESTAMP",
        )
        .execute(pool.as_ref())
        .await
        .map_err(|e| DbError::QueryError(e.to_string()))?;

        Ok(result.rows_affected())
    }

    // ==================== STATISTICS OPERATIONS ====================

    /// 记录统计数据
    pub async fn record_stats(&self, stats: &CacheStats) -> DbResult<()> {
        let pool = self.conn.pool();

        sqlx::query(
            "INSERT INTO cache_stats (total_requests, total_hits, total_misses, hit_rate, avg_latency_ms)
             VALUES ($1, $2, $3, $4, $5)"
        )
        .bind(stats.total_requests)
        .bind(stats.total_hits)
        .bind(stats.total_misses)
        .bind(stats.hit_rate)
        .bind(stats.avg_latency_ms)
        .execute(pool.as_ref())
        .await
        .map_err(|e| DbError::QueryError(e.to_string()))?;

        Ok(())
    }

    // ==================== EVENT LOG OPERATIONS ====================

    /// 记录缓存事件
    pub async fn log_event(
        &self,
        event_type: &str,
        key: &str,
        value_size: i32,
        latency_ms: i32,
    ) -> DbResult<()> {
        let pool = self.conn.pool();

        sqlx::query(
            "INSERT INTO cache_events (event_type, key, value_size, latency_ms)
             VALUES ($1, $2, $3, $4)",
        )
        .bind(event_type)
        .bind(key)
        .bind(value_size)
        .bind(latency_ms)
        .execute(pool.as_ref())
        .await
        .map_err(|e| DbError::QueryError(e.to_string()))?;

        Ok(())
    }
}
