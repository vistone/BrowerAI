//! 数据库表结构定义

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// 缓存条目结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub ttl_seconds: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub hits: i64,
}

/// 缓存统计结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_requests: i64,
    pub total_hits: i64,
    pub total_misses: i64,
    pub hit_rate: f64,
    pub avg_latency_ms: f64,
    pub recorded_at: DateTime<Utc>,
}

/// 缓存事件日志
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEvent {
    pub id: i64,
    pub event_type: String,
    pub key: String,
    pub value_size: i32,
    pub latency_ms: i32,
    pub timestamp: DateTime<Utc>,
}

pub const SQL_CREATE_CACHE_ENTRIES: &str = r#"
CREATE TABLE IF NOT EXISTS cache_entries (
    key VARCHAR(255) PRIMARY KEY,
    value BYTEA NOT NULL,
    ttl_seconds INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    hits BIGINT DEFAULT 0,
    deleted_at TIMESTAMP WITH TIME ZONE
);
"#;

pub const SQL_CREATE_CACHE_STATS: &str = r#"
CREATE TABLE IF NOT EXISTS cache_stats (
    id BIGSERIAL PRIMARY KEY,
    total_requests BIGINT,
    total_hits BIGINT,
    total_misses BIGINT,
    hit_rate DOUBLE PRECISION,
    avg_latency_ms DOUBLE PRECISION,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"#;

pub const SQL_CREATE_CACHE_EVENTS: &str = r#"
CREATE TABLE IF NOT EXISTS cache_events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(20),
    key VARCHAR(255),
    value_size INTEGER,
    latency_ms INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"#;

pub const SQL_CREATE_INDEXES: &str = r#"
CREATE INDEX IF NOT EXISTS idx_cache_entries_created_at ON cache_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_cache_entries_ttl ON cache_entries(ttl_seconds);
CREATE INDEX IF NOT EXISTS idx_cache_events_timestamp ON cache_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_cache_events_event_type ON cache_events(event_type);
"#;
