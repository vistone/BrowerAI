use browerai_db::{CacheStats, DbConnection, DbOperations};
use std::env;

fn get_database_url() -> Option<String> {
    env::var("DATABASE_URL").ok()
}

#[tokio::test]
async fn db_connect_and_init_tables() {
    let Some(url) = get_database_url() else {
        eprintln!("Skipping DB test: DATABASE_URL not set");
        return;
    };

    let conn = match DbConnection::new(&url, 5).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping DB test: cannot connect ({})", e);
            return;
        }
    };

    if let Err(e) = conn.init_tables().await {
        panic!("Failed to init tables: {}", e);
    }

    if let Err(e) = conn.health_check().await {
        panic!("Health check failed: {}", e);
    }
}

#[tokio::test]
async fn crud_set_get_delete_flow() {
    let Some(url) = get_database_url() else {
        eprintln!("Skipping DB test: DATABASE_URL not set");
        return;
    };

    let Ok(conn) = DbConnection::new(&url, 5).await else {
        eprintln!("Skipping DB test: cannot connect");
        return;
    };
    conn.init_tables().await.expect("init tables");

    let ops = DbOperations::new(conn.clone());

    // SET
    let key = "it_key_1";
    let value = vec![1u8, 2, 3, 4];
    ops.set(key, value.clone(), 60).await.expect("set");

    // GET
    let got = ops.get(key).await.expect("get");
    assert_eq!(got, Some(value));

    // DELETE
    ops.delete(key).await.expect("delete");
    let got_after = ops.get(key).await.expect("get after delete");
    assert_eq!(got_after, None);
}

#[tokio::test]
async fn statistics_and_events() {
    let Some(url) = get_database_url() else {
        eprintln!("Skipping DB test: DATABASE_URL not set");
        return;
    };

    let Ok(conn) = DbConnection::new(&url, 5).await else {
        eprintln!("Skipping DB test: cannot connect");
        return;
    };
    conn.init_tables().await.expect("init tables");

    let ops = DbOperations::new(conn.clone());

    // Record stats
    let stats = CacheStats {
        total_requests: 100,
        total_hits: 80,
        total_misses: 20,
        hit_rate: 0.8,
        avg_latency_ms: 1.5,
        recorded_at: chrono::Utc::now(),
    };
    ops.record_stats(&stats).await.expect("record stats");

    // Log event
    ops.log_event("hit", "it_key_2", 123, 2)
        .await
        .expect("log event");
}

#[tokio::test]
async fn batch_set_get_flow() {
    let Some(url) = get_database_url() else {
        eprintln!("Skipping DB test: DATABASE_URL not set");
        return;
    };

    let Ok(conn) = DbConnection::new(&url, 5).await else {
        eprintln!("Skipping DB test: cannot connect");
        return;
    };
    conn.init_tables().await.expect("init tables");

    let ops = DbOperations::new(conn.clone());

    let entries = vec![
        ("batch_key_1".to_string(), vec![10u8; 4], 120),
        ("batch_key_2".to_string(), vec![20u8; 4], 120),
        ("batch_key_3".to_string(), vec![30u8; 4], 120),
    ];

    let affected = ops.set_batch(&entries).await.expect("set batch");
    assert!(affected >= 3);

    let keys: Vec<String> = entries.iter().map(|(k, _, _)| k.clone()).collect();
    let got = ops.get_batch(&keys).await.expect("get batch");

    let map: std::collections::HashMap<_, _> = got.into_iter().collect();
    for (k, v, _) in entries {
        assert_eq!(map.get(&k), Some(&v));
    }
}
