use anyhow::Result;
use deadpool_redis::redis;
use tokio::time::{timeout, Duration};
use tracing::{debug, warn};

/// 通过 Sentinel 解析主节点地址。
pub async fn resolve_master_addr(
    sentinels: &[String],
    master_name: &str,
    op_timeout: Duration,
) -> Result<Option<String>> {
    for url in sentinels {
        let client = match redis::Client::open(url.as_str()) {
            Ok(c) => c,
            Err(e) => {
                warn!(sentinel = %url, %e, "Failed to open sentinel client");
                continue;
            }
        };
        let mut conn = match client.get_multiplexed_tokio_connection().await {
            Ok(c) => c,
            Err(e) => {
                warn!(sentinel = %url, %e, "Failed to connect sentinel");
                continue;
            }
        };
        let res = timeout(op_timeout, async {
            // SENTINEL get-master-addr-by-name <master-name>
            let reply: redis::RedisResult<Vec<String>> = redis::cmd("SENTINEL")
                .arg("get-master-addr-by-name")
                .arg(master_name)
                .query_async(&mut conn)
                .await;
            reply
        })
        .await;
        match res {
            Ok(Ok(vec)) if vec.len() >= 2 => {
                let host = &vec[0];
                let port = &vec[1];
                let addr = format!("redis://{}:{}", host, port);
                debug!(sentinel = %url, master = %master_name, addr = %addr, "Resolved master address from sentinel");
                return Ok(Some(addr));
            }
            Ok(Ok(_)) => {
                warn!(sentinel = %url, master = %master_name, "Sentinel returned invalid address vector");
            }
            Ok(Err(e)) => {
                warn!(sentinel = %url, %e, "Sentinel command failed");
            }
            Err(_) => {
                warn!(sentinel = %url, "Sentinel query timeout");
            }
        }
    }
    Ok(None)
}
