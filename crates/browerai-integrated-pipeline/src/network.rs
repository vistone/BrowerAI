//! 网络模块 - 获取和缓存网页

use anyhow::Result;
use std::path::PathBuf;

#[allow(dead_code)]
pub struct NetworkManager {
    cache_dir: PathBuf,
    timeout_secs: u64,
}

impl NetworkManager {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            timeout_secs: 30,
        }
    }

    pub async fn fetch(&self, url: &str) -> Result<String> {
        log::debug!("获取: {}", url);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .build()?;

        let response = client.get(url).send().await?;
        let html = response.text().await?;

        Ok(html)
    }
}
