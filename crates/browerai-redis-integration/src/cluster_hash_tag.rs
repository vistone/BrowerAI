/// Redis 集群哈希标签工具，用于在批量操作中保证同 slot 的键聚集。
///
/// Redis Cluster 中，键通过 CRC16 哈希映射到 16384 个 slot。
/// 哈希标签规则（RFC）：仅 {} 之间的部分参与哈希计算。
/// 例如：
/// - "key{user:123}" 与 "data{user:123}" 映射到同一 slot（哈希值基于 "user:123"）
/// - "prefix{bucket1}" 与 "counter{bucket1}" 同 slot
pub struct ClusterHashTag;

impl ClusterHashTag {
    /// 为键应用哈希标签前缀。
    ///
    /// 如果 `tag` 非空，则返回 `<key>{<tag>}`；否则返回原键。
    /// 这确保所有使用同一 `tag` 的键会映射到同一 slot。
    ///
    /// # 示例
    /// ```ignore
    /// let key1 = ClusterHashTag::apply("user:1:name", Some("user:1"));
    /// let key2 = ClusterHashTag::apply("user:1:email", Some("user:1"));
    /// // key1 = "user:1:name{user:1}"
    /// // key2 = "user:1:email{user:1}"
    /// // 两者映射到同一 slot，跨键批量操作（MSET/MGET）可原子执行
    /// ```
    pub fn apply(key: &str, tag: Option<&str>) -> String {
        match tag {
            Some(t) => format!("{}{{{}}}", key, t),
            None => key.to_string(),
        }
    }

    /// 批量为键列表应用同一标签。
    pub fn apply_batch(keys: &[String], tag: Option<&str>) -> Vec<String> {
        keys.iter().map(|k| Self::apply(k, tag)).collect()
    }

    /// 移除键中的哈希标签，恢复原始键。
    ///
    /// 例如 "user:1:name{user:1}" 返回 "user:1:name"。
    pub fn strip(key: &str) -> &str {
        if let Some(brace_pos) = key.rfind('{') {
            &key[..brace_pos]
        } else {
            key
        }
    }

    /// 提取键中的哈希标签（如果存在）。
    ///
    /// 例如 "user:1:name{user:1}" 返回 Some("user:1")。
    pub fn extract(key: &str) -> Option<&str> {
        if let Some(open) = key.rfind('{') {
            if let Some(close) = key.rfind('}') {
                if close > open {
                    return Some(&key[open + 1..close]);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_tag_apply() {
        let key = ClusterHashTag::apply("user:1:name", Some("user:1"));
        assert_eq!(key, "user:1:name{user:1}");

        let key_no_tag = ClusterHashTag::apply("key", None);
        assert_eq!(key_no_tag, "key");
    }

    #[test]
    fn test_hash_tag_apply_batch() {
        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let tagged = ClusterHashTag::apply_batch(&keys, Some("bucket"));
        assert_eq!(
            tagged,
            vec![
                "a{bucket}".to_string(),
                "b{bucket}".to_string(),
                "c{bucket}".to_string()
            ]
        );
    }

    #[test]
    fn test_hash_tag_strip() {
        let key = ClusterHashTag::strip("user:1:name{user:1}");
        assert_eq!(key, "user:1:name");

        let key_no_tag = ClusterHashTag::strip("simple_key");
        assert_eq!(key_no_tag, "simple_key");
    }

    #[test]
    fn test_hash_tag_extract() {
        let tag = ClusterHashTag::extract("user:1:name{user:1}");
        assert_eq!(tag, Some("user:1"));

        let no_tag = ClusterHashTag::extract("simple_key");
        assert_eq!(no_tag, None);
    }
}
