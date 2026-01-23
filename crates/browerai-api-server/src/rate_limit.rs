use dashmap::DashMap;
use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use std::net::IpAddr;
use std::num::NonZeroU32;
use std::sync::Arc;

/// Rate limiter configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Global requests per second
    pub global_rps: u32,
    /// Per-IP requests per second
    pub per_ip_rps: u32,
    /// Per-endpoint requests per second (optional)
    pub per_endpoint_rps: Option<u32>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            global_rps: 10000, // 10K RPS global
            per_ip_rps: 100,   // 100 RPS per IP
            per_endpoint_rps: None,
        }
    }
}

/// Rate limiter for API requests
pub struct RequestRateLimiter {
    global_limiter: RateLimiter<NotKeyed, InMemoryState, DefaultClock>,
    ip_limiters: Arc<DashMap<IpAddr, RateLimiter<NotKeyed, InMemoryState, DefaultClock>>>,
    config: RateLimitConfig,
}

impl RequestRateLimiter {
    /// Create a new rate limiter with configuration
    pub fn new(config: RateLimitConfig) -> Self {
        let global_quota = Quota::per_second(
            NonZeroU32::new(config.global_rps).unwrap_or(NonZeroU32::new(10000).unwrap()),
        );

        Self {
            global_limiter: RateLimiter::direct(global_quota),
            ip_limiters: Arc::new(DashMap::new()),
            config,
        }
    }

    /// Check if request is allowed globally
    pub fn check_global(&self) -> bool {
        self.global_limiter.check().is_ok()
    }

    /// Check if request from IP is allowed
    pub fn check_ip(&self, ip: IpAddr) -> bool {
        let limiter = self.ip_limiters.entry(ip).or_insert_with(|| {
            let quota = Quota::per_second(
                NonZeroU32::new(self.config.per_ip_rps).unwrap_or(NonZeroU32::new(100).unwrap()),
            );
            RateLimiter::direct(quota)
        });

        limiter.check().is_ok()
    }

    /// Check if both global and IP limits are respected
    pub fn check(&self, ip: IpAddr) -> bool {
        self.check_global() && self.check_ip(ip)
    }

    /// Get number of tracked IPs
    pub fn ip_count(&self) -> usize {
        self.ip_limiters.len()
    }

    /// Clear old IP entries (for cleanup)
    pub fn cleanup_old_ips(&self, max_ips: usize) {
        if self.ip_limiters.len() > max_ips {
            // Remove roughly half the entries
            let to_remove = (max_ips / 2) as i32;
            let mut count = 0;
            self.ip_limiters.retain(|_, _| {
                count += 1;
                count > to_remove
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_global_rate_limit() {
        let config = RateLimitConfig {
            global_rps: 10,
            per_ip_rps: 100,
            per_endpoint_rps: None,
        };
        let limiter = RequestRateLimiter::new(config);

        // Should allow first 10 requests
        for _ in 0..10 {
            assert!(limiter.check_global());
        }

        // 11th should fail
        assert!(!limiter.check_global());
    }

    #[test]
    fn test_per_ip_rate_limit() {
        let config = RateLimitConfig {
            global_rps: 1000,
            per_ip_rps: 5,
            per_endpoint_rps: None,
        };
        let limiter = RequestRateLimiter::new(config);
        let ip = IpAddr::from_str("192.168.1.1").unwrap();

        // Should allow first 5 requests
        for _ in 0..5 {
            assert!(limiter.check_ip(ip));
        }

        // 6th should fail
        assert!(!limiter.check_ip(ip));
    }

    #[test]
    fn test_different_ips() {
        let config = RateLimitConfig {
            global_rps: 1000,
            per_ip_rps: 5,
            per_endpoint_rps: None,
        };
        let limiter = RequestRateLimiter::new(config);

        let ip1 = IpAddr::from_str("192.168.1.1").unwrap();
        let ip2 = IpAddr::from_str("192.168.1.2").unwrap();

        // Both IPs should have independent limits
        for _ in 0..5 {
            assert!(limiter.check_ip(ip1));
            assert!(limiter.check_ip(ip2));
        }

        // Both should be rate limited
        assert!(!limiter.check_ip(ip1));
        assert!(!limiter.check_ip(ip2));
    }
}
