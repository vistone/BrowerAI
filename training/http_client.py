"""
Real HTTP Client with Retry Logic and Timeout Handling
Replaces simulation with actual network requests
"""

import time
import requests
from typing import Any, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry as UrllibRetry
import logging


logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Request timeout error"""
    pass


class ConnectionError(Exception):
    """Connection error"""
    pass


class HttpClientConfig:
    """HTTP Client Configuration"""
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5000",
        timeout: float = 5.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        pool_connections: int = 10,
        pool_maxsize: int = 10
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.pool_connections = pool_connections
        self.pool_maxsize = pool_maxsize


class RealHttpClient:
    """Real HTTP client with retry logic and timeout handling"""

    def __init__(self, config: Optional[HttpClientConfig] = None):
        """Initialize HTTP client with configuration"""
        self.config = config or HttpClientConfig()
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = UrllibRetry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        
        # Apply retry strategy to HTTP and HTTPS
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.config.pool_connections,
            pool_maxsize=self.config.pool_maxsize
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(
            f"Initialized HttpClient: {self.config.base_url} "
            f"(timeout={self.config.timeout}s, retries={self.config.max_retries})"
        )

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        retry: bool = True
    ) -> requests.Response:
        """
        Perform GET request with timeout and retry handling
        
        Args:
            endpoint: API endpoint path (e.g., "/api/v1/health")
            params: Query parameters
            retry: Whether to retry on failure
            
        Returns:
            Response object
            
        Raises:
            TimeoutError: If request times out
            ConnectionError: If connection fails after retries
        """
        url = f"{self.config.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.config.timeout
            )
            duration = (time.time() - start_time) * 1000
            
            logger.debug(
                f"GET {endpoint} → {response.status_code} "
                f"({duration:.2f}ms)"
            )
            
            return response
            
        except requests.Timeout as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"GET {endpoint} timeout after {duration:.2f}ms")
            raise TimeoutError(f"Request timeout for {endpoint}: {str(e)}")
            
        except requests.ConnectionError as e:
            logger.error(f"GET {endpoint} connection error: {str(e)}")
            raise ConnectionError(f"Connection error for {endpoint}: {str(e)}")
            
        except Exception as e:
            logger.error(f"GET {endpoint} error: {type(e).__name__}: {str(e)}")
            raise

    def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        retry: bool = True
    ) -> requests.Response:
        """
        Perform POST request with timeout and retry handling
        
        Args:
            endpoint: API endpoint path (e.g., "/api/v1/generate")
            json: JSON data to send
            data: Form data to send
            retry: Whether to retry on failure
            
        Returns:
            Response object
            
        Raises:
            TimeoutError: If request times out
            ConnectionError: If connection fails after retries
        """
        url = f"{self.config.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            response = self.session.post(
                url,
                json=json,
                data=data,
                timeout=self.config.timeout
            )
            duration = (time.time() - start_time) * 1000
            
            logger.debug(
                f"POST {endpoint} → {response.status_code} "
                f"({duration:.2f}ms)"
            )
            
            return response
            
        except requests.Timeout as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"POST {endpoint} timeout after {duration:.2f}ms")
            raise TimeoutError(f"Request timeout for {endpoint}: {str(e)}")
            
        except requests.ConnectionError as e:
            logger.error(f"POST {endpoint} connection error: {str(e)}")
            raise ConnectionError(f"Connection error for {endpoint}: {str(e)}")
            
        except Exception as e:
            logger.error(f"POST {endpoint} error: {type(e).__name__}: {str(e)}")
            raise

    def health_check(self) -> bool:
        """
        Check if server is healthy
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = self.get("/api/v1/health", retry=False)
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """Close HTTP session"""
        self.session.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class HttpClientMetrics:
    """Track HTTP client metrics"""
    
    def __init__(self):
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.timeout_count = 0
        self.retry_count = 0
        self.latencies = []
        self.errors = []

    def record_request(self, duration_ms: float, status_code: int, error: Optional[str] = None):
        """Record request metrics"""
        self.request_count += 1
        self.latencies.append(duration_ms)
        
        if 200 <= status_code < 300:
            self.success_count += 1
        else:
            self.error_count += 1
            
        if error:
            self.errors.append(error)
            if "timeout" in error.lower():
                self.timeout_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.latencies:
            return {}
            
        import statistics
        return {
            "total_requests": self.request_count,
            "successful": self.success_count,
            "failed": self.error_count,
            "timeouts": self.timeout_count,
            "success_rate": f"{(self.success_count / self.request_count * 100):.1f}%",
            "avg_latency_ms": f"{statistics.mean(self.latencies):.2f}",
            "min_latency_ms": f"{min(self.latencies):.2f}",
            "max_latency_ms": f"{max(self.latencies):.2f}",
            "median_latency_ms": f"{statistics.median(self.latencies):.2f}",
            "error_count": len(self.errors)
        }
