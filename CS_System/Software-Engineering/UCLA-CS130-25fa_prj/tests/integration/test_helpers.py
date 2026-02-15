"""
Test Helpers for Integration Tests
Provides utilities for making HTTP requests and assertions

Written with genai assistance.
"""

import requests
import time
from typing import Optional, Dict, Any


class TestHelpers:
    """Helper utilities for integration tests"""
    
    def __init__(self, base_url: str = "http://localhost", port: int = 8080, timeout=(5, 10)):
        self.base_url = base_url.rstrip('/')
        self.port = port
        self.session = requests.Session()
        self.timeout = timeout
    
    def _get_base_url(self):
        """Get the base URL for the test helpers"""
        return f"{self.base_url}:{self.port}"
    
    def get(self, path: str, **kwargs) -> requests.Response:
        """Make a GET request"""
        url = f"{self._get_base_url()}{path}"
        kwargs = self._set_default_timeout(kwargs)
        return self.session.get(url, **kwargs)
    
    def post(self, path: str, data: Any = None, **kwargs) -> requests.Response:
        """Make a POST request"""
        url = f"{self._get_base_url()}{path}"
        kwargs = self._set_default_timeout(kwargs)
        return self.session.post(url, data=data, **kwargs)
    
    def put(self, path: str, data: Any = None, **kwargs) -> requests.Response:
        """Make a PUT request"""
        url = f"{self._get_base_url()}{path}"
        kwargs = self._set_default_timeout(kwargs)
        return self.session.put(url, data=data, **kwargs)
    
    def delete(self, path: str, **kwargs) -> requests.Response:
        """Make a DELETE request"""
        url = f"{self._get_base_url()}{path}"
        kwargs = self._set_default_timeout(kwargs)
        return self.session.delete(url, **kwargs)
    
    def head(self, path: str, **kwargs) -> requests.Response:
        """Make a HEAD request"""
        url = f"{self._get_base_url()}{path}"
        kwargs = self._set_default_timeout(kwargs)
        return self.session.head(url, **kwargs)
    
    def assert_status_code(self, response: requests.Response, expected: int) -> bool:
        """Assert that response has expected status code"""
        if response.status_code != expected:
            print(f"Expected status {expected}, got {response.status_code}")
            return False
        return True
    
    def assert_contains(self, response: requests.Response, text: str) -> bool:
        """Assert that response contains expected text"""
        if text not in response.text:
            print(f"Expected text '{text}' not found in response")
            print(f"Response content: {response.text[:200]}...")
            return False
        return True
    
    def assert_header(self, response: requests.Response, header: str, value: str) -> bool:
        """Assert that response has expected header value"""
        actual_value = response.headers.get(header)
        if actual_value != value:
            print(f"Expected header {header}: {value}, got: {actual_value}")
            return False
        return True
    
    def wait_for_server(self, max_attempts: int = 30, delay: float = 0.1) -> bool:
        """Wait for server to be ready by polling"""
        for _ in range(max_attempts):
            try:
                response = self.get("/", timeout=(1, 1))  # Use tuple format
                return True
            except requests.exceptions.RequestException:
                time.sleep(delay)
        return False
    
    def _set_default_timeout(self, kwargs):
        """Set the default timeout for the request if it's not already set."""
        kwargs.setdefault("timeout", self.timeout)
        return kwargs
