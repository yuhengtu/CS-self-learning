"""
Test Metadata Classes
Defines metadata for integration tests

This is written with genai assistance.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import tempfile
import os


@dataclass
class TestConfig:
    """
    Configuration for a test case to be passed into the test runner decorator.
    
    This is used to indicate the server configuration to use for the test.    
    """
    config_content: Optional[str] = None
    config_file: Optional[str] = None
    port: int = 8080 # Note: this port should match the port in the config file 
    binary_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set defaults"""
        if self.config_content is None and self.config_file is None:
            # Default config
            self.config_content = "listen 8080;\n"

@dataclass
class TestCase:
    """
    Represents a single test case with its metadata after registration with the test runner decorator.
    
    This is used to store the test case name, test case function ref, and configuration.
    """
    name: str
    func: Callable
    config: TestConfig
    
    def create_config_file(self) -> str:
        """Create a temporary config file for this test"""
        if self.config.config_file:
            return self.config.config_file
        
        # Create temp file with config content
        temp_fd, temp_path = tempfile.mkstemp(suffix='.conf', prefix=f'test_{self.name}_')
        assert self.config.config_content is not None
        with os.fdopen(temp_fd, 'w') as f:
            f.write(self.config.config_content)
        
        return temp_path
