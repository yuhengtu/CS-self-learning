"""
Server Fixture for Integration Tests
Manages the lifecycle of the webserver process during testing.

Includes the server setup and tear down as well as managing server logs.

This test fixture is written with genai assistance.
"""

import subprocess
import time
import os
import signal
import tempfile
import shutil
from typing import Optional, Tuple
import socket

class ServerFixture:
    """Manages webserver process lifecycle for integration tests"""
    
    def __init__(self, binary_path: str, config_path: str, port: int = 8080):
        '''
        Note: the server's port should be specified in the config file, but the port arg is to know which socket to do a health check / conenction check to know that the server is running.
        
        '''
        self.binary_path = binary_path
        self.config_path = config_path
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.temp_dir = None
        self.server_log = None
        
    def start(self, timeout: int = 3) -> bool:
        """Start the webserver process"""
        if self.process is not None:
            raise RuntimeError("Server is already running")
            
        # Create temp directory for server logs
        self.temp_dir = tempfile.mkdtemp(prefix="integration_test_")
        self.server_log = os.path.join(self.temp_dir, "server.log")
        
        # check that config path exists or fail fast, log with real path for debugging
        if not os.path.exists(self.config_path):
            raise RuntimeError(f"Config file not found: {self.config_path}. Real path: {os.path.abspath(self.config_path)}")        
        
        try:
            # Start the server process
            with open(self.server_log, 'w') as log_file:
                self.process = subprocess.Popen(
                    [self.binary_path, os.path.abspath(self.config_path)],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(self.binary_path)
                )
            
            # Wait for server to start up
            return self._wait_for_startup(timeout)
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to start server: {e}")
    
    def stop(self):
        """Stop the webserver process"""
        if self.process is None:
            return
            
        try:
            # Try graceful shutdown first
            self.process.terminate()
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown fails
            self.process.kill()
            self.process.wait()
        finally:
            self.process = None
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def is_running(self) -> bool:
        """Check if server is running"""
        if self.process is None:
            return False
        return self.process.poll() is None
    
    def get_logs(self) -> str:
        """Get server output logs"""
        if not self.server_log or not os.path.exists(self.server_log):
            return ""
        with open(self.server_log, 'r') as f:
            return f.read()
    
    def _wait_for_startup(self, timeout: int = 3) -> bool:
        """Wait for server to start up and be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self.is_running():
                return False
                
            # Periodically poll to connect to the server to see successful startup.
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', self.port))
                sock.close()
                
                if result == 0:
                    return True
            except:
                pass
                
            time.sleep(0.1)
        
        return False
    
    # Context manager methods for with statement
    
    def __enter__(self):
        if not self.start():
            raise RuntimeError("Failed to start server")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
