#!/usr/bin/env python3
"""
Integration Test Runner Class
Handles test execution and reporting

Usage:
1. Tests are registered when test_cases.py is imported (via decorators)
2. Create runner instance: runner = IntegrationTestRunner(binary_path, verbose)
3. Add registered tests: runner.add_registered_tests(test_filter="optional_substring")
4. Run tests: runner.run_tests()

Test Registration Flow:
- @IntegrationTestRunner.register_test(TestConfig(...)) decorates test functions
- When test_cases.py is imported, decorators execute and register tests in _test_registry
- add_registered_tests() filters registry and adds matching tests to runner's test list
- run_tests() executes all tests in the runner's test `list`

tldr: the main steps are: register with decorator, and add the tests to the runner instance's test list to run.

Written with genai assistance.
"""

import os
import sys
import traceback
from typing import List, Callable, Tuple, Dict, Optional

from server_fixture import ServerFixture
from test_helpers import TestHelpers
from test_metadata import TestConfig, TestCase


class IntegrationTestRunner:
    """Main test runner for integration tests"""
    
    # Class-level registry: all tests registered via decorators (shared across instances)
    _test_registry: Dict[str, TestCase] = {}
    
    def __init__(self, binary_path: str, verbose: bool = False):
        self.binary_path = binary_path
        self.verbose = verbose
        # Instance-level test list: tests to run for this specific runner instance
        self.tests: List[TestCase] = []
        self.temp_configs = []  # Track temp configs for cleanup
        
    def add_test(self, test_case: TestCase):
        """Add a test to the test suite"""
        self.tests.append(test_case)
    
    @classmethod
    def register_test(cls, config: Optional[TestConfig] = None, name: Optional[str] = None):
        """Decorator to register a test function with metadata"""
        def decorator(func: Callable):
            test_name = name or func.__name__
            test_config = config or TestConfig()  # Use default config
            test_case = TestCase(name=test_name, func=func, config=test_config)
            cls._test_registry[test_name] = test_case
            return func
        return decorator
    
    def add_registered_tests(self, test_filter: Optional[str] = None):
        """Add all registered tests to the test suite"""
        for name, test_case in self._test_registry.items():
            if test_filter is None or test_filter in name:
                self.add_test(test_case)
    
    def run_tests(self) -> bool:
        """Run all tests and return success status"""
        if not self.tests:
            print("No tests to run")
            return True
        
        print(f"Running {len(self.tests)} integration test(s)...")
        print("=" * 50)
        
        passed = 0
        failed = 0
        
        for test_case in self.tests:
            print(f"\nRunning test: {test_case.name}")
            print("-" * 30)
            
            try:
                # Create config file for this test
                config_path = test_case.create_config_file()
                
                # Only track temp files for cleanup (not static config files)
                if test_case.config.config_file is None:
                    self.temp_configs.append(config_path)
                
                # Create server fixture with test-specific config
                server = ServerFixture(
                    test_case.config.binary_path or self.binary_path,
                    config_path, 
                    test_case.config.port
                )
                
                # Create test helpers with test-specific port
                helpers = TestHelpers(
                    base_url="http://localhost",
                    port=test_case.config.port
                )
                
                # Run the test, using the server fixture context for setup and teardown
                with server:
                    success = test_case.func(server, helpers)
                
                if success:
                    print(f". {test_case.name} PASSED")
                    passed += 1
                else:
                    print(f"X {test_case.name} FAILED")
                    failed += 1
                    if self.verbose and server.get_logs():
                        print("Server logs:")
                        print(server.get_logs())
                        
            except Exception as e:
                print(f"X {test_case.name} ERROR: {e}")
                if self.verbose:
                    traceback.print_exc()
                failed += 1
        
        # Cleanup temp config files
        self._cleanup_temp_configs()
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"Test Results: {passed} passed, {failed} failed")
        
        return failed == 0
    
    def _cleanup_temp_configs(self):
        """Clean up temporary config files"""
        for config_path in self.temp_configs:
            try:
                if os.path.exists(config_path):
                    os.unlink(config_path)
            except OSError:
                pass
        self.temp_configs.clear()
