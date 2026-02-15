#!/usr/bin/env python3
"""
Integration Test Main Entry Point
Main entry point for running integration tests

This test runner is written with genai assistance.
"""

import os
import sys
import argparse

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from test_runner import IntegrationTestRunner
import test_cases


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Webserver Integration Tests")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    parser.add_argument("--test", "-t", 
                       help="Run specific test by name")
    parser.add_argument("--binary", 
                       default="bin/webserver",
                       help="Path to webserver binary (relative to build dir)")
    
    args = parser.parse_args()
    
    # Check if binary exists
    if not os.path.exists(args.binary):
        print(f"ERROR: Webserver binary not found at {args.binary}")
        print("Build the project first:")
        print("  mkdir -p build && cd build")
        print("  cmake .. && make")
        sys.exit(1)
    
    # Create test runner
    runner = IntegrationTestRunner(args.binary, args.verbose)
    
    # Add all registered test cases
    runner.add_registered_tests(test_filter=args.test)
    
    # Run tests
    success = runner.run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
