# Integration Test Framework

This directory contains the integration test framework for the webserver project.

Written with genai assistance.

## Overview

The integration testing framework provides a decorator-based system for writing integration tests of the webserver. Tests automatically start a webserver instance, send HTTP requests, verify responses, and clean up resources.

Tests also manage temp files for server logs which are outputted on failure for debugging purposes.

## Prerequisites
(All included in CS130 Dev Docker container)

Integration tests require Python3.

**Dependencies:**
- `requests` library 

**Verify requests is installed:**
```bash
pip freeze | grep requests
# or check requirements.txt
```

**If not in Docker container, set up a virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r tests/integration/requirements.txt
```

## Usage

### Running Tests

As usual build
**From build directory (via CMake):**
```bash
cd build
cmake ..
make
make test
```

Run from build directory with
```bash
../tests/integration/run_integration_tests.sh
```

**With options:**
```bash
../tests/integration/run_integration_tests.sh --verbose           # Show server logs on failure
../tests/integration/run_integration_tests.sh --test basic        # Run tests matching "basic" (i.e. basic is in the test name)
```

### Test Reporting

- **`make test`**: The integration test is just one monolithic test (pass/fail)
- **Shell script**: Provides detailed test-by-test breakdown with pass/fail status

### Adding New Tests

1. Open `test_cases.py`
2. Add a new test function:
   ```python
   @IntegrationTestRunner.register_test(TestConfig(port=8080))
   def test_my_feature(server, helpers):
       """Test description"""
       response = helpers.get("/my-endpoint")
       return helpers.assert_status_code(response, 200)
   ```
3. Run the tests to verify

`TestConfig` is used to store test metadata:
- `config_file`: Path to server config file (relative to working directory, or use absolute path)
  - From `make test`: relative to `build/`, e.g., `../tests/integration/test_configs/myconfig.conf`
- `config_content`: Config file content as string (generates temp file automatically)
- `port`: Port number (default: 8080) - must match the port in config file
  - This is for the test client to know where to send requests
  - Not passed to webserver binary (binary reads port from config)
  - Should align with `listen` directive in config
- `name`: Override test name (default: function name)

**Test function signature:**
- `server`: ServerFixture instance (provides server management), 
- `helpers`: TestHelpers instance (provides HTTP request utilities and test asserts)
- **Return**: `True` if test passes, `False` if test fails

## Internals

### Architecture

The framework consists of several components:

#### 1. Server Fixture (`server_fixture.py`)

Manages the webserver process lifecycle:
- **Setup**: Starts the webserver binary with a specified config file
- **Teardown**: Stops the server and cleans up resources
- **Context Manager**: Use with `with server:` for automatic cleanup
- **Logging**: Captures server stdout/stderr for debugging

```python
server = ServerFixture(binary_path, config_path, port)
with server:
    # Server is running
    # Make requests
    pass
# Server is stopped and cleaned up
```

#### 2. Test Registration (`test_cases.py`)

Tests are declared using a decorator that registers them in a class-level registry:

```python
@IntegrationTestRunner.register_test(TestConfig(port=8080))
def test_basic_http_request(server, helpers):
    # Test implementation
    pass
```

**What happens:**
1. Decorator executes when `test_cases.py` is imported
2. Test is registered in `IntegrationTestRunner._test_registry` (class-level)
3. Test metadata (name, config, port) is stored with the test function

#### 3. Test Runner (`test_runner.py`)

The `IntegrationTestRunner` class manages test execution:

**Class-level registry:**
- `_test_registry`: Dict of all registered tests (shared across instances)

**Instance-level test list:**
- `tests`: List of tests to run for this specific runner instance

**Test selection:**
```python
runner = IntegrationTestRunner(binary_path, verbose)
runner.add_registered_tests(test_filter="basic")  # Filter by substring
runner.run_tests()
```

#### 4. Test Metadata (`test_metadata.py`)

**TestConfig:**
- `config_content`: Config file content (dynamic generation)
- `config_file`: Path to static config file
- `port`: Port number for test client connections
- `binary_path`: Path to webserver binary (optional)

^ Test config is the decorator configuration

**TestCase:**
- `name`: Test name
- `func`: Test function reference
- `config`: TestConfig instance

^ Upon registry the tests are stored as TestCases with the config and other metadata.

### Decorator Parameters

```python
@IntegrationTestRunner.register_test(TestConfig(
    port=8080,                    # Port for HTTP requests
    config_content="listen 8080;" # Server config content
))
def test_example(server, helpers):
    pass
```

**Available parameters:**
- `port`: Port number (default: 8080)
- `config_content`: Dynamic config content (default: "listen 8080;\n")
- `config_file`: Static config file path (overrides config_content)
- `binary_path`: Override default binary path (rarely needed)
- `name`: Override test name (default: function name)

### Port Number Nuance

**Important:** There are TWO port numbers that must match:

1. **Server config port**: `listen 8080;` in the config file
   - Read by the C++ webserver binary
   - Determines which port the server listens on

2. **Test fixture port**: `TestConfig(port=8080)` in the decorator
   - Used by the Python test framework
   - Determines where HTTP requests are sent if you use the helper

**! These must match !** The test framework doesn't parse the config file to extract the port, so you must specify it in both places.

It's possible that we expose the parser via a port_parser_main down the line that the test harness calls to get the port directly from config to deprecate this field. But for the most part 8080 is the default so this may be unnecessary. 

### Test Discovery Flow

1. `integration_test.py` imports `test_runner` (defines IntegrationTestRunner class)
2. `integration_test.py` imports `test_cases` (triggers decorator execution)
3. Decorators register tests in `_test_registry`
4. Runner instance calls `add_registered_tests()` to filter and load tests
5. Runner executes tests with `run_tests()`

### CMake Integration

The integration tests are added to CMake via:

```cmake
add_test(NAME integration_tests
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/run_integration_tests.sh
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
```

**Key points:**
- Runs the bash wrapper script (handles dependencies and paths)
- Working directory is the build directory
- Returns 0 on success, 1 on failure (ctest compatible)

## File Structure

```
tests/integration/
├── run_integration_tests.sh    # Bash wrapper (entry point)
├── integration_test.py          # Main Python entry point
├── test_runner.py               # Test runner class
├── test_cases.py                # Test case definitions
├── test_metadata.py             # Test metadata classes
├── server_fixture.py            # Server lifecycle management & temp server log management
├── test_helpers.py              # HTTP request utilities
├── requirements.txt             
└── README.md                    
```

Server configs for integration tests are found under `tests/integration/test_configs`.

## Troubleshooting

**"Missing 'requests' library"**
- Install in venv: `pip install -r tests/integration/requirements.txt`
- Or use CS130 Docker container which should have requests installed already

**"Webserver binary not found"**
- Build the project: `cd build && cmake .. && make`

**"Failed to start server"
- Check if port is already in use
- Verify config file is valid
