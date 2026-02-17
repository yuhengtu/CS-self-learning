This project is done with wonderful collaborators: Leo Liu, Justin Mealey, and Simon Traub, and our AIs.

# Learning the Codebase

## Entrypoint
main.cc is the entrypoint of the program, it will:
- Parse the config file passed as a command line argument
  - Parsing logic is in config_parser.cc
- Build the server config based on tokens from input config file
  - Server config logic is in server_config.cc
- Create a new server object with the server config, start_accept() called in constructor

## Accepting new connections
In server.cc in start_accept():
- Using Boost, asynchronously accept incoming client connections
- Use the handle_accept() callback once a connection is found
- handle_accept() calls session's start() method

## Session logic
Once a session is start()ed, it instantly calls handle_read(). This sets up the following loop:
- Async read from buffer, once this completes handle_read_callback() is called
- Using the RequestParser object 'parse_', parse the incoming request. This does the following:
  - Parses the request character by character using a finite state machine
  - Based on the request, returns either PROPER_REQUEST, IN_PROGRESS, or BAD_REQUEST
- Once we get a status, handle it accordingly:
  - if IN_PROGRESS, go back to handle_read as full HTTP request is not completed
  - if BAD_REQUEST:
    - Use the 'dispatcher_' object to handle the bad request: (in dispatcher.cc)
      - Using the ResponseBuilder, create and return the HTTP 400 response for a bad request
    - Call handle_write() now that the m_response member variable has been filled
  - if PROPER_REQUEST:
    - Same as BAD_REQUEST case, but also reset member variables for next request
- Once handle_write() is called:
  - Use Boost async_write() to write the m_response out to client, handle_write_callback() is called
- handle_write_callback() logs any errors and shuts down the HTTP session

## Miscellaneous
- the Dispatcher constructor builds a vector of routes, where a route is a location (URI) and a factory to the corresponding request handler
- the Dispatch() method that gets called when a proper request is found does the following:
  - Look at each route, if it finds a prefix match, build the corresponding handler via 
  the factory and call handler->handle_request(req)
- Requests are formed via a Request object (request.cc) that has all of the components of a proper HTTP request (method, URI, version, body, etc.)
- Responses are built via a ResponseBuilder object (response_builder.cc) which creates a response object (response.cc) that contains a vector of 'boost::asio::const_buffer' for the status line, headers, and content of the response

# Actions

## Development Setup (Boost JSON)

The URL shortener manager uses Boost.JSON (Boost >= 1.75) for DTO serialization.
Install the following packages in your dev environment:

```bash
sudo apt-get update && sudo apt-get install -y \
  libboost-json-dev \
  libboost-filesystem-dev \
  libboost-thread-dev \
  libboost-log-dev \
  libboost-system-dev \
  libssl-dev
```

Our base Docker image installs these via `docker/base.Dockerfile`.

## Quick Start

Build and run locally:

```bash
cd build
cmake ..
make
bin/webserver ../my_config
```

In another terminal (inside the development environment):

```bash
curl -v http://localhost:8080/
```

Common issues:
- Not connecting the client to the CS130 development environment

## Coverage

```bash
rm -rf build build_coverage
mkdir build_coverage && cd build_coverage
cmake -DCMAKE_BUILD_TYPE=Coverage ..
make -j$(nproc)
ctest --output-on-failure
make coverage
```

The coverage report is written to `build_coverage/report/index.html`.

## Docker

Build and run:

```bash
docker build -f docker/base.Dockerfile -t semiconductor-savants:base .
docker build -f docker/coverage.Dockerfile -t semiconductor-savants:coverage .
docker build -f docker/Dockerfile -t webserver:latest .
docker run -p 8080:80 --rm webserver:latest
```

In another terminal (outside the dev environment):

```bash
curl -v localhost:8080
```

Common issues:
- Not running these commands from the project root
- "docker: Error response from daemon: ... Bind for 0.0.0.0:8080 failed: port is already allocated"
  - Another instance of this container is already running. Use `docker ps` to find its ID, then `docker stop <ID>`.

## Release

Automated release:

```bash
chmod +x release_latest.sh
./release_latest.sh
```

Manual inspection (skip pulling the image):

```bash
chmod +x release_latest.sh
./release_latest.sh --no-pull
```

Common issue:
- Not authenticating to GCR before attempting to pull/push an image.

## Cloud Build

```bash
gcloud builds submit --project semiconductor-savants --config docker/cloudbuild.yaml .
```

## Restart VM with Latest Release

```bash
gcloud compute instances update-container web-server \
  --zone=us-west1-b \
  --project=semiconductor-savants \
  --container-image=gcr.io/semiconductor-savants/semiconductor-savants:release
```

## Usage

For configuration format see: `docs/config/README.md`

## Testing

- Integration testing docs: `tests/integration/README.md`

Run integration tests (from the build directory):

```bash
../tests/integration/run_integration_tests.sh
```

With options:

```bash
../tests/integration/run_integration_tests.sh --verbose           # Show server logs on failure
../tests/integration/run_integration_tests.sh --test basic        # Run tests matching "basic"
```

They are also included in `make test`.

To add tests, add functions in `tests/integration/test_cases.py`. See the integration README for details.


## URL Shortener

The web server includes a full-featured URL shortening service with RESTful management APIs, automatic redirects, visit tracking, and optional password protection.

### Overview

The URL shortener consists of three integrated handler types:

1. **Link Management Handler** (`link_manage`) - RESTful API for creating, reading, updating, and deleting short links
2. **Link Redirect Handler** (`link_redirect`) - Serves HTTP redirects for short codes
3. **Analytics Handler** (`analytics`) - Provides visit statistics and metrics

All three handlers share a common filesystem-based storage layer via the `LinkManager` component, ensuring consistent state across concurrent requests.

### Architecture

**Storage Model:**
- Short codes are generated as base62-encoded sequential integers, ensuring uniqueness and URL-safe characters
- Each short link is stored as a JSON file: `<data_path>/urls/<code>.json`
- A counter file tracks the next available code: `<data_path>/urls/.counter`
- Visit statistics are aggregated per destination URL in `<data_path>/urls/url_stats.json`

**Concurrency:**
- Thread-safe in-process synchronization via mutex locks in the `LinkManager`
- Atomic file writes using temp files and rename operations
- Multiple concurrent requests are handled safely

**Password Protection:**
- Optional per-link password protection for management operations
- Uses SHA256 hashing with random 16-byte salts
- Passwords never stored in plaintext
- Authentication via `Link-Password` HTTP header

### Configuration

Configure all three handlers with the same `data_path` to share storage:

```nginx
server {
  listen 8080;

  # Management API
  location /api/link {
    handler link_manage;
    data_path /var/lib/webserver/links;
  }

  # Redirect endpoint
  location /l {
    handler link_redirect;
    data_path /var/lib/webserver/links;
  }

  # Analytics
  location /analytics {
    handler analytics;
    data_path /var/lib/webserver/links;
  }
}
```

### Creating Short Links

Create a short link by sending a POST request with the target URL:

```bash
curl -X POST http://localhost:8080/api/link \
     -H "Content-Type: application/json" \
     -d '{"url":"https://cs130.org"}'
```

Response:
```json
{"code":"abc123"}
```

**With Password Protection:**

To protect management operations (GET, PUT, DELETE) with a password:

```bash
curl -X POST http://localhost:8080/api/link \
     -H "Content-Type: application/json" \
     -d '{"url":"https://cs130.org","password":"secret"}'
```

The password is salted and hashed using SHA256 before storage. All subsequent management operations on this link will require the `Link-Password` header.

### Serving Redirects

Once created, short links are immediately available for redirection:

```bash
# Test the redirect (returns HTTP 302 with Location header)
curl -i http://localhost:8080/l/abc123

# Follow the redirect automatically
curl -L http://localhost:8080/l/abc123
```

**How Redirects Work:**
1. Client requests `GET /l/<code>`
2. Handler looks up the code in storage
3. If found, returns `HTTP 302 Found` with `Location: <target-url>`
4. Visit counters are incremented (per-code and per-URL aggregates)
5. Client's browser automatically follows the redirect

### Managing Links

**Read Link Details:**
```bash
curl http://localhost:8080/api/link/abc123
```

**Update Target URL:**
```bash
curl -X PUT http://localhost:8080/api/link/abc123 \
     -H "Content-Type: application/json" \
     -d '{"url":"https://updated.example.com"}'
```

**Delete a Link:**
```bash
curl -X DELETE http://localhost:8080/api/link/abc123
```

**With Password Protection:**

If the link was created with a password, include the `Link-Password` header:

```bash
# Read
curl -H "Link-Password: secret" http://localhost:8080/api/link/abc123

# Update
curl -H "Link-Password: secret" -X PUT http://localhost:8080/api/link/abc123 \
     -H "Content-Type: application/json" \
     -d '{"url":"https://newexample.com"}'

# Delete
curl -H "Link-Password: secret" -X DELETE http://localhost:8080/api/link/abc123
```

Missing or incorrect passwords return `403 Forbidden`.

### Analytics and Visit Tracking

**Per-Code Statistics:**
```bash
# Get visit count for a specific short code
curl http://localhost:8080/analytics/abc123
```

Response:
```json
{
  "code": "abc123",
  "url": "https://cs130.org",
  "visits": 42,
  "url_visits": 150
}
```

- `visits` - how many times this specific short code was accessed
- `url_visits` - aggregate visits across all codes pointing to the same destination URL

**Top URLs:**
```bash
# Get top 5 most visited destination URLs
curl http://localhost:8080/analytics/top/5
```

Response:
```json
[
  {"url": "https://popular.site", "visits": 1000},
  {"url": "https://another.site", "visits": 500}
]
```

**Note:** Analytics endpoints do not require password authentication. Only management operations (GET/PUT/DELETE on `/api/link/<code>`) require the password for protected links.

### Complete Example Workflow

1. **Create a password-protected short link:**
   ```bash
   curl -X POST http://localhost:8080/api/link \
        -H "Content-Type: application/json" \
        -d '{"url":"https://example.com/important-page","password":"mySecret123"}'
   ```
   Response: `{"code":"xyz789"}`

2. **Access the short link (no password needed for redirects):**
   ```bash
   curl -L http://localhost:8080/l/xyz789
   # Redirects to https://example.com/important-page
   ```

3. **Check analytics:**
   ```bash
   curl http://localhost:8080/analytics/xyz789
   # Shows visit counts
   ```

4. **Update the link (requires password):**
   ```bash
   curl -H "Link-Password: mySecret123" -X PUT http://localhost:8080/api/link/xyz789 \
        -H "Content-Type: application/json" \
        -d '{"url":"https://example.com/updated-page"}'
   ```

5. **Delete the link (requires password):**
   ```bash
   curl -H "Link-Password: mySecret123" -X DELETE http://localhost:8080/api/link/xyz789
   ```

### Additional Documentation

For detailed API specifications and handler implementation details:
- **Link Management Handler:** `docs/handlers/link-manage-handler/LINK_MANAGE_HANDLER.md`
- **Analytics Handler:** `docs/handlers/analytics-handler/ANALYTICS_HANDLER.md`

## Serve Static Files (Cloud Shell)

```bash
tools/env/start.sh <USER> -- -p 127.0.0.1:8080:8080
# build the webserver, cd to semiconductor-savants
build/bin/webserver my_config
```

Open another terminal and run:

```bash
curl -i http://localhost:8080/static/hello.html
curl -v http://localhost:8080/echo
```

Click "Web Preview" of Cloud Shell → Preview on port 8080.

Then test: `static/hello.html`, `static/hello.jpg`, `static/hello.txt`, `static/hello.zip`.

## Adding a Handler Factory (Dev Guide)

The server uses a registry of handler factories. The dispatcher holds factories per route and creates short‑lived handlers per request.

Steps to add a new handler type `Foo`:

1) Implement the handler
- Create `include/foo_request_handler.h` and `src/foo_request_handler.cc` as a `RequestHandler` subclass.
- Example: `FooRequestHandler(const std::string& location, /* typed args */);`

2) Implement the factory
- `include/foo_handler_factory.h`:

```cpp
#include "request_handler_factory.h"
#include "server_config.h"

class FooHandlerFactory : public RequestHandlerFactory {
public:
  explicit FooHandlerFactory(const HandlerSpec& spec);
  std::unique_ptr<RequestHandler> create(const std::string& location,
                                         const std::string& url) override;
private:
  int ttl_ = 0; // example option
};

void RegisterFooHandlerFactory();
```

- `src/foo_handler_factory.cc`:

```cpp
#include "foo_handler_factory.h"
#include "handler_registry.h"
#include "handler_types.h" // add handler_types::FOO
#include "logger.h"

FooHandlerFactory::FooHandlerFactory(const HandlerSpec& spec) {
  if (auto it = spec.options.find("ttl"); it != spec.options.end()) {
    ttl_ = std::stoi(it->second);
  }
}

std::unique_ptr<RequestHandler> FooHandlerFactory::create(const std::string& location,
                                                          const std::string& url) {
  return std::unique_ptr<RequestHandler>(/* new FooRequestHandler(location, ttl_) */);
}

void RegisterFooHandlerFactory() {
  HandlerRegistry::Register(handler_types::FOO,
    [](const HandlerSpec& spec) -> std::unique_ptr<RequestHandlerFactory> {
      // Validate required options; log + return nullptr on failure
      return std::unique_ptr<RequestHandlerFactory>(new FooHandlerFactory(spec));
    });
}
```

3) Register with built‑ins
- In `src/handler_registry.cc`, include your factory header and call `RegisterFooHandlerFactory()` inside `HandlerRegistry::RegisterBuiltins()`.

4) Wire into CMake
- Add new `.cc` files to `request_handler_lib` sources in `CMakeLists.txt`.

5) Tests
- Add unit tests like `tests/factory_test.cc`:
  - Happy path: `CreateFactory` returns a factory; `factory->create(...)` returns a working handler.
  - Edge cases: missing required options → `CreateFactory` returns `nullptr`.

Notes:
- Factory responsibilities and contract:
  - Factories receive a `HandlerSpec` containing: `name` (label for logging purposes), `path` (location/prefix), `type`, and an `options` map of key/value strings for additional arguments.
  - Factories parse ONLY their own options from `HandlerSpec` (typically in the factory constructor) and store typed members (e.g., ints, strings, enums).
  - Handler constructors do not perform config parsing; they receive already‑typed arguments from the factory.
  - The factory’s `create(location, url)` uses stored members and the provided `location` to construct a short‑lived handler instance.
- Dispatcher is type‑agnostic; routing uses longest‑prefix match.

## Server CRUD Handler & Crud Configuration
CRUD Handler
The server includes a CRUD (Create, Read, Update, Delete) handler for managing JSON entities with persistent file-based storage.
Documentation: See CRUD_HANDLER.md for complete API reference, configuration, and usage examples.
Configuration: Server configs for integration tests are found under tests/integration/test_configs.
