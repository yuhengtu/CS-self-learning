### SLEEP Handler

The sleep handler simulates slow requests to test concurrency/performance. It can sleep for a configurable duration before returning a simple text response.

#### Configuration

Add a sleep handler location to your config:

```nginx
location /sleep {
  handler sleep;
  sleep_ms 2000; # optional; default is 2000 (2 seconds)
}
```

Options:
- `sleep_ms` (optional): sleep duration in milliseconds. Use a lower value in unit tests (e.g., 50) to keep test suites fast.

#### Behavior
- Always returns `200 OK` with body `SLEPT` and `Content-Type: text/plain`.
- Useful for verifying that other requests can complete while this handler is sleeping (e.g., to test multithreading).

#### Unit Testing Tip
- Override `sleep_ms` to a small value (e.g., 50 or 100) when testing locally or in CI to avoid slowing down test suites.

