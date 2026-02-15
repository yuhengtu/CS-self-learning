#include "sleep_request_handler.h"
#include "request.h"
#include "response.h"

#include <gtest/gtest.h>
#include <chrono>
#include <string>

// Basic sanity test: constructor + handle_request produce a valid HTTP 200 response
// with the expected headers and body.
TEST(SleepRequestHandlerTest, ReturnsOkWithExpectedBodyAndHeaders) {
    SleepRequestHandler handler("test_sleep_instance", std::chrono::milliseconds(10));

    request req;   // SleepRequestHandler ignores the request contents
    auto resp = handler.handle_request(req);

    ASSERT_NE(resp, nullptr);

    // Check status line
    std::string status_line = resp->get_status_line();
    // Expected something like: "HTTP/1.1 200 OK\r\n"
    EXPECT_NE(status_line.find("HTTP/1.1"), std::string::npos);
    EXPECT_NE(status_line.find("200"), std::string::npos);
    EXPECT_NE(status_line.find("OK"), std::string::npos);

    // Check body
    std::string body = resp->get_content();
    EXPECT_EQ(body, "SLEPT");

    // Check headers contain Content-Type and Content-Length
    std::string headers = resp->get_headers();
    EXPECT_NE(headers.find("Content-Type: text/plain"), std::string::npos);

    // Content-Length should match body size
    std::string content_length_str = "Content-Length: " + std::to_string(body.size());
    EXPECT_NE(headers.find(content_length_str), std::string::npos);

    // Connection header should be set to close by ResponseBuilder
    EXPECT_NE(headers.find("Connection: close"), std::string::npos);
}

// Timing test: ensure that handle_request actually sleeps for approximately
// the duration specified in the implementation (2 seconds).
// We don't require exact 2000 ms, but we check a reasonable window.
TEST(SleepRequestHandlerTest, SleepsForExpectedDuration) {
    SleepRequestHandler handler("timing_instance", std::chrono::milliseconds(50));

    request req;

    auto start = std::chrono::steady_clock::now();
    auto resp = handler.handle_request(req);
    auto end = std::chrono::steady_clock::now();

    ASSERT_NE(resp, nullptr);

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Allow some slack for scheduling / CI load.
    EXPECT_GE(duration_ms, 40);
    EXPECT_LT(duration_ms, 500);
}
