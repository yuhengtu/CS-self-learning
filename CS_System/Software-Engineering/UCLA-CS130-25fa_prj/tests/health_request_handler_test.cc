#include "gtest/gtest.h"

#include "health_request_handler.h"
#include "response.h"
#include "request.h"

#include <string>
#include <string_view>
#include <vector>
#include <memory> // Added for std::unique_ptr
#include <boost/asio.hpp>
#include "handler_types.h"

constexpr std::string_view kCrlf = "\r\n";
constexpr std::string_view kDoubleCrlf = "\r\n\r\n";
constexpr std::string_view kContentLengthHeader = "Content-Length: ";

static std::string flatten(const std::unique_ptr<response>& resp) {
  std::string out;
  const auto& bufs = resp->get_bufs(); // Get bufs from the response object
  for (const auto& b : bufs) {
    const char* p = static_cast<const char*>(b.data());
    out.append(p, p + b.size());
  }
  return out;
}

request MakeRequest() { // example request for /health
  request req;
  req.method = "GET";
  req.uri = "/health";
  req.version = "1.1";
  req.raw =
      "GET /health HTTP/1.1\r\n"
      "Host: localhost\r\n"
      "\r\n";
  return req;
}

TEST(HealthRequestHandler, DefaultNameIsHealth) {
  HealthRequestHandler handler;  // default ctor
  EXPECT_EQ(handler.name(), handler_types::HEALTH_HANDLER);
}

TEST(HealthRequestHandler, CustomNameIsReturned) {
  HealthRequestHandler handler("my_health");
  EXPECT_EQ(handler.name(), "my_health");
}

TEST(healthRequestHandler, HandlesRequestCorrectly) {
  HealthRequestHandler handler("tester");
  
  std::unique_ptr<response> resp; // To hold the result

  auto req = MakeRequest();

  // Call the new method
  resp = handler.handle_request(req);
  ASSERT_NE(resp, nullptr); // Check that we got a response

  const std::string resp_str = flatten(resp); // Pass the unique_ptr

  EXPECT_NE(resp_str.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
  EXPECT_NE(resp_str.find("Content-Type: text/plain"), std::string::npos);
  EXPECT_EQ(resp->get_content(), "OK");
}
