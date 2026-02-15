#include "gtest/gtest.h"

#include "echo_request_handler.h"
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

// Updated flatten to take a unique_ptr
static std::string flatten(const std::unique_ptr<response>& resp) {
  std::string out;
  const auto& bufs = resp->get_bufs(); // Get bufs from the response object
  for (const auto& b : bufs) {
    const char* p = static_cast<const char*>(b.data());
    out.append(p, p + b.size());
  }
  return out;
}

// Build a minimal request with given raw + uri (defaults to /)
static request MakeRequest(const std::string& raw, const std::string& uri = "/") {
  request req;
  req.method = "GET";
  req.uri = uri;
  req.version = "1.1";
  req.raw = raw;
  return req;
}

TEST(EchoRequestHandler, DefaultNameIsEcho) {
  EchoRequestHandler handler;  // default ctor
  EXPECT_EQ(handler.name(), handler_types::ECHO_HANDLER);
}

TEST(EchoRequestHandler, CustomNameIsReturned) {
  EchoRequestHandler handler("my_echo");
  EXPECT_EQ(handler.name(), "my_echo");
}

TEST(EchoRequestHandler, HandlesRequestAndMirrorsRawWithHeaders) {
  EchoRequestHandler handler("tester");
  // response resp; // Changed
  std::unique_ptr<response> resp; // To hold the result

  const std::string raw =
      "GET /hello HTTP/1.1\r\n"
      "Host: localhost\r\n"
      "X-Id: 42\r\n"
      "\r\n";
  auto req = MakeRequest(raw, "/hello");

  // Call the new method
  resp = handler.handle_request(req);
  ASSERT_NE(resp, nullptr); // Check that we got a response

  const std::string resp_str = flatten(resp); // Pass the unique_ptr

  EXPECT_NE(resp_str.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
  EXPECT_NE(resp_str.find("Content-Type: text/plain"), std::string::npos);

  // (All assertions on resp_str remain the same)
  const auto cl_pos = resp_str.find(kContentLengthHeader);
  ASSERT_NE(cl_pos, std::string::npos);
  const auto cl_end = resp_str.find(kCrlf, cl_pos);
  ASSERT_NE(cl_end, std::string::npos);
  const auto len = std::stoul(resp_str.substr(cl_pos + kContentLengthHeader.length(), 
                                               cl_end - (cl_pos + kContentLengthHeader.length())));
  EXPECT_EQ(len, raw.size());

  const auto body_pos = resp_str.find(kDoubleCrlf);
  ASSERT_NE(body_pos, std::string::npos);
  EXPECT_EQ(resp_str.substr(body_pos + kDoubleCrlf.length()), raw);
}

TEST(EchoRequestHandler, HandlesEmptyRawBody) {
  EchoRequestHandler handler;
  std::unique_ptr<response> resp; // To hold the result
  
  auto req = MakeRequest("", "/empty");

  // Call the new method
  resp = handler.handle_request(req);
  ASSERT_NE(resp, nullptr);

  const std::string resp_str = flatten(resp);
  EXPECT_NE(resp_str.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
  EXPECT_NE(resp_str.find("Content-Type: text/plain"), std::string::npos);

  const auto cl_pos = resp_str.find(kContentLengthHeader);
  ASSERT_NE(cl_pos, std::string::npos);
  const auto cl_end = resp_str.find(kCrlf, cl_pos);
  ASSERT_NE(cl_end, std::string::npos);
  const auto len = std::stoul(resp_str.substr(cl_pos + kContentLengthHeader.length(), 
                                               cl_end - (cl_pos + kContentLengthHeader.length())));
  EXPECT_EQ(len, 0u);

  const auto body_pos = resp_str.find(kDoubleCrlf);
  ASSERT_NE(body_pos, std::string::npos);
  EXPECT_TRUE(resp_str.substr(body_pos + kDoubleCrlf.length()).empty());
}

TEST(EchoRequestHandler, HandlesBinaryPayload) {
  EchoRequestHandler handler;
  std::unique_ptr<response> resp; // To hold the result

  std::string raw = "GET /raw HTTP/1.1\r\nHost: localhost\r\n\r\n";
  raw.append("\x01\x02\x03", 3);

  auto req = MakeRequest(raw, "/raw");
  
  // Call the new method
  resp = handler.handle_request(req);
  ASSERT_NE(resp, nullptr);

  const std::string resp_str = flatten(resp);
  const auto body_pos = resp_str.find(kDoubleCrlf);
  ASSERT_NE(body_pos, std::string::npos);
  const std::string body = resp_str.substr(body_pos + kDoubleCrlf.length());
  EXPECT_EQ(body.size(), raw.size());
  EXPECT_TRUE(std::equal(body.begin(), body.end(), raw.begin()));
}