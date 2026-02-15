#include "gtest/gtest.h"
#include "not_found_request_handler.h"
#include "not_found_handler_factory.h"
#include "handler_registry.h"
#include "handler_types.h"
#include "request.h"
#include "response.h"
#include "server_config.h" // For HandlerSpec

#include <string>
#include <memory>
#include <boost/asio.hpp>

namespace {

// Helper to flatten the response buffers into a single string
static std::string flatten(const std::unique_ptr<response>& resp) {
  std::string out;
  const auto& bufs = resp->get_bufs();
  for (const auto& b : bufs) {
    const char* p = static_cast<const char*>(b.data());
    out.append(p, p + b.size());
  }
  return out;
}

// Helper to create a minimal request for a given URI
static request MakeRequest(const std::string& uri) {
  request req;
  req.method = "GET";
  req.uri = uri;
  req.version = "1.1";
  req.raw = "GET " + uri + " HTTP/1.1\r\nHost: test\r\n\r\n";
  return req;
}

// Test the handler directly
TEST(NotFoundHandlerTest, HandlerReturns404) {
  // Test with default name
  NotFoundRequestHandler handler;
  EXPECT_EQ(handler.name(), handler_types::NOT_FOUND_HANDLER);

  auto req = MakeRequest("/some/missing/page");
  std::unique_ptr<response> resp = handler.handle_request(req);

  ASSERT_NE(resp, nullptr);
  
  std::string resp_str = flatten(resp);
  
  // Check for 404 status line
  EXPECT_NE(resp_str.find("HTTP/1.1 404 Not Found"), std::string::npos);
  
  // Check that our custom body message is included
  EXPECT_NE(resp_str.find("The requested resource could not be found."), std::string::npos);
}

// Test the factory and its registration with the HandlerRegistry
TEST(NotFoundHandlerTest, FactoryIntegrationTest) {
  // This test relies on RegisterBuiltins() being called by CreateFactory
  // which in turn should call RegisterNotFoundHandlerFactory().

  HandlerSpec spec;
  spec.name = "my_custom_404"; // Use a custom name
  spec.path = "/";
  spec.type = handler_types::NOT_FOUND_HANDLER;

  // 1. Test Factory Creation
  auto factory = HandlerRegistry::CreateFactory(spec);
  ASSERT_NE(factory, nullptr);

  // 2. Test Handler Creation
  auto handler = factory->create("/", "/");
  ASSERT_NE(handler, nullptr);

  // Check that the custom name from the spec was used
  EXPECT_EQ(handler->name(), "my_custom_404");

  // 3. Test the handler that the factory created
  auto req = MakeRequest("/any/path/will/do");
  std::unique_ptr<response> resp = handler->handle_request(req);

  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 404 Not Found\r\n");
}

// Test repeated calls to ensure consistent 404 behavior
TEST(NotFoundHandlerTest, MultipleRequestsReturnConsistent404) {
  NotFoundRequestHandler handler;

  // Simulate several different missing URLs
  std::vector<std::string> uris = {
      "/missing1",
      "/does/not/exist",
      "/abc/123/zzz",
      "/something/else"
  };

  for (const auto& uri : uris) {
    auto req = MakeRequest(uri);
    std::unique_ptr<response> resp = handler.handle_request(req);

    ASSERT_NE(resp, nullptr) << "Response should not be null for URI: " << uri;

    std::string resp_str = flatten(resp);

    // Verify 404 status
    EXPECT_NE(resp_str.find("HTTP/1.1 404 Not Found"), std::string::npos)
        << "Expected 404 for URI: " << uri;

    // Verify the standard not-found message is included
    EXPECT_NE(resp_str.find("The requested resource could not be found."), std::string::npos)
        << "Expected error message for URI: " << uri;
  }
}

} // namespace