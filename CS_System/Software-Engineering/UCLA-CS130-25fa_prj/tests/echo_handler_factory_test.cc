#include "gtest/gtest.h"

#include <memory>
#include <string>

#include "handler_registry.h"
#include "handler_types.h"
#include "request.h"
#include "response.h"

namespace {

request MakeRequest(const std::string& uri, const std::string& raw) {
    request req;
    req.method = "GET";
    req.uri = uri;
    req.version = "1.1";
    req.raw = raw;
    return req;
}

} // namespace

TEST(EchoHandlerFactoryTest, CreatesWorkingHandler) {
    HandlerSpec spec;
    spec.name = "my_echo";
    spec.path = "/echo";
    spec.type = handler_types::ECHO_HANDLER;

    auto factory = HandlerRegistry::CreateFactory(spec);
    ASSERT_NE(factory, nullptr);

    auto handler = factory->create(spec.path, spec.path);
    ASSERT_NE(handler, nullptr);
    EXPECT_EQ(handler->name(), spec.name);

    std::unique_ptr<response> resp; // To hold result
    const std::string payload = "GET /echo HTTP/1.1\r\nHost: localhost\r\n\r\n";
    auto req = MakeRequest("/echo", payload);

    resp = handler->handle_request(req);
    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_content(), payload);
}