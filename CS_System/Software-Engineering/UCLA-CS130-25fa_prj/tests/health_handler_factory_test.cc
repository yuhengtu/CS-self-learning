#include "gtest/gtest.h"

#include <memory>
#include <string>

#include "handler_registry.h"
#include "handler_types.h"
#include "request.h"
#include "response.h"

namespace {

// Health request from remote
request MakeRequest() {
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

} // namespace

TEST(HealthHandlerFactoryTest, CreatesWorkingHandler) {
    HandlerSpec spec;
    spec.name = "my_health";
    spec.path = "/health";
    spec.type = handler_types::HEALTH_HANDLER;

    auto factory = HandlerRegistry::CreateFactory(spec);
    ASSERT_NE(factory, nullptr);

    auto handler = factory->create(spec.path, spec.path);
    ASSERT_NE(handler, nullptr);
    EXPECT_EQ(handler->name(), spec.name);

    auto req = MakeRequest(); // standard request to /health from a remote client

    std::unique_ptr<response> resp = handler->handle_request(req);
    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 200 OK\r\n");
    EXPECT_EQ(resp->get_content(), "OK");
}
