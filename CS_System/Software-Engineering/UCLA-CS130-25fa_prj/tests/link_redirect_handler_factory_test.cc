#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "handler_registry.h"
#include "handler_types.h"
#include "request.h"
#include "response.h"
#include "link_redirect_handler_factory.h"
#include "link_redirect_request_handler.h"
#include "link_manager_provider.h"

namespace {

request MakeRequest(const std::string& uri, const std::string& raw) {
    request req;
    req.method = "GET";
    req.uri = uri;
    req.version = "1.1";
    req.raw = raw;
    return req;
}

}  // namespace

TEST(LinkRedirectHandlerFactoryTest, CreatesWorkingHandler) {
    // Arrange handler spec
    HandlerSpec spec;
    spec.name = "link_redirect";
    spec.path = "/l";
    spec.type = handler_types::LINK_REDIRECT_HANDLER;

    // Required option
    spec.options["data_path"] = "/tmp/test_links";

    RegisterLinkRedirectHandlerFactory();

    auto factory = HandlerRegistry::CreateFactory(spec);
    ASSERT_NE(factory, nullptr);

    auto handler = factory->create(spec.path, spec.path);
    ASSERT_NE(handler, nullptr);

    EXPECT_EQ(handler->name(), spec.name);
}
