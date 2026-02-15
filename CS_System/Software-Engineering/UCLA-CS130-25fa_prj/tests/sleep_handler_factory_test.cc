#include "gtest/gtest.h"

#include <memory>
#include <string>

#include "handler_registry.h"
#include "handler_types.h"
#include "request.h"
#include "response.h"

namespace {

request MakeRequest() {
    request req;
    req.method = "GET";
    req.uri = "/sleep";
    req.version = "1.1";
    req.raw = "GET /sleep HTTP/1.1\r\nHost: localhost\r\n\r\n";
    return req;
}

} // namespace

TEST(SleepHandlerFactoryTest, ConstructorWithDefaultSleepDuration) {
    HandlerSpec spec;
    spec.name = "my_sleep";
    spec.path = "/sleep";
    spec.type = handler_types::SLEEP_HANDLER;
    // No sleep_ms option provided, should use default (2000ms)

    auto factory = HandlerRegistry::CreateFactory(spec);
    ASSERT_NE(factory, nullptr);

    auto handler = factory->create(spec.path, spec.path);
    ASSERT_NE(handler, nullptr);
    EXPECT_EQ(handler->name(), spec.name);
}

TEST(SleepHandlerFactoryTest, ConstructorWithCustomSleepDuration) {
    HandlerSpec spec;
    spec.name = "custom_sleep";
    spec.path = "/sleep";
    spec.type = handler_types::SLEEP_HANDLER;
    spec.options["sleep_ms"] = "100";

    auto factory = HandlerRegistry::CreateFactory(spec);
    ASSERT_NE(factory, nullptr);

    auto handler = factory->create(spec.path, spec.path);
    ASSERT_NE(handler, nullptr);
    EXPECT_EQ(handler->name(), spec.name);

    // Verify handler works and responds
    auto req = MakeRequest();
    auto resp = handler->handle_request(req);
    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_content(), "SLEPT");
}

TEST(SleepHandlerFactoryTest, ConstructorWithInvalidSleepDurationUsesDefault) {
    HandlerSpec spec;
    spec.name = "invalid_sleep";
    spec.path = "/sleep";
    spec.type = handler_types::SLEEP_HANDLER;
    spec.options["sleep_ms"] = "not_a_number";

    // Should not crash, should use default duration
    auto factory = HandlerRegistry::CreateFactory(spec);
    ASSERT_NE(factory, nullptr);

    auto handler = factory->create(spec.path, spec.path);
    ASSERT_NE(handler, nullptr);
    EXPECT_EQ(handler->name(), spec.name);
}

TEST(SleepHandlerFactoryTest, ConstructorWithNegativeSleepDurationUsesDefault) {
    HandlerSpec spec;
    spec.name = "negative_sleep";
    spec.path = "/sleep";
    spec.type = handler_types::SLEEP_HANDLER;
    spec.options["sleep_ms"] = "-100";

    // Negative values should be ignored, default used
    auto factory = HandlerRegistry::CreateFactory(spec);
    ASSERT_NE(factory, nullptr);

    auto handler = factory->create(spec.path, spec.path);
    ASSERT_NE(handler, nullptr);
    EXPECT_EQ(handler->name(), spec.name);
}

TEST(SleepHandlerFactoryTest, ConstructorWithZeroSleepDurationUsesDefault) {
    HandlerSpec spec;
    spec.name = "zero_sleep";
    spec.path = "/sleep";
    spec.type = handler_types::SLEEP_HANDLER;
    spec.options["sleep_ms"] = "0";

    // Zero or negative values should be ignored
    auto factory = HandlerRegistry::CreateFactory(spec);
    ASSERT_NE(factory, nullptr);

    auto handler = factory->create(spec.path, spec.path);
    ASSERT_NE(handler, nullptr);
    EXPECT_EQ(handler->name(), spec.name);
}

TEST(SleepHandlerFactoryTest, ConstructorWithEmptyNameUsesDefaultName) {
    HandlerSpec spec;
    // spec.name left empty
    spec.path = "/sleep";
    spec.type = handler_types::SLEEP_HANDLER;

    auto factory = HandlerRegistry::CreateFactory(spec);
    ASSERT_NE(factory, nullptr);

    auto handler = factory->create(spec.path, spec.path);
    ASSERT_NE(handler, nullptr);
    // Should use handler_types::SLEEP_HANDLER as default name
    EXPECT_EQ(handler->name(), handler_types::SLEEP_HANDLER);
}

TEST(SleepHandlerFactoryTest, CreatesWorkingHandler) {
    HandlerSpec spec;
    spec.name = "working_sleep";
    spec.path = "/sleep";
    spec.type = handler_types::SLEEP_HANDLER;
    spec.options["sleep_ms"] = "10"; // Short sleep for test speed

    auto factory = HandlerRegistry::CreateFactory(spec);
    ASSERT_NE(factory, nullptr);

    auto handler = factory->create(spec.path, spec.path);
    ASSERT_NE(handler, nullptr);
    EXPECT_EQ(handler->name(), spec.name);

    auto req = MakeRequest();
    auto resp = handler->handle_request(req);
    ASSERT_NE(resp, nullptr);
    
    // Verify standard sleep handler response
    EXPECT_NE(resp->get_status_line().find("200"), std::string::npos);
    EXPECT_EQ(resp->get_content(), "SLEPT");
}

