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

// Flatten reply buffers to a single string
std::string Flatten(const std::unique_ptr<response>& resp) {
    std::string out;
    const auto& bufs = resp->get_bufs();
    for (const auto& b : bufs) {
        const char* p = static_cast<const char*>(b.data());
        out.append(p, p + b.size());
    }
    return out;
}

} // namespace

TEST(StaticHandlerFactoryTest, CreatesHandlerHappyPath) {
    HandlerSpec spec;
    spec.name = "assets";
    spec.path = "/static";
    spec.type = handler_types::STATIC_HANDLER;
    // CMake sets WORKING_DIRECTORY to ${CMAKE_CURRENT_SOURCE_DIR}/tests
    spec.options["root"] = "static_test_files";

    auto factory = HandlerRegistry::CreateFactory(spec);
    ASSERT_NE(factory, nullptr);

    auto handler = factory->create(spec.path, spec.path + std::string("/note.txt"));
    ASSERT_NE(handler, nullptr);
    EXPECT_EQ(handler->name(), handler_types::STATIC_HANDLER);

    std::unique_ptr<response> resp; // To hold the result
    auto req = MakeRequest("/static/note.txt",
        "GET /static/note.txt HTTP/1.1\r\nHost: localhost\r\n\r\n");

    resp = handler->handle_request(req);
    ASSERT_NE(resp, nullptr);

    const std::string resp_str = Flatten(resp);
    EXPECT_NE(resp_str.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
    EXPECT_NE(resp_str.find("Content-Type: text/plain"), std::string::npos);
}

TEST(StaticHandlerFactoryTest, MissingRootReturnsNullFactory) {
    HandlerSpec spec;
    spec.name = "bad_static";
    spec.path = "/assets";
    spec.type = handler_types::STATIC_HANDLER;
    // no options["root"]

    auto factory = HandlerRegistry::CreateFactory(spec);
    EXPECT_EQ(factory, nullptr);
}