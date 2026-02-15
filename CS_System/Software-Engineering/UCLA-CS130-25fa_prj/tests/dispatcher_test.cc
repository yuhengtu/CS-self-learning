#include "gtest/gtest.h"

#include "dispatcher.h"
#include "response.h"
#include "response_builder.h"
#include "request.h"
#include "request_handler.h"
#include "echo_request_handler.h"
#include "request_handler_factory.h"
#include "server_config.h"

#include <fstream>
#include <sys/stat.h>
#include "handler_types.h"
#include <memory>

namespace {

request MakeRequest(std::string raw, std::string uri = "/") {
    request req;
    req.method = "GET";
    req.uri = std::move(uri);
    req.version = "1.1";
    req.raw = raw;
    return req;
}
// RecordingHandler is a lightweight request handler used only by these tests to
// capture which request URI would reach a route. Verifies that the
// dispatcher calls the expected handler so that routing logic can be tested.
class RecordingHandler : public RequestHandler {
public:
    RecordingHandler(std::string id, std::string* last_uri_out)
        : id_(std::move(id)), last_uri_out_(last_uri_out) {}

    std::unique_ptr<response> handle_request(const request& req) override {
        if (last_uri_out_) *last_uri_out_ = req.uri;
        
        // Must allocate and return a response
        auto out = std::make_unique<response>();
        ResponseBuilder::createNotFound().build(*out); // Return 404 by default
        return out;
    }

    std::string name() const override { return id_; }

private:
    std::string id_;
    std::string* last_uri_out_;
};

class RecordingFactory : public RequestHandlerFactory {
public:
    explicit RecordingFactory(std::string id) : id_(std::move(id)) {}
    std::unique_ptr<RequestHandler> create(const std::string& /*location*/, const std::string& /*url*/) override {
        return std::make_unique<RecordingHandler>(id_, &last_uri);
    }
    std::string last_uri;
private:
    std::string id_;
};

class EchoFactory : public RequestHandlerFactory {
public:
    explicit EchoFactory(std::string name) : name_(std::move(name)) {}
    std::unique_ptr<RequestHandler> create(const std::string& /*location*/, const std::string& /*url*/) override {
        return std::make_unique<EchoRequestHandler>(name_);
    }
private:
    std::string name_;
};

// Factory that intentionally returns nullptr to exercise error paths
class NullFactory : public RequestHandlerFactory {
public:
    std::unique_ptr<RequestHandler> create(const std::string&, const std::string&) override {
        return nullptr; // simulate factory failure
    }
};

}  // namespace

TEST(DispatcherTest, RoutesToEchoHandler) {
    std::vector<Dispatcher::RouteInit> routes;
    Dispatcher::RouteInit route;
    route.location = "/";
    route.factory = std::unique_ptr<RequestHandlerFactory>(new EchoFactory("test_route"));
    routes.emplace_back(std::move(route));
    Dispatcher disp(std::move(routes));
    
    std::unique_ptr<response> resp; // To hold the result
    auto req = MakeRequest(
        "GET /file HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "\r\n",
        "/file");

    resp = disp.Dispatch(req); // This line needed the semicolon
    ASSERT_NE(resp, nullptr);
    // We don't check the enum result anymore, just the content
    EXPECT_EQ(resp->get_content(), req.raw);
}

TEST(DispatcherTest, BadRequestHelperProducesStock400) {
    std::vector<Dispatcher::RouteInit> routes;
    Dispatcher disp(std::move(routes));
    std::unique_ptr<response> resp; // To hold the result

    resp = disp.HandleBadRequest();
    ASSERT_NE(resp, nullptr);

    const auto status = resp->get_status_line();
    EXPECT_EQ(status, "HTTP/1.1 400 Bad Request\r\n");
}

TEST(DispatcherTest, UnmatchedRouteReturns404) {
    std::vector<Dispatcher::RouteInit> routes;
    Dispatcher::RouteInit route;
    auto rec_factory = std::unique_ptr<RecordingFactory>(new RecordingFactory("api"));
    RecordingFactory* rec_ptr = rec_factory.get();
    route.location = "/api";
    route.factory = std::move(rec_factory);
    routes.emplace_back(std::move(route));
    Dispatcher disp(std::move(routes));

    std::unique_ptr<response> resp; // To hold the result
    auto req = MakeRequest("GET /static HTTP/1.1\r\n\r\n", "/static");
    
    resp = disp.Dispatch(req);
    ASSERT_NE(resp, nullptr);
    // Just check the status line
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 404 Not Found\r\n");
}

TEST(DispatcherTest, LongestPrefixMatchPickedFirst) {
    auto factory_short = std::unique_ptr<RecordingFactory>(new RecordingFactory("short"));
    auto factory_long = std::unique_ptr<RecordingFactory>(new RecordingFactory("long"));
    RecordingFactory* factory_long_ptr = factory_long.get();

    std::vector<Dispatcher::RouteInit> routes;
    Dispatcher::RouteInit short_route; short_route.location = "/static"; short_route.factory = std::move(factory_short);
    Dispatcher::RouteInit long_route; long_route.location = "/static/images"; long_route.factory = std::move(factory_long);
    routes.emplace_back(std::move(short_route));
    routes.emplace_back(std::move(long_route));

    Dispatcher disp(std::move(routes));
    std::unique_ptr<response> resp; // To hold the result
    auto req = MakeRequest("GET /static/images/logo.png HTTP/1.1\r\n\r\n",
                           "/static/images/logo.png");

    resp = disp.Dispatch(req); 
    ASSERT_NE(resp, nullptr);
    ASSERT_NE(factory_long_ptr, nullptr);
    EXPECT_EQ(factory_long_ptr->last_uri, "/static/images/logo.png");
}

TEST(DispatcherTest, FactoryBuildsFromHandlerSpecs) {
    HandlerSpec spec;
    spec.name = handler_types::ECHO_HANDLER;
    spec.path = "/api";
    spec.type = handler_types::ECHO_HANDLER;
    std::vector<HandlerSpec> specs = {spec};

    auto dispatcher = Dispatcher::FromSpecs(specs);
    ASSERT_NE(dispatcher, nullptr);

    std::unique_ptr<response> resp; // To hold the result
    auto req = MakeRequest("GET /api/status HTTP/1.1\r\n\r\n", "/api/status");
    
    resp = dispatcher->Dispatch(req); 
    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_content(), req.raw);
}

TEST(DispatcherTest, EmptyRoutesReturn404) {
    // Construct a dispatcher with no routes at all
    std::vector<Dispatcher::RouteInit> routes;
    Dispatcher disp(std::move(routes));

    std::unique_ptr<response> resp; // To hold the result
    auto req = MakeRequest("GET /any HTTP/1.1\r\n\r\n", "/any");
    
    resp = disp.Dispatch(req); 
    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 404 Not Found\r\n");
}

TEST(DispatcherTest, FactorySkipsUnknownTypeAndBuildsFromValidOnes) {
    // One unknown handler (should be skipped) and one valid echo
    HandlerSpec bad;
    bad.name = "mystery";
    bad.path = "/mystery";
    bad.type = "unknown_type";

    HandlerSpec good;
    good.name = "echo_handler";
    good.path = "/echo";
    good.type = handler_types::ECHO_HANDLER;

    std::vector<HandlerSpec> specs = {bad, good};

    auto dispatcher = Dispatcher::FromSpecs(specs);
    ASSERT_NE(dispatcher, nullptr) << "Should still build with at least one valid handler";

    std::unique_ptr<response> resp; // To hold the result
    auto req = MakeRequest("GET /echo/ping HTTP/1.1\r\n\r\n", "/echo/ping");
    
    resp = dispatcher->Dispatch(req); 
    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_content(), req.raw);
}

TEST(DispatcherTest, StaticHandlerMissingRootIsRejectedButEchoStillWorks) {
    // One bad static (missing root) and one good echo
    HandlerSpec bad_static;
    bad_static.name = "bad_static";
    bad_static.path = "/assets";
    bad_static.type = handler_types::STATIC_HANDLER;
    // no options["root"]

    HandlerSpec echo;
    echo.name = "echo_ok";
    echo.path = "/echo";
    echo.type = handler_types::ECHO_HANDLER;

    auto dispatcher = Dispatcher::FromSpecs({bad_static, echo});
    ASSERT_NE(dispatcher, nullptr);

    std::unique_ptr<response> resp; // To hold the result
    auto req = MakeRequest("GET /echo/ok HTTP/1.1\r\n\r\n", "/echo/ok");
    
    resp = dispatcher->Dispatch(req); 
    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_content(), req.raw);

    std::unique_ptr<response> resp2; // To hold the result
    auto req2 = MakeRequest("GET /assets/logo.jpg HTTP/1.1\r\n\r\n", "/assets/logo.jpg");
    
    resp2 = dispatcher->Dispatch(req2); 
    ASSERT_NE(resp2, nullptr);
    EXPECT_EQ(resp2->get_status_line(), "HTTP/1.1 404 Not Found\r\n");
}

TEST(DispatcherTest, AllInvalidSpecsInjectDefaultNotFoundAtRoot) {
    // Both specs are invalid: unknown type and static missing 'root'.
    HandlerSpec bad_unknown;
    bad_unknown.name = "bad1";
    bad_unknown.path = "/bad1";
    bad_unknown.type = "does_not_exist";

    HandlerSpec bad_static_missing_root;
    bad_static_missing_root.name = "bad2";
    bad_static_missing_root.path = "/bad2";
    bad_static_missing_root.type = handler_types::STATIC_HANDLER;
    // no options["root"]

    auto dispatcher = Dispatcher::FromSpecs({bad_unknown, bad_static_missing_root});
    ASSERT_NE(dispatcher, nullptr) << "Factory should still succeed by injecting not_found at '/'";

    // Since all handlers were invalid, the factory should have added a default not_found at "/".
    auto req = MakeRequest("GET /anything HTTP/1.1\r\n\r\n", "/anything");
    std::unique_ptr<response> resp = dispatcher->Dispatch(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 404 Not Found\r\n");
}

TEST(DispatcherTest, ConstructorSkipsNullHandlersAndReturns404) {
    // Build a routes vector that includes an entry with a null handler.
    std::vector<Dispatcher::RouteInit> routes;
    Dispatcher::RouteInit null_route;
    null_route.location = "/null";
    null_route.factory.reset();  // explicitly null
    routes.emplace_back(std::move(null_route));
    // The constructor should skip this entry and end up with an empty routing table (no routes).
    Dispatcher disp(std::move(routes));

    std::unique_ptr<response> resp; // To hold the result
    auto req = MakeRequest("GET /null/test HTTP/1.1\r\n\r\n", "/null/test");
    
    resp = disp.Dispatch(req); 
    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 404 Not Found\r\n");
}

// If a matching route's factory returns nullptr and there is NO '/' route,
// Dispatch() should log an error and return a 500.
TEST(DispatcherTest, MatchingRouteFactoryReturnsNullWithoutRootReturns500) {
    std::vector<Dispatcher::RouteInit> routes;
    Dispatcher::RouteInit bad;
    bad.location = "/api";
    bad.factory = std::unique_ptr<RequestHandlerFactory>(new NullFactory());
    routes.emplace_back(std::move(bad));

    Dispatcher disp(std::move(routes));

    auto req = MakeRequest("GET /api/ping HTTP/1.1\r\n\r\n", "/api/ping");
    std::unique_ptr<response> resp = disp.Dispatch(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 500 Internal Server Error\r\n");
}

// If a matching route's factory returns nullptr and there IS a '/' route,
// Dispatch() should fall back to the root handler.
TEST(DispatcherTest, MatchingRouteFactoryReturnsNullFallsBackToRootHandler) {
    std::vector<Dispatcher::RouteInit> routes;

    // Bad route that matches but can't create a handler
    Dispatcher::RouteInit bad;
    bad.location = "/api";
    bad.factory = std::unique_ptr<RequestHandlerFactory>(new NullFactory());
    routes.emplace_back(std::move(bad));

    // Root route that works (Echo)
    Dispatcher::RouteInit root;
    root.location = "/";
    root.factory = std::unique_ptr<RequestHandlerFactory>(new EchoFactory("root_echo"));
    routes.emplace_back(std::move(root));

    Dispatcher disp(std::move(routes));

    auto req = MakeRequest("GET /api/bad HTTP/1.1\r\n\r\n", "/api/bad");
    std::unique_ptr<response> resp = disp.Dispatch(req);

    ASSERT_NE(resp, nullptr);
    // Root Echo should echo the raw request
    EXPECT_EQ(resp->get_content(), req.raw);
}

// FromSpecs should inject a default NotFound at '/' if no explicit root is given.
// Verify that a '/' request is then handled as 404 by the injected handler.
TEST(DispatcherTest, FromSpecsInjectsNotFoundWhenRootMissingHandlesSlash) {
    HandlerSpec echo_only;
    echo_only.name = "echo_only";
    echo_only.path = "/echo";
    echo_only.type = handler_types::ECHO_HANDLER;

    auto dispatcher = Dispatcher::FromSpecs({echo_only});
    ASSERT_NE(dispatcher, nullptr);

    // Hit root '/' — should be handled by injected not_found handler
    auto req = MakeRequest("GET / HTTP/1.1\r\n\r\n", "/");
    std::unique_ptr<response> resp = dispatcher->Dispatch(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 404 Not Found\r\n");
}

// If an explicit root is provided in specs, FromSpecs must NOT inject NotFound.
// Verify that root uses the provided root handler (Echo here).
TEST(DispatcherTest, FromSpecsWithExplicitRootUsesProvidedRootHandler) {
    HandlerSpec root_echo;
    root_echo.name = "root_echo";
    root_echo.path = "/";
    root_echo.type = handler_types::ECHO_HANDLER;

    HandlerSpec echo_sub;
    echo_sub.name = "echo_sub";
    echo_sub.path = "/echo";
    echo_sub.type = handler_types::ECHO_HANDLER;

    auto dispatcher = Dispatcher::FromSpecs({root_echo, echo_sub});
    ASSERT_NE(dispatcher, nullptr);

    // Request to '/' should be handled by the explicit root echo handler
    auto req = MakeRequest("GET / HTTP/1.1\r\n\r\n", "/");
    std::unique_ptr<response> resp = dispatcher->Dispatch(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_content(), req.raw);
}