// Generated with ChatGPT
// tests/session_test.cc
#include "gtest/gtest.h"

#include <boost/asio.hpp>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "session.h"
#include "dispatcher.h"
// If you have handler type constants, you can include:
// #include "handler_types.h"

using boost::asio::ip::tcp;

// ---------------------- Helpers & wiring ----------------------

static tcp::endpoint BindLoopback(tcp::acceptor& acc, boost::asio::io_service& ios) {
    tcp::endpoint ep(tcp::v4(), 0);
    acc.open(ep.protocol());
    acc.set_option(tcp::acceptor::reuse_address(true));
    acc.bind(ep);
    acc.listen();
    return acc.local_endpoint();
}

// Build a Dispatcher with an echo handler at /echo (adjust the type string if yours differs)
static std::shared_ptr<Dispatcher> MakeEchoDispatcher() {
    std::vector<HandlerSpec> specs;
    HandlerSpec hs;
    hs.path = "/echo";
    hs.type = "echo"; // or handler_types::ECHO_HANDLER if available
    specs.push_back(hs);

    auto up = Dispatcher::FromSpecs(specs);
    return std::shared_ptr<Dispatcher>(std::move(up));
}

// Bundle to hold server/client sockets and the session
struct ConnectedSession {
    boost::asio::io_service ios;
    tcp::acceptor acceptor{ios};
    tcp::socket client{ios};
    std::shared_ptr<Session> session;
};

// Fill an existing ConnectedSession (avoids copying io_service)
static void MakeConnectedSession(std::shared_ptr<Dispatcher> disp, ConnectedSession* cs) {
    auto ep = BindLoopback(cs->acceptor, cs->ios);

    // Connect client to our acceptor
    cs->client.connect(ep);

    // Accept on the server side
    tcp::socket server_sock(cs->ios);
    cs->acceptor.accept(server_sock);

    // Create the Session and move the accepted socket into it
    cs->session = std::make_shared<Session>(cs->ios, std::move(disp));
    cs->session->socket() = std::move(server_sock);
}

// Run io_service for up to `ms`; stops if idle earlier
static void RunIosWithTimeout(boost::asio::io_service& ios,
                              std::chrono::milliseconds ms = std::chrono::milliseconds(1000)) {
    auto fut = std::async(std::launch::async, [&] { ios.run(); });
    if (fut.wait_for(ms) == std::future_status::timeout) ios.stop();
}

// Nonblocking read of whatever is available on the client
static std::string ReadAllNow(tcp::socket& s) {
    s.non_blocking(true);
    std::string out;
    for (;;) {
        char buf[4096];
        boost::system::error_code ec;
        std::size_t n = s.read_some(boost::asio::buffer(buf), ec);
        if (ec == boost::asio::error::would_block || ec == boost::asio::error::try_again) break;
        if (ec) break;
        out.append(buf, buf + n);
        if (n < sizeof(buf)) break;
    }
    return out;
}

// ---------------------- Tests ----------------------

// 1) Happy path: valid request to /echo → 200 OK; exercises start → read → dispatch → write → write_callback
TEST(SessionTest, EchoRoute_ValidRequest_Produces200AndWrites) {
    auto disp = MakeEchoDispatcher();
    ConnectedSession cs;
    MakeConnectedSession(disp, &cs);

    cs.session->start();

    const std::string req =
        "GET /echo HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Connection: close\r\n"
        "\r\n";
    boost::asio::write(cs.client, boost::asio::buffer(req));

    RunIosWithTimeout(cs.ios, std::chrono::milliseconds(1000));

    const std::string resp = ReadAllNow(cs.client);
    ASSERT_FALSE(resp.empty());
    EXPECT_NE(resp.find("HTTP/1.1 200"), std::string::npos);
}

// 2) Missing route: expect 404; exercises dispatch of not-found path, write, shutdown
TEST(SessionTest, MissingRoute_Produces404) {
    auto disp = MakeEchoDispatcher();
    ConnectedSession cs;
    MakeConnectedSession(disp, &cs);

    cs.session->start();

    const std::string req =
        "GET /missing HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "\r\n";
    boost::asio::write(cs.client, boost::asio::buffer(req));

    RunIosWithTimeout(cs.ios, std::chrono::milliseconds(1000));

    const std::string resp = ReadAllNow(cs.client);
    ASSERT_FALSE(resp.empty());
    EXPECT_NE(resp.find("HTTP/1.1 404"), std::string::npos);
}

// 4) BAD_REQUEST path: malformed request → 400; exercises BAD_REQUEST branch and HandleBadRequest()
TEST(SessionTest, MalformedRequest_Produces400) {
    auto disp = MakeEchoDispatcher();
    ConnectedSession cs;
    MakeConnectedSession(disp, &cs);

    cs.session->start();

    const std::string bad = "GARBAGE\r\n\r\n";
    boost::asio::write(cs.client, boost::asio::buffer(bad));

    RunIosWithTimeout(cs.ios, std::chrono::milliseconds(1000));

    const std::string resp = ReadAllNow(cs.client);
    ASSERT_FALSE(resp.empty());
    EXPECT_NE(resp.find("HTTP/1.1 400"), std::string::npos);
}

// 5) Write-error path: close client early so async_write / write_callback sees an error; no crash
TEST(SessionTest, WriteError_ClientClosed_NoCrash) {
    auto disp = MakeEchoDispatcher();
    ConnectedSession cs;
    MakeConnectedSession(disp, &cs);

    cs.session->start();

    const std::string req =
        "GET /echo HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "\r\n";
    boost::asio::write(cs.client, boost::asio::buffer(req));

    // Close client to induce write error on server side
    boost::system::error_code ignore;
    cs.client.shutdown(tcp::socket::shutdown_both, ignore);
    cs.client.close(ignore);

    EXPECT_NO_THROW({
        RunIosWithTimeout(cs.ios, std::chrono::milliseconds(1000));
    });
}

// 6) start() called twice quickly is harmless; still handles a request
TEST(SessionTest, StartIsSafeIfCalledTwiceQuickly) {
    auto disp = MakeEchoDispatcher();
    ConnectedSession cs;
    MakeConnectedSession(disp, &cs);

    EXPECT_NO_THROW({
        cs.session->start();
        cs.session->start();
    });

    const std::string req =
        "GET /echo HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "\r\n";
    boost::asio::write(cs.client, boost::asio::buffer(req));
    RunIosWithTimeout(cs.ios, std::chrono::milliseconds(1000));

    const std::string resp = ReadAllNow(cs.client);
    ASSERT_FALSE(resp.empty());
    EXPECT_NE(resp.find("HTTP/1.1 200"), std::string::npos);
}
