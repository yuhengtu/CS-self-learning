// Generated with ChatGPT
// tests/server_test.cc
#include "gtest/gtest.h"

#include <boost/asio.hpp>
#include <memory>

#include "server.h"
#include "session.h"
#include "server_config.h"
#include "dispatcher.h"

using boost::asio::ip::tcp;

namespace {

// Minimal config: port 0 (ephemeral), no handlers (FromSpecs will inject not_found at "/").
ServerConfig MakeMinimalConfig(unsigned short port = 0) {
    ServerConfig cfg;
    cfg.port = port;
    cfg.handlers.clear();
    return cfg;
}

// Session requires a std::shared_ptr<Dispatcher>. FromSpecs returns unique_ptr,
// so wrap it into shared_ptr for construction when we need a Session.
std::shared_ptr<Dispatcher> MakeSharedDispatcher(const std::vector<HandlerSpec>& specs = {}) {
    auto up = Dispatcher::FromSpecs(specs);               // unique_ptr<Dispatcher>
    return std::shared_ptr<Dispatcher>(std::move(up));    // -> shared_ptr<Dispatcher>
}

} // namespace

// Server constructor:
// - Initializes acceptor on the given port (0 = ephemeral)
// - Initializes dispatcher via FromSpecs
// - Logs initialization
// - Calls start_accept(), which creates a Session and schedules async_accept()
// We don't run the io_service; we just assert "no throw".
TEST(ServerTest, ConstructorInvokesStartAccept_NoThrow) {
    boost::asio::io_service ios;
    ServerConfig cfg = MakeMinimalConfig(/*port=*/0);

    EXPECT_NO_THROW({
        Server server(ios, cfg);
    });
}

// Direct coverage of the "error" branch in handle_accept():
// If error != success, it does not start the session but still calls start_accept()
// to re-arm. We assert "no throw".
TEST(ServerTest, HandleAccept_Error_NoThrow) {
    boost::asio::io_service ios;
    ServerConfig cfg = MakeMinimalConfig(0);
    Server server(ios, cfg);

    auto disp = MakeSharedDispatcher();
    auto session = std::make_shared<Session>(ios, disp);

    auto ec = make_error_code(boost::asio::error::operation_aborted);
    EXPECT_NO_THROW({
        server.handle_accept(session, ec);
    });
}

// Sanity: We can construct a Session the same way Server would, and touch the socket.
// No network I/O is performed; we don't run the io_service.
TEST(ServerTest, SessionConstructAndSocket_NoThrow) {
    boost::asio::io_service ios;
    auto disp = MakeSharedDispatcher();

    EXPECT_NO_THROW({
        auto s = std::make_shared<Session>(ios, disp);
        tcp::socket& sock = s->socket();
        (void)sock;
    });
}
