// server.cc
#include <memory>
#include <boost/bind.hpp>
#include <boost/asio.hpp>

#include "server.h"
#include "session.h"
#include "logger.h"

using boost::asio::ip::tcp;

Server::Server(boost::asio::io_service& io_service, const ServerConfig& config)
    : io_service_(io_service),
      acceptor_(io_service, tcp::endpoint(tcp::v4(), static_cast<short>(config.port))),
      config_(config),
      dispatcher_(Dispatcher::FromSpecs(config.handlers)) {
        Logger& log = Logger::getInstance();
        log.logServerInitialization();
        start_accept();
    }

void Server::start_accept() {
    auto new_session = std::make_shared<Session>(io_service_, dispatcher_);

    // Changed: Pass handle_accept directly instead of using a lambda.
    // This ensures that handle_accept (not the lambda) controls the accept loop.
    // handle_accept calls start_accept() again after each connection is accepted,
    // so the server continues listening for new clients asynchronously.
    Logger& log = Logger::getInstance();
    log.logTrace("server: accepting connection");
    acceptor_.async_accept(new_session->socket(), boost::bind(&Server::handle_accept, this, new_session, boost::asio::placeholders::error));
}

void Server::handle_accept(std::shared_ptr<Session> new_session,
                           const boost::system::error_code& error) {
    if (!error) {
        Logger& log = Logger::getInstance();
        log.logTrace("server: starting session");
        log.logConnectionDetails(new_session->socket());
        new_session->start();
    }

    // Using shared_ptr ensures the session object remains alive as long as
    // there are async operations referencing it. We do NOT call 'delete'
    // manually anywhere — Boost.Asio async handlers capture shared_ptrs
    start_accept();
}
