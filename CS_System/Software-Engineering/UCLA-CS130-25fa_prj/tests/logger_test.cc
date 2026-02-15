#include "gtest/gtest.h"

#include "logger.h"
#include "request.h"

#include <string>

// Just verifying that calls don't throw and the singleton is stable.
TEST(LoggerTest, SingletonIsStable) {
  Logger& a = Logger::getInstance();
  Logger& b = Logger::getInstance();
  // Same instance (by address comparison via references).
  EXPECT_EQ(&a, &b);
}

TEST(LoggerTest, InitIsIdempotent) {
  Logger& log = Logger::getInstance();
  // Re-invoking init() should be safe and not throw.
  EXPECT_NO_THROW(log.init());
}

TEST(LoggerTest, LogsAllSeverityHelpers) {
  Logger& log = Logger::getInstance();

  // Simple messages across all helpers
  EXPECT_NO_THROW(log.logServerInitialization());
  EXPECT_NO_THROW(log.logTrace("trace message"));
  EXPECT_NO_THROW(log.logDebug("debug message"));
  EXPECT_NO_THROW(log.logWarning("warning message"));
  EXPECT_NO_THROW(log.logError("error message"));
}

TEST(LoggerTest, LogsHttpRequestDumpAndSignal) {
  Logger& log = Logger::getInstance();

  request req;
  req.method = "GET";
  req.uri = "/echo";
  req.version = "1.1";
  req.raw =
      "GET /echo HTTP/1.1\r\n"
      "Host: localhost\r\n"
      "Accept: */*\r\n"
      "\r\n";

  // Covers logTraceHTTPRequest body (string stream + severity trace)
  EXPECT_NO_THROW(log.logTraceHTTPRequest(req));

  // Covers warning severity via logSignal()
  EXPECT_NO_THROW(log.logSignal());
}

TEST(LoggerTest, LogsConnectionDetails) {
    Logger& log = Logger::getInstance();

    boost::asio::io_context io;

    // Create a pair of connected sockets using a temporary acceptor
    tcp::acceptor acceptor(io, tcp::endpoint(tcp::v4(), 0)); // ephemeral port
    unsigned short port = acceptor.local_endpoint().port();

    tcp::socket client(io);
    client.connect(tcp::endpoint(tcp::v4(), port));

    tcp::socket server(io);
    acceptor.accept(server);

    EXPECT_NO_THROW(log.logConnectionDetails(server));

    client.close();
    server.close();
}
