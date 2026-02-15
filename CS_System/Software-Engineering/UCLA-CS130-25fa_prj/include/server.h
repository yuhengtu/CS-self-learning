// server.h
#ifndef SERVER_H
#define SERVER_H

#include <memory>
#include <boost/asio.hpp>
#include "session.h"
#include "server_config.h"
#include "dispatcher.h"

#include <gtest/gtest_prod.h>

using boost::asio::ip::tcp;

class Server {
public:
  Server(boost::asio::io_service& io_service, const ServerConfig& config);

private:
  void start_accept();

  // NOTE: Use shared_ptr so the session stays alive across async ops.
  void handle_accept(std::shared_ptr<Session> new_session,
                     const boost::system::error_code& error);

  boost::asio::io_service& io_service_;
  tcp::acceptor acceptor_;
  ServerConfig config_;
  std::shared_ptr<Dispatcher> dispatcher_;

  FRIEND_TEST(ServerTest, HandleAccept_Success_NoThrow);
  FRIEND_TEST(ServerTest, HandleAccept_Error_NoThrow);
};

#endif // SERVER_H
