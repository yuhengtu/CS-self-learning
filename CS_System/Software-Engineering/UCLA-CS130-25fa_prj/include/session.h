#ifndef SESSION_H
#define SESSION_H

#include <cstdlib>
#include <iostream>
#include <memory>
#include <boost/asio.hpp>

#include "response.h"
#include "request.h"
#include "request_parser.h"
#include "request_handler.h"
#include "dispatcher.h"

#include <gtest/gtest_prod.h>

using boost::asio::ip::tcp;

class Session : public std::enable_shared_from_this<Session>
{
public:
    Session(boost::asio::io_service& io_service,
            std::shared_ptr<Dispatcher> dispatcher);
    tcp::socket& socket();

    void start();

private:
    void handle_read();
    void handle_read_callback(std::shared_ptr<Session> self,
                            boost::system::error_code error,
                            std::size_t bytes_transferred);

    void handle_write();
    void handle_write_callback(std::shared_ptr<Session> self,
                            boost::system::error_code error,
                            std::size_t bytes_transferred); 

    tcp::socket m_socket;
    enum { max_length = 1024 };
    char data_[max_length];

    std::unique_ptr<response> m_response;
    
    RequestParser parser_;
    request current_request_;
    std::shared_ptr<Dispatcher> dispatcher_;

    FRIEND_TEST(SessionInternalTest, ReadCallback_InProgressThenProper);
    FRIEND_TEST(SessionInternalTest, ReadCallback_BadRequestPath);
    FRIEND_TEST(SessionInternalTest, WriteCallback_ErrorAndSuccess);
};

#endif // SESSION_H