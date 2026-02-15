#include <cstdlib>
#include <iostream>
#include <vector>
#include <boost/asio.hpp>
#include <string>
#include <sstream>

#include "session.h"
#include "response.h"
#include "request_parser.h"
#include "request.h"
#include "dispatcher.h"
#include "logger.h"

using boost::asio::ip::tcp;

Session::Session(boost::asio::io_service& io_service, std::shared_ptr<Dispatcher> dispatcher)
    : m_socket(io_service), dispatcher_(std::move(dispatcher)) {}

tcp::socket& Session::socket(){return m_socket;}

void Session::start(){handle_read();}

void Session::handle_read()
{
    auto self(shared_from_this());
    Logger& log = Logger::getInstance();
    log.logTrace("session: Session began, reading from buffer");
    m_socket.async_read_some(
        boost::asio::buffer(data_),
        [this, self](const boost::system::error_code& error, std::size_t bytes_transferred) {handle_read_callback(self, error, bytes_transferred);
    });
}

void Session::handle_read_callback(std::shared_ptr<Session> self,
                                  boost::system::error_code error,
                                  std::size_t bytes_transferred) 
{
    Logger& log = Logger::getInstance();
    if (error) {
        log.logError(error.message());
        return;
    }

    const auto parse_status = parser_.parse(current_request_, data_, bytes_transferred);
    switch (parse_status) {
        case RequestParser::PROPER_REQUEST: {
            log.logTraceHTTPRequest(current_request_);
            
            // 1. Get the response from the dispatcher
            m_response = dispatcher_->Dispatch(current_request_);
            
            parser_.reset();
            current_request_.reset();
            
            // 2. The old switch(handler_result) is gone. We just write whatever
            //    response the dispatcher gave us (200, 404, 500, etc).
            log.logTrace("session: Dispatch complete, writing response");
            handle_write();
            return;
        }
        case RequestParser::BAD_REQUEST:
            log.logWarning("session: Parser returned BAD_REQUEST");
            parser_.reset();
            current_request_.reset();
            
            // 1. Get the 400 Bad Request response from the dispatcher
            m_response = dispatcher_->HandleBadRequest();
            
            // 2. Write the response
            handle_write();
            return;
        case RequestParser::IN_PROGRESS:
            log.logTrace("session: Parser returned IN_PROGRESS");
            handle_read(); // Continue reading
            return;
    }
}


void Session::handle_write()
{
    auto self(shared_from_this());

    Logger& log = Logger::getInstance();
    std::string status_line = m_response->get_status_line();

    std::istringstream iss(status_line);
    std::string http_version;
    std::string status_code;
    iss >> http_version >> status_code;  // extract second token into status_code

    tcp::endpoint remote_ep = m_socket.remote_endpoint();
    std::string client_ip = remote_ep.address().to_string();
    
    // Special machine-parsable line, always of form:
    // [ResponseMetrics] Code:<status code>, IP:<client IP address>, Content:<response body>
    // Example from a /health request: [ResponseMetrics] Code:200, IP:127.0.0.1, Content:OK
    std::string response_metric = "[ResponseMetrics] Code:" + status_code + ", IP:" + client_ip + ", Content:" + m_response->get_content();
    log.logTrace(response_metric);
    boost::asio::async_write(      
        m_socket, 
        // Use '->' operator to get buffers
        m_response->get_bufs(),
        [this, self](const boost::system::error_code& write_error, std::size_t bytes_transferred) {handle_write_callback(self, write_error, bytes_transferred);});
}

void Session::handle_write_callback(std::shared_ptr<Session> self,
                                    boost::system::error_code error,
                                    std::size_t bytes_transferred) 
{
    Logger& log = Logger::getInstance();
    if (error) {
        log.logError(error.message());
        return;
    }

    boost::system::error_code dummy_error_code;
    log.logSignal();
    m_socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both,
                    dummy_error_code);
}
