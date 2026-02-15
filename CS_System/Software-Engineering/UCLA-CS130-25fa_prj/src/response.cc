#include <vector>
#include <string>
#include <boost/asio.hpp>
#include <stdexcept>

#include "response.h"
#include "logger.h"

response::response() : m_status_line("unset"), m_headers("unset"), m_content("unset"){}

void response::set_status_line(std::string line){ m_status_line = line; }
void response::set_headers(std::string line){ m_headers = line; }
void response::set_content(std::string line){ m_content = line; }

std::string response::get_status_line() const { return m_status_line; }
std::string response::get_headers() const { return m_headers; }
std::string response::get_content() const { return m_content; }

void response::set_bufs() {
    Logger& log = Logger::getInstance();
    if (m_status_line == "unset"){
        log.logError("response: set_bufs called before set_status_line");
        throw std::runtime_error("response: set_bufs called before set_status_line");
    }
    if (m_headers == "unset"){
        log.logError("response: set_bufs called before set_headers");
        throw std::runtime_error("response: set_bufs called before set_headers");
    }
    if (m_content == "unset"){
        log.logError("response: set_bufs called before set_content");
        throw std::runtime_error("response: set_bufs called before set_content");
    }
    
    m_bufs.clear();
    m_bufs.push_back(boost::asio::buffer(m_status_line));
    m_bufs.push_back(boost::asio::buffer(m_headers));
    m_bufs.push_back(boost::asio::buffer(m_content));
}

const std::vector<response::buffer_type>& response::get_bufs() const { return m_bufs; }
