#ifndef RESPONSE_H
#define RESPONSE_H

#include <vector>
#include <string>
#include <boost/asio.hpp>

class response {
public:
    using buffer_type = boost::asio::const_buffer;

    response();
    
    void set_status_line(std::string line);
    void set_headers(std::string line);
    void set_content(std::string line);

    std::string get_status_line() const;
    std::string get_headers() const;
    std::string get_content() const;

    void set_bufs();
    const std::vector<buffer_type>& get_bufs() const;

private:
    std::string m_status_line;
    std::string m_headers;
    std::string m_content;
    std::vector<buffer_type> m_bufs;
};

#endif // RESPONSE_H
