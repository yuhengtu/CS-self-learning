#ifndef REQUEST_H
#define REQUEST_H

#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

struct request {
    std::string method;
    std::string uri;
    std::string version;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;
    std::string raw;

    void reset();
    std::string get_header_value(std::string_view key) const;
};

#endif // REQUEST_H
