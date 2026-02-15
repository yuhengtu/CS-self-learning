#include "request.h"

#include <cctype>

void request::reset() {
    method.clear();
    uri.clear();
    version.clear();
    headers.clear();
    body.clear();
    raw.clear();
}

std::string request::get_header_value(std::string_view key) const {
    for (const auto& header : headers) {
        if (header.first.size() == key.size()) {
            bool match = true;
            for (std::size_t i = 0; i < key.size(); ++i) {
                if (std::tolower(header.first[i]) != std::tolower(key[i])) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return header.second;
            }
        }
    }
    return "";
}
