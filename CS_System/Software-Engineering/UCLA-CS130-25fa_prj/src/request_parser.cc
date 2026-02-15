#include <cstddef>

#include "request_parser.h"
#include "logger.h"

namespace {
inline bool is_upper(char c) {
    return c >= 'A' && c <= 'Z';
}

inline bool is_uri_char(char c) {
    return c > 0x20 && c != '\r' && c != '\n' && c != '\t';
}
}

RequestParser::RequestParser() : m_state(START) {}

void RequestParser::reset() {
    m_state = START;
    current_header_name_.clear();
    current_header_value_.clear();
    expected_body_len_ = 0;
    remaining_body_len_ = 0;
}

RequestParser::Status RequestParser::parse(request& req, const char* data, std::size_t len) {
    Logger& log = Logger::getInstance();

    if (data == nullptr || len == 0) {
        log.logDebug("RequestParser: No data provided, staying IN_PROGRESS");
        return IN_PROGRESS;
    }

    if (m_state == START) {
        req.reset();
    }

    req.raw.append(data, len);

    // If we are currently expecting a body, consume as much as available and return
    if (m_state == BODY) {
        std::size_t to_copy = std::min(remaining_body_len_, len);
        if (to_copy > 0) {
            req.body.append(data, to_copy);
            remaining_body_len_ -= to_copy;
        }
        if (remaining_body_len_ == 0) {
            log.logDebug("RequestParser: Completed HTTP body read");
            return PROPER_REQUEST;
        }
        return IN_PROGRESS;
    }

    for (std::size_t i = 0; i < len; ++i) {
        char c = data[i];
    switch (m_state) {
            case START:
                if (is_upper(c)) {
                    req.method.push_back(c);
                    m_state = METHOD;
                } else {
                    return make_bad_request_("START");
                }
                break;
            case METHOD:
                if (is_upper(c)) {
                    req.method.push_back(c);
                } else if (c == ' ') {
                    m_state = URI;
                } else {
                    return make_bad_request_("METHOD");
                }
                break;
            case URI:
                if (c == ' ') {
                    m_state = H;
                } else if (!is_uri_char(c)) {
                    return make_bad_request_("URI");
                } else {
                    req.uri.push_back(c);
                }
                break;
            case H:
                if (c == 'H') m_state = T1;
                else return make_bad_request_("H");
                break;
            case T1:
                if (c == 'T') m_state = T2;
                else return make_bad_request_("T1");
                break;
            case T2:
                if (c == 'T') m_state = P;
                else return make_bad_request_("T2");
                break;
            case P:
                if (c == 'P') m_state = SLASH;
                else return make_bad_request_("P");
                break;
            case SLASH:
                if (c == '/') {
                    m_state = VERSION;
                } else {
                    return make_bad_request_("SLASH");
                }
                break;
            case VERSION:
                if (c == '\r') {
                    if (req.version != "1.1") {
                        return make_bad_request_("VERSION");
                    }
                    m_state = REQUEST_LINE_CR;
                } else if (c == '1' || c == '.') {
                    req.version.push_back(c);
                } else {
                    return make_bad_request_("VERSION");
                }
                break;
            case REQUEST_LINE_CR:
                if (c == '\n') {
                    m_state = HEADER_START;
                } else {
                    return make_bad_request_("REQUEST_LINE_CR");
                }
                break;
            case HEADER_START:
                if (c == '\r') {
                    m_state = END_CR;
                } else {
                    current_header_name_.clear();
                    current_header_value_.clear();
                    m_state = HEADER_NAME;
                    --i; // reprocess same char as part of header name
                }
                break;
            case HEADER_NAME:
                if (c == ':') {
                    if (current_header_name_.empty()) {
                        return make_bad_request_("HEADER_NAME");
                    }
                    m_state = HEADER_COLON;
                } else if (c == '\r' || c == '\n') {
                    return make_bad_request_("HEADER_NAME");
                } else {
                    current_header_name_.push_back(c);
                }
                break;
            case HEADER_COLON:
                if (c == ' ') {
                    m_state = HEADER_SPACE;
                } else {
                    return make_bad_request_("HEADER_COLON");
                }
                break;
            case HEADER_SPACE:
                if (c == '\r') {
                    m_state = HEADER_LINE_CR;
                } else {
                    current_header_value_.push_back(c);
                    m_state = HEADER_VALUE;
                }
                break;
            case HEADER_VALUE:
                if (c == '\r') {
                    m_state = HEADER_LINE_CR;
                } else {
                    current_header_value_.push_back(c);
                }
                break;
            case HEADER_LINE_CR:
                if (c == '\n') {
                    req.headers.emplace_back(current_header_name_, current_header_value_);          
                    m_state = HEADER_START;
                } else {
                    return make_bad_request_("HEADER_LINE_CR");
                }
                break;
            case END_CR:
                if (c == '\n') {
                    // End of headers. Determine if a body is expected.
                    std::string cl_str = req.get_header_value("Content-Length");
                    if (cl_str.empty()) {
                        log.logDebug("RequestParser: No Content-Length, finishing at headers");
                        return PROPER_REQUEST;
                    }
                    // Strictly require digits only (no signs, no whitespace)
                    for (char ch : cl_str) {
                        if (ch < '0' || ch > '9') {
                            return make_bad_request_("CONTENT_LENGTH_PARSE");
                        }
                    }
                    try {
                        expected_body_len_ = static_cast<std::size_t>(std::stoul(cl_str));
                    } catch (...) {
                        return make_bad_request_("CONTENT_LENGTH_PARSE");
                    }
                    remaining_body_len_ = expected_body_len_;

                    // Copy any remaining bytes from this buffer into the body
                    std::size_t start = i + 1; // next byte after this \n
                    if (start < len) {
                        std::size_t available = len - start;
                        std::size_t to_copy = std::min(available, remaining_body_len_);
                        if (to_copy > 0) {
                            req.body.append(data + start, to_copy);
                            remaining_body_len_ -= to_copy;
                        }
                    }

                    if (remaining_body_len_ == 0) {
                        log.logDebug("RequestParser: Headers and full body parsed in single buffer");
                        return PROPER_REQUEST;
                    }
                    // Need more body bytes
                    m_state = BODY;
                    return IN_PROGRESS;
                } else {
                    // Invalid character after CR in header line
                    return make_bad_request_("END_CR");
                }
                break;
            default:
                return fail_impossible_state_();
        }
    }

    log.logDebug("RequestParser: HTTP Request not fully parsed, returning IN_PROGRESS");
    return IN_PROGRESS;
}

RequestParser::Status RequestParser::fail_impossible_state_() {
    Logger& log = Logger::getInstance();
    log.logError("RequestParser: Impossible FSM state reached\n");
    return BAD_REQUEST;
}

RequestParser::Status RequestParser::make_bad_request_(const std::string& context) {
    Logger& log = Logger::getInstance();
    log.logDebug("RequestParser: Bad HTTP Request at state " + context);
    return BAD_REQUEST;
}
