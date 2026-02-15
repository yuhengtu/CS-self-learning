#ifndef REQUEST_PARSER_H
#define REQUEST_PARSER_H

#include <cstddef>
#include <string>

#include "request.h"

class RequestParserTestPeer;

class RequestParser {
public:
    enum Status { PROPER_REQUEST, IN_PROGRESS, BAD_REQUEST };

    RequestParser();

    Status parse(request& req, const char* data, std::size_t len);
    void reset();

private:
    friend class RequestParserTestPeer;
    Status fail_impossible_state_();
    Status make_bad_request_(const std::string& context);

    enum StateMachine {
        START,
        METHOD,
        URI,
        H, T1, T2, P, SLASH, VERSION,
        REQUEST_LINE_CR,
        HEADER_START,
        HEADER_NAME,
        HEADER_COLON,
        HEADER_SPACE,
        HEADER_VALUE,
        HEADER_LINE_CR,
        END_CR,
        BODY
    } m_state;

    std::string current_header_name_;
    std::string current_header_value_;

    // Body handling
    std::size_t expected_body_len_ = 0;
    std::size_t remaining_body_len_ = 0;
};

#endif // REQUEST_PARSER_H
