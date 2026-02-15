#ifndef REQUEST_HANDLER_H
#define REQUEST_HANDLER_H

#include <string>
#include <memory>

#include "response.h"
#include "request.h"

class RequestHandler {
public:
    enum class result { HANDLED, CLIENT_ERROR, SERVER_ERROR };
    virtual ~RequestHandler() = default;

    virtual std::unique_ptr<response> handle_request(const request& request) = 0;
    virtual std::string name() const = 0;
};

#endif // REQUEST_HANDLER_H