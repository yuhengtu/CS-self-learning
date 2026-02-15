#ifndef NOT_FOUND_REQUEST_HANDLER_H
#define NOT_FOUND_REQUEST_HANDLER_H

#include <string>
#include <memory>

#include "request_handler.h"
#include "handler_types.h"

// This handler always returns a 404 Not Found response.
class NotFoundRequestHandler : public RequestHandler {
public:
    explicit NotFoundRequestHandler(std::string instance_name = handler_types::NOT_FOUND_HANDLER);

    // Creates and returns a 404 response.
    std::unique_ptr<response> handle_request(const request& req) override;
    
    std::string name() const override { return instance_name_; }

private:
    std::string instance_name_;
};

#endif // NOT_FOUND_REQUEST_HANDLER_H