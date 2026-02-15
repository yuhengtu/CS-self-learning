#ifndef ECHO_REQUEST_HANDLER_H
#define ECHO_REQUEST_HANDLER_H

#include <string>
#include <memory>

#include "request_handler.h"
#include "handler_types.h"

class EchoRequestHandler : public RequestHandler {
public:
    explicit EchoRequestHandler(std::string instance_name = handler_types::ECHO_HANDLER);

    std::unique_ptr<response> handle_request(const request& req) override;
    
    std::string name() const override { return instance_name_; }

private:
    std::string instance_name_;
};

#endif // ECHO_REQUEST_HANDLER_H