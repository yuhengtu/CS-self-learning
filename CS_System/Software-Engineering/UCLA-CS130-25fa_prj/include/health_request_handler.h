#ifndef HEALTH_REQUEST_HANDLER_H
#define HEALTH_REQUEST_HANDLER_H

#include <string>
#include <memory>

#include "request_handler.h"
#include "handler_types.h"

class HealthRequestHandler : public RequestHandler {
public:
    explicit HealthRequestHandler(std::string instance_name = handler_types::HEALTH_HANDLER);

    std::unique_ptr<response> handle_request(const request& req) override;
    
    std::string name() const override { return instance_name_; }

private:
    std::string instance_name_;
};

#endif // HEALTH_REQUEST_HANDLER_H