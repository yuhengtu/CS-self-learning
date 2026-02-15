#include "health_request_handler.h"
#include "response_builder.h"

#include "logger.h"
#include <memory>

HealthRequestHandler::HealthRequestHandler(std::string instance_name) : instance_name_(std::move(instance_name)) {}

std::unique_ptr<response>
HealthRequestHandler::handle_request(const request& req) {
    Logger& log = Logger::getInstance();
    log.logTrace("health_request_handler[" + instance_name_ + "]: handling request");

    auto out = std::make_unique<response>();

    // Simply returns 200 OK with a payload of "OK" for any request for /health
    ResponseBuilder(200)
        .withContentType("text/plain")
        .withBody("OK")
        .build(*out);

    return out;
}
