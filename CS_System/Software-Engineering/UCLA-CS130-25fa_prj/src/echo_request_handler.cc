#include "echo_request_handler.h"
#include "response_builder.h"

#include "logger.h"
#include <memory>

EchoRequestHandler::EchoRequestHandler(std::string instance_name)
    : instance_name_(std::move(instance_name)) {}

std::unique_ptr<response>
EchoRequestHandler::handle_request(const request& req) {
    Logger& log = Logger::getInstance();
    log.logTrace("echo_request_handler[" + instance_name_ + "]: handling request");

    // 1. Handler allocates the response
    auto out = std::make_unique<response>();

    // 2. Handler builds the response
    ResponseBuilder(200)
        .withContentType("text/plain")
        .withBody(req.raw)
        .build(*out);

    // 3. Handler returns the allocated response
    return out;
}
