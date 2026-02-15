#include "not_found_request_handler.h"
#include "response_builder.h"
#include "logger.h"
#include <memory> // For std::make_unique

NotFoundRequestHandler::NotFoundRequestHandler(std::string instance_name)
    : instance_name_(std::move(instance_name)) {}

std::unique_ptr<response>
NotFoundRequestHandler::handle_request(const request& req) {
    Logger& log = Logger::getInstance();
    log.logTrace("not_found_request_handler[" + instance_name_ + "]: handling request");

    // 1. Allocate the response
    auto out = std::make_unique<response>();

    // 2. Build a 404 response
    ResponseBuilder::createNotFound("The requested resource could not be found.")
        .build(*out);

    // 3. Return the 404 response
    return out;
}