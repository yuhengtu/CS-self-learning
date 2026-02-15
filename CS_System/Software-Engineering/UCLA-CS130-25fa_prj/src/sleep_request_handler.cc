#include "sleep_request_handler.h"

#include "response_builder.h"
#include "logger.h"

#include <chrono>
#include <thread>
#include <memory>

SleepRequestHandler::SleepRequestHandler(std::string instance_name,
                                         std::chrono::milliseconds sleep_duration)
    : instance_name_(std::move(instance_name)), sleep_duration_(sleep_duration) {}

std::unique_ptr<response>
SleepRequestHandler::handle_request(const request& /*req*/) {
    Logger& log = Logger::getInstance();
    log.logTrace("sleep_request_handler[" + instance_name_ + "]: handling request (sleeping)");

    // Simulate a slow handler
    std::this_thread::sleep_for(sleep_duration_);

    auto out = std::make_unique<response>();

    ResponseBuilder(200)
        .withContentType("text/plain")
        .withBody("SLEPT")
        .build(*out);

    return out;
}

std::string SleepRequestHandler::name() const {
    return instance_name_;
}
