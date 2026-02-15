#pragma once

#include <chrono>
#include <memory>
#include <string>

#include "request_handler.h"
#include "request.h"
#include "response.h"

class SleepRequestHandler : public RequestHandler {
public:
    explicit SleepRequestHandler(std::string instance_name,
                                 std::chrono::milliseconds sleep_duration = std::chrono::milliseconds(2000));

    std::unique_ptr<response> handle_request(const request& req) override;

    std::string name() const override;

private:
    std::string instance_name_;
    std::chrono::milliseconds sleep_duration_;
};
