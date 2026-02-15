#pragma once

#include <memory>
#include <string>

#include "request_handler_factory.h"
#include "server_config.h"

class SleepHandlerFactory : public RequestHandlerFactory {
public:
    explicit SleepHandlerFactory(const HandlerSpec& spec);

    std::unique_ptr<RequestHandler> create(const std::string& location,
                                           const std::string& url) override;

private:
    std::string instance_name_;
    std::chrono::milliseconds sleep_duration_{2000};
};

void RegisterSleepHandlerFactory();
