#include "sleep_handler_factory.h"

#include "sleep_request_handler.h"
#include "handler_types.h"
#include "handler_registry.h"
#include "logger.h"

SleepHandlerFactory::SleepHandlerFactory(const HandlerSpec& spec) {
    instance_name_ = spec.name.empty() ? handler_types::SLEEP_HANDLER : spec.name;
    if (auto it = spec.options.find("sleep_ms"); it != spec.options.end()) {
        try {
            int ms = std::stoi(it->second);
            if (ms > 0) {
                sleep_duration_ = std::chrono::milliseconds(ms);
            }
        } catch (const std::exception& e) {
            Logger::getInstance().logWarning("sleep_handler_factory: invalid sleep_ms='" + it->second + "'");
        }
    }
}

std::unique_ptr<RequestHandler> SleepHandlerFactory::create(const std::string& /*location*/,
                                                            const std::string& /*url*/) {
    return std::make_unique<SleepRequestHandler>(instance_name_, sleep_duration_);
}

void RegisterSleepHandlerFactory() {
    HandlerRegistry::Register(handler_types::SLEEP_HANDLER,
        [](const HandlerSpec& spec) -> std::unique_ptr<RequestHandlerFactory> {
            return std::unique_ptr<RequestHandlerFactory>(new SleepHandlerFactory(spec));
        });
}
