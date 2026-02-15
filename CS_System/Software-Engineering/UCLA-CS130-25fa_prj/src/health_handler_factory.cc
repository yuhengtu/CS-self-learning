#include "health_handler_factory.h"

#include "health_request_handler.h"
#include "handler_types.h"
#include "handler_registry.h"

HealthHandlerFactory::HealthHandlerFactory(const HandlerSpec& spec) {
    instance_name_ = spec.name.empty() ? handler_types::HEALTH_HANDLER : spec.name;
}

std::unique_ptr<RequestHandler> HealthHandlerFactory::create(const std::string& /*location*/,
                                                           const std::string& /*url*/) {
    return std::make_unique<HealthRequestHandler>(instance_name_);
}

void RegisterHealthHandlerFactory() {
    HandlerRegistry::Register(handler_types::HEALTH_HANDLER,
        [](const HandlerSpec& spec) -> std::unique_ptr<RequestHandlerFactory> {
            return std::unique_ptr<RequestHandlerFactory>(new HealthHandlerFactory(spec));
        });
}
