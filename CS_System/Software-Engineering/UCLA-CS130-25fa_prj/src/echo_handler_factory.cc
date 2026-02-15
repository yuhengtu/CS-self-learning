#include "echo_handler_factory.h"

#include "echo_request_handler.h"
#include "handler_types.h"
#include "handler_registry.h"

EchoHandlerFactory::EchoHandlerFactory(const HandlerSpec& spec) {
    instance_name_ = spec.name.empty() ? handler_types::ECHO_HANDLER : spec.name;
}

std::unique_ptr<RequestHandler> EchoHandlerFactory::create(const std::string& /*location*/,
                                                           const std::string& /*url*/) {
    return std::make_unique<EchoRequestHandler>(instance_name_);
}

void RegisterEchoHandlerFactory() {
    HandlerRegistry::Register(handler_types::ECHO_HANDLER,
        [](const HandlerSpec& spec) -> std::unique_ptr<RequestHandlerFactory> {
            return std::unique_ptr<RequestHandlerFactory>(new EchoHandlerFactory(spec));
        });
}
