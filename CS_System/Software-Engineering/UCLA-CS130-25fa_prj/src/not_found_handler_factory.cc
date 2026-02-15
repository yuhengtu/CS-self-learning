#include "not_found_handler_factory.h"
#include "not_found_request_handler.h"
#include "handler_types.h"
#include "handler_registry.h"

NotFoundHandlerFactory::NotFoundHandlerFactory(const HandlerSpec& spec) {
    instance_name_ = spec.name.empty() ? handler_types::NOT_FOUND_HANDLER : spec.name;
}

std::unique_ptr<RequestHandler> NotFoundHandlerFactory::create(const std::string& /*location*/,
                                                               const std::string& /*url*/) {
    return std::make_unique<NotFoundRequestHandler>(instance_name_);
}

void RegisterNotFoundHandlerFactory() {
    HandlerRegistry::Register(handler_types::NOT_FOUND_HANDLER,
        [](const HandlerSpec& spec) -> std::unique_ptr<RequestHandlerFactory> {
            return std::unique_ptr<RequestHandlerFactory>(new NotFoundHandlerFactory(spec));
        });
}