#include "static_handler_factory.h"

#include "static_request_handler.h"
#include "handler_types.h"
#include "handler_registry.h"
#include "logger.h"

StaticHandlerFactory::StaticHandlerFactory(const HandlerSpec& spec) {
    auto it = spec.options.find("root");
    if (it != spec.options.end()) {
        root_dir_ = it->second;
    }
}

std::unique_ptr<RequestHandler> StaticHandlerFactory::create(const std::string& location,
                                                             const std::string& [[maybe_unused]] url) {
    if (root_dir_.empty()) {
        // Defensive: should not happen if registry/dispatcher validated, but guard anyway.
        return nullptr;
    }
    return std::make_unique<StaticRequestHandler>(location, root_dir_);
}

void RegisterStaticHandlerFactory() {
    HandlerRegistry::Register(handler_types::STATIC_HANDLER,
        [](const HandlerSpec& spec) -> std::unique_ptr<RequestHandlerFactory> {
            // Validate required options early; log and return nullptr if missing
            auto it = spec.options.find("root");
            if (it == spec.options.end()) {
                Logger& log = Logger::getInstance();
                log.logError("dispatcher: static handler missing 'root' option");
                return std::unique_ptr<RequestHandlerFactory>(nullptr);
            }
            return std::unique_ptr<RequestHandlerFactory>(new StaticHandlerFactory(spec));
        });
}
