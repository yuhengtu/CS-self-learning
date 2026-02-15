#include "handler_registry.h"

#include "logger.h"
#include "handler_types.h"
#include "echo_handler_factory.h"
#include "static_handler_factory.h"
#include "crud_handler_factory.h"
#include "not_found_handler_factory.h"
#include "health_handler_factory.h"
#include "sleep_handler_factory.h"
#include "link_manage_handler_factory.h"
#include "link_redirect_handler_factory.h"
#include "analytics_handler_factory.h"

namespace {
    std::unordered_map<std::string, HandlerRegistry::FactoryCtor> g_registry;
}

std::unordered_map<std::string, HandlerRegistry::FactoryCtor>& HandlerRegistry::map() {
    return g_registry;
}

void HandlerRegistry::Register(const std::string& type, FactoryCtor ctor) {
    map()[type] = std::move(ctor);
}

void HandlerRegistry::RegisterBuiltins() {
    static bool done = false;
    if (done) return;
    done = true;

    RegisterEchoHandlerFactory();
    RegisterStaticHandlerFactory();
    RegisterCrudHandlerFactory();
    RegisterNotFoundHandlerFactory();
    RegisterHealthHandlerFactory();
    RegisterSleepHandlerFactory();
    RegisterLinkManageHandlerFactory();
    RegisterLinkRedirectHandlerFactory();
    RegisterAnalyticsHandlerFactory();
}

std::unique_ptr<RequestHandlerFactory> HandlerRegistry::CreateFactory(const HandlerSpec& spec) {
    if (map().empty()) {
        // Lazily ensure builtins are available.
        RegisterBuiltins();
    }
    auto it = map().find(spec.type);
    if (it == map().end()) {
        Logger& log = Logger::getInstance();
        log.logError("dispatcher: Unknown handler type '" + spec.type + "'");
        return nullptr;
    }
    return (it->second)(spec);
}
