#ifndef HANDLER_REGISTRY_H
#define HANDLER_REGISTRY_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "server_config.h"
#include "request_handler_factory.h"

class HandlerRegistry {
public:
    using FactoryCtor = std::function<std::unique_ptr<RequestHandlerFactory>(const HandlerSpec&)>;

    static void Register(const std::string& type, FactoryCtor ctor);
    static std::unique_ptr<RequestHandlerFactory> CreateFactory(const HandlerSpec& spec);
    // Ensure built-in handler types are registered exactly once.
    static void RegisterBuiltins();

private:
    static std::unordered_map<std::string, FactoryCtor>& map();
};

#endif // HANDLER_REGISTRY_H
