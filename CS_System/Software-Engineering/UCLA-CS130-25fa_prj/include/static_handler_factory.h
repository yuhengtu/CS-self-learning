#ifndef STATIC_HANDLER_FACTORY_H
#define STATIC_HANDLER_FACTORY_H

#include <memory>
#include <string>

#include "request_handler_factory.h"
#include "server_config.h"

class StaticHandlerFactory : public RequestHandlerFactory {
public:
    explicit StaticHandlerFactory(const HandlerSpec& spec);
    std::unique_ptr<RequestHandler> create(const std::string& location,
                                           const std::string& [[maybe_unused]] url) override;

private:
    std::string root_dir_;
};

// Registers the static handler factory with the global registry.
void RegisterStaticHandlerFactory();

#endif // STATIC_HANDLER_FACTORY_H
