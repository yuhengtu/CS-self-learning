#ifndef HEALTH_HANDLER_FACTORY_H
#define HEALTH_HANDLER_FACTORY_H

#include <memory>
#include <string>

#include "request_handler_factory.h"
#include "server_config.h"

class HealthHandlerFactory : public RequestHandlerFactory {
public:
    explicit HealthHandlerFactory(const HandlerSpec& spec);
    std::unique_ptr<RequestHandler> create(const std::string& location,
                                           const std::string& url) override;

private:
    std::string instance_name_;
};

// Registers the health handler factory with the global registry.
void RegisterHealthHandlerFactory();

#endif // HEALTH_HANDLER_FACTORY_H
