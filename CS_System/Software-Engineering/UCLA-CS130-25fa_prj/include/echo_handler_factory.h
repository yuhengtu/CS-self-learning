#ifndef ECHO_HANDLER_FACTORY_H
#define ECHO_HANDLER_FACTORY_H

#include <memory>
#include <string>

#include "request_handler_factory.h"
#include "server_config.h"

class EchoHandlerFactory : public RequestHandlerFactory {
public:
    explicit EchoHandlerFactory(const HandlerSpec& spec);
    std::unique_ptr<RequestHandler> create(const std::string& location,
                                           const std::string& url) override;

private:
    std::string instance_name_;
};

// Registers the echo handler factory with the global registry.
void RegisterEchoHandlerFactory();

#endif // ECHO_HANDLER_FACTORY_H
