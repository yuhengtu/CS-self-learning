#ifndef NOT_FOUND_HANDLER_FACTORY_H
#define NOT_FOUND_HANDLER_FACTORY_H

#include <memory>
#include <string>

#include "request_handler_factory.h"
#include "server_config.h"

// Factory for creating NotFoundRequestHandler instances.
class NotFoundHandlerFactory : public RequestHandlerFactory {
public:
    explicit NotFoundHandlerFactory(const HandlerSpec& spec);
    std::unique_ptr<RequestHandler> create(const std::string& location,
                                           const std::string& url) override;

private:
    std::string instance_name_;
};

// Registers this factory with the global registry.
void RegisterNotFoundHandlerFactory();

#endif // NOT_FOUND_HANDLER_FACTORY_H