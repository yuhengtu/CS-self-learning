#ifndef REQUEST_HANDLER_FACTORY_H
#define REQUEST_HANDLER_FACTORY_H

#include <memory>
#include <string>

#include "request_handler.h"

class RequestHandlerFactory {
public:
    virtual ~RequestHandlerFactory() = default;
    virtual std::unique_ptr<RequestHandler> create(const std::string& location,
                                                   const std::string& url) = 0;
};

#endif // REQUEST_HANDLER_FACTORY_H

