#ifndef DISPATCHER_H
#define DISPATCHER_H

#include <memory>
#include <string>
#include <vector>

#include "request.h"
#include "response.h"
#include "request_handler.h"
#include "server_config.h"

#include "request_handler_factory.h"

class Dispatcher {
public:
    struct RouteInit {
        std::string location;
        std::unique_ptr<RequestHandlerFactory> factory;
    };

    static std::unique_ptr<Dispatcher> FromSpecs(const std::vector<HandlerSpec>& specs);

    explicit Dispatcher(std::vector<RouteInit> routes);
    ~Dispatcher();

    std::unique_ptr<response> Dispatch(const request& req) const;
    
    std::unique_ptr<response> HandleBadRequest() const;

private:
    struct Route {
        std::string location;
        std::unique_ptr<RequestHandlerFactory> factory;
    };

    // Routes are kept ordered longest-prefix-first so Dispatch can stop at
    // the first match and still satisfy longest-prefix routing semantics.
    std::vector<Route> routes_;
};

#endif // DISPATCHER_H