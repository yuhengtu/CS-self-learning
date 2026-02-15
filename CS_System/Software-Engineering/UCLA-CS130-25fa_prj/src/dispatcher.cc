#include "dispatcher.h"

#include <algorithm>
#include <utility>
#include <memory>

#include "echo_request_handler.h"
#include "response_builder.h"
#include "logger.h"
#include "handler_types.h"
#include "request_handler_factory.h"
#include "handler_registry.h"

namespace {
std::vector<Dispatcher::RouteInit> BuildRoutesFromSpecs(const std::vector<HandlerSpec>& specs) {
    std::vector<Dispatcher::RouteInit> routes;
    for (const auto& spec : specs) {
        auto factory = HandlerRegistry::CreateFactory(spec);
        if (!factory) {
            continue;
        }
        Dispatcher::RouteInit route;
        route.location = spec.path;
        route.factory = std::move(factory);
        routes.emplace_back(std::move(route));
    }

    bool has_root = false;

    for (const auto& route : routes) {
        if (route.location == "/") {
            has_root = true;
            break;
        }
    }

    if (!has_root) {
        Logger& log = Logger::getInstance();
        log.logDebug("dispatcher: injecting NotFoundHandler at '/'");
        HandlerSpec not_found_spec;
        not_found_spec.name = "default_not_found";
        not_found_spec.path = "/";
        not_found_spec.type = handler_types::NOT_FOUND_HANDLER;

        auto not_found_factory = HandlerRegistry::CreateFactory(not_found_spec);
        if (not_found_factory) {
            Dispatcher::RouteInit route;
            route.location = "/";
            route.factory = std::move(not_found_factory);
            routes.emplace_back(std::move(route));
        }
    }

    return routes;
}
}  // namespace

std::unique_ptr<Dispatcher> Dispatcher::FromSpecs(const std::vector<HandlerSpec>& specs) {
    auto routes = BuildRoutesFromSpecs(specs);
    return std::unique_ptr<Dispatcher>(new Dispatcher(std::move(routes)));
}

Dispatcher::Dispatcher(std::vector<RouteInit> routes) {
    std::sort(routes.begin(), routes.end(),
              [](const RouteInit& lhs, const RouteInit& rhs) {
                  return lhs.location.size() > rhs.location.size();
              });

    for (auto& init : routes) {
        if (!init.factory) {
            continue;
        }
        Route route;
        route.location = std::move(init.location);
        route.factory = std::move(init.factory);
        routes_.emplace_back(std::move(route));
    }
}

Dispatcher::~Dispatcher() = default;

std::unique_ptr<response> Dispatcher::Dispatch(const request& req) const {
    Logger& log = Logger::getInstance();
    for (const auto& route : routes_) {
        if (req.uri.rfind(route.location, 0) == 0 && route.factory) {
            auto handler = route.factory->create(route.location, req.uri);
            if (!handler) {
                log.logError("dispatcher: factory failed to create handler for location '" + route.location + "'");
                
                // No route matched — delegate to root (/) handler if present
                Logger& log = Logger::getInstance();
                for (const auto& route : routes_) {
                    if (route.location == "/") {
                        log.logTrace("dispatcher: no route matched, using root '/' handler");
                        auto handler = route.factory->create("/", req.uri);
                        return handler->handle_request(req);
                    }
                }
                log.logError("dispatcher: no root (/) handler found — this should never happen");
                auto resp = std::make_unique<response>();
                ResponseBuilder::createInternalServerError().build(*resp);
                return resp;
            }
            
            // Call the handler and return its response
            return handler->handle_request(req);
        }
    }

    // No matching handler found, create and return a 404 response
    auto resp = std::make_unique<response>();
    ResponseBuilder::createNotFound().build(*resp);
    return resp;
}

std::unique_ptr<response> Dispatcher::HandleBadRequest() const {
    auto resp = std::make_unique<response>();
    ResponseBuilder::createBadRequest().build(*resp);
    return resp;
}