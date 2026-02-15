#ifndef STATIC_REQUEST_HANDLER_H
#define STATIC_REQUEST_HANDLER_H

#include <string>
#include <memory>

#include "request_handler.h"
#include "handler_types.h"

class StaticRequestHandler : public RequestHandler {
public:
    // mount_prefix: the configured route path (e.g. "/static")
    // root: filesystem root to serve from (e.g. "/file/list")
    StaticRequestHandler(std::string mount_prefix, std::string root);
    std::unique_ptr<response> handle_request(const request& req) override;
    
    std::string name() const override { return handler_types::STATIC_HANDLER; }

private:
    std::string mount_prefix_;
    std::string root_;

    static std::string content_type_for_extension(const std::string& path);
};

#endif // STATIC_REQUEST_HANDLER_H