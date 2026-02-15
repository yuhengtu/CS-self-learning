#ifndef CRUD_REQUEST_HANDLER_H
#define CRUD_REQUEST_HANDLER_H

#include <memory>
#include <string>
#include <string_view>

#include "crud_manager_interface.h"
#include "handler_types.h"
#include "request_handler.h"

/**
 * Handles HTTP requests for CRUD operations.
 *
 * This handler translates REST-style requests (POST, GET, PUT, DELETE)
 * into operations on the underlying data store via a CrudManager.
 *
 * URI Format: /<mount_prefix>/<entity_type>/[id]
 * 
 * Examples:
 * - POST /api/users -> Create a new user with next available id
 * - GET  /api/users/1 -> Retrieve user with ID 1
 * - GET  /api/users -> List all valid user IDs
 * - PUT  /api/users/1 -> Update user with ID 1
 * - DELETE /api/users/1 -> Delete user with ID 1
 */
class CrudRequestHandler : public RequestHandler {
public:
    static constexpr std::string_view Create = "POST";
    static constexpr std::string_view Retrieve = "GET";
    static constexpr std::string_view Update = "PUT";
    static constexpr std::string_view Delete = "DELETE";

public:
    CrudRequestHandler(std::string mount_prefix,
                       std::shared_ptr<CrudManagerInterface> manager);
    std::unique_ptr<response> handle_request(const request& req) override;
    std::string name() const override { return handler_types::CRUD_HANDLER; }

private:
    const std::string mount_prefix_;
    std::shared_ptr<CrudManagerInterface> manager_;
};

#endif // CRUD_REQUEST_HANDLER_H