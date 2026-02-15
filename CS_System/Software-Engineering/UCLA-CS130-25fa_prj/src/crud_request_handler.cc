#include "crud_request_handler.h"

#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "logger.h"
#include "response_builder.h"

namespace {

// Helper to split a string by a delimiter.
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

} // namespace

CrudRequestHandler::CrudRequestHandler(
    std::string mount_prefix, std::shared_ptr<CrudManagerInterface> manager)
    : mount_prefix_(std::move(mount_prefix)), manager_(std::move(manager)) {}


std::unique_ptr<response>
CrudRequestHandler::handle_request(const request& req) {
    Logger& log = Logger::getInstance();
    log.logDebug("crud_request_handler: Handling " + req.method + " request for " + req.uri);

    auto out = std::make_unique<response>();

    // Extract path relative to the handler's mount prefix.
    std::string relative_path = req.uri.substr(mount_prefix_.length());
    if (!relative_path.empty() && relative_path.front() == '/') {
        relative_path = relative_path.substr(1);
    }

    // Parse the relative path into entity type and optional ID.
    // Entity Type
    auto path_parts = split(relative_path, '/');
    if (path_parts.empty() || path_parts[0].empty()) {
        log.logWarning("crud_request_handler: Bad request - entity type is missing in URI: " + req.uri);
        ResponseBuilder::createBadRequest("Entity type is missing.").build(*out);
        return out;
    }

    // Optional ID
    std::string entity_type = path_parts[0];
    std::optional<int> id;
    if (path_parts.size() > 1) {
        try {
            id = std::stoi(path_parts[1]);
        } catch (const std::exception&) {
            log.logWarning("crud_request_handler: Bad request - invalid ID format in URI: " + req.uri);
            ResponseBuilder::createBadRequest("Invalid ID format.").build(*out);
            return out;
        }
    }

    log.logDebug("crud_request_handler: Parsed entity_type=" + entity_type + ", id=" + (id ? std::to_string(*id) : "none"));


    // Dispatch to the correct CRUD operation based on HTTP method.
    if (req.method == Create) { // CREATE
        log.logDebug("crud_request_handler: CREATE operation for entity_type=" + entity_type);
        auto new_id = manager_->Create(entity_type, req.body);
        if (new_id) {
            std::string body = "{\"id\": " + std::to_string(*new_id) + "}";
            log.logDebug("crud_request_handler: CREATE success, new_id=" + std::to_string(*new_id));
            ResponseBuilder::createOk().withContentType("application/json").withBody(body).build(*out);
        } else {
            // If we get here, creation failed 
            log.logError("crud_request_handler: CREATE failed for entity_type=" + entity_type);
            ResponseBuilder::createInternalServerError("failed to create entity.").build(*out);
        }
    } else if (req.method == Retrieve) { // RETRIEVE or LIST
        if (id) { // Retrieve specific entity if there is an id
            log.logDebug("crud_request_handler: RETRIEVE operation for entity_type=" + entity_type + ", id=" + std::to_string(*id));
            auto data = manager_->Read(entity_type, *id);
            if (data) {
                log.logDebug("crud_request_handler: RETRIEVE success for id=" + std::to_string(*id));
                ResponseBuilder::createOk().withContentType("application/json").withBody(*data).build(*out);
            } else {
                log.logWarning("crud_request_handler: RETRIEVE failed, entity not found for id=" + std::to_string(*id));
                ResponseBuilder::createNotFound("Entity not found.").build(*out);
            }
        } else { // List all entities
            log.logDebug("crud_request_handler: LIST operation for entity_type=" + entity_type);
            auto ids = manager_->List(entity_type);
            std::ostringstream json_list;
            json_list << "[";
            for (size_t i = 0; i < ids.size(); ++i) {
                json_list << ids[i] << (i == ids.size() - 1 ? "" : ", ");
            }
            json_list << "]";
            log.logDebug("crud_request_handler: LIST success, found " + std::to_string(ids.size()) + " entities.");
            ResponseBuilder::createOk().withContentType("application/json").withBody(json_list.str()).build(*out);
        }
    } else if (req.method == Update) { // UPDATE
        if (!id) {
            log.logWarning("crud_request_handler: Bad request - ID is required for PUT.");
            ResponseBuilder::createBadRequest("ID is required for PUT.").build(*out);
            return out;
        }
        log.logDebug("crud_request_handler: UPDATE operation for entity_type=" + entity_type + ", id=" + std::to_string(*id));
        if (manager_->Update(entity_type, *id, req.body)) {
            std::string body = "{\"id\": " + std::to_string(*id) + "}";
            log.logDebug("crud_request_handler: UPDATE success for id=" + std::to_string(*id));
            ResponseBuilder::createOk().withContentType("application/json").withBody(body).build(*out);
        } else {
            // If we get here, JSON was valid but filesystem operation failed
            log.logError("crud_request_handler: UPDATE failed for id=" + std::to_string(*id));
            ResponseBuilder::createInternalServerError("failed to update entity.").build(*out);
        }
    } else if (req.method == Delete) { // DELETE
        if (!id) {
            log.logWarning("crud_request_handler: Bad request - ID is required for DELETE.");
            ResponseBuilder::createBadRequest("ID is required for DELETE.").build(*out);
            return out;
        }
        log.logDebug("crud_request_handler: DELETE operation for entity_type=" + entity_type + ", id=" + std::to_string(*id));
        if (manager_->Delete(entity_type, *id)) {
            std::string body = "{\"id\": " + std::to_string(*id) + "}";
            log.logDebug("crud_request_handler: DELETE success for id=" + std::to_string(*id));
            ResponseBuilder::createOk().withContentType("application/json").withBody(body).build(*out);
        } else {
            log.logError("crud_request_handler: DELETE failed for id=" + std::to_string(*id));
            ResponseBuilder::createNotFound("Entity not found.").build(*out);
        }
    } else { // All unsupported methods
        log.logWarning("crud_request_handler: Unsupported method '" + req.method + "'");
        ResponseBuilder(405, "Method Not Allowed")
            .withHeader("Allow", "GET, POST, PUT, DELETE")
            .build(*out);
    }

    return out;
}