#ifndef CRUD_MANAGER_H
#define CRUD_MANAGER_H

#include "crud_manager_interface.h"

#include <mutex>

/**
 * Manages CRUD (Create, Read, Update, Delete) operations for entities
 * on the filesystem.
 *
 * This class provides an abstraction over the filesystem, allowing entities
 * to be stored and retrieved without exposing file I/O details to the caller.
 * It is designed to be used by a request handler that translates HTTP requests
 * into calls to this manager.
 */
class CrudManager : public CrudManagerInterface {
public:
    explicit CrudManager(const std::string& data_path);

    // Creates a new entity with the given data, returning the new ID.
    std::optional<int> Create(const std::string& entity_type, const std::string& data) override;

    // Retrieves the data for a given entity and ID.
    std::optional<std::string> Read(const std::string& entity_type, int id) override;

    // Creates or updates an entity with the given data at a specific ID.
    bool Update(const std::string& entity_type, int id, const std::string& data) override;

    // Deletes an entity with the given ID.
    bool Delete(const std::string& entity_type, int id) override;

    // Lists all existing IDs for a given entity type.
    std::vector<int> List(const std::string& entity_type) override;

private:
    const std::string data_path_;
    mutable std::mutex mutex_;
};

#endif // CRUD_MANAGER_H