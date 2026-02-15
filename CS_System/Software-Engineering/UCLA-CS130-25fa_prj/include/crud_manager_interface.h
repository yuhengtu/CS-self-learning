#ifndef CRUD_MANAGER_INTERFACE_H
#define CRUD_MANAGER_INTERFACE_H

#include <string>
#include <vector>
#include <optional>
#include <memory>

/**
 * Defines the interface for a class that manages CRUD operations.
 *
 * This abstract class allows for dependency injection, enabling consumers
 * (like request handlers) to be tested with mock implementations.
 *
 * IMPORTANT: Any implementation of this interface must adhere to the following
 * edge case behaviors to ensure consistency across implementations (including
 * test fakes). These behaviors are documented per-method below.
 */
class CrudManagerInterface {
public:
    virtual ~CrudManagerInterface() = default;

    virtual std::optional<int> Create(const std::string& entity_type, const std::string& data) = 0;

    virtual std::optional<std::string> Read(const std::string& entity_type, int id) = 0;

    virtual bool Update(const std::string& entity_type, int id, const std::string& data) = 0;

    virtual bool Delete(const std::string& entity_type, int id) = 0;

    virtual std::vector<int> List(const std::string& entity_type) = 0;
};

#endif // CRUD_MANAGER_INTERFACE_H