#include "crud_handler_factory.h"

#include <memory>

#include "crud_manager.h"
#include "crud_request_handler.h"
#include "handler_registry.h"
#include "handler_types.h"
#include "logger.h"

CrudHandlerFactory::CrudHandlerFactory(const HandlerSpec& spec) {
  if (auto it = spec.options.find("data_path"); it != spec.options.end()) {
    data_path_ = it->second;
    manager_ = std::make_shared<CrudManager>(data_path_);
  }
}

std::unique_ptr<RequestHandler>
CrudHandlerFactory::create(const std::string& location,
                           const std::string& /*url*/) {
  return std::make_unique<CrudRequestHandler>(location, manager_);
}

void RegisterCrudHandlerFactory() {
  HandlerRegistry::Register(
      handler_types::CRUD_HANDLER,
      [](const HandlerSpec& spec) -> std::unique_ptr<RequestHandlerFactory> {
        // Validate required options; log + return nullptr on failure
        // Validate data_path:
        if (spec.options.find("data_path") == spec.options.end()) {
          Logger& log = Logger::getInstance();
          log.logError("dispatcher: crud handler missing 'data_path' option");
          return nullptr;
        }
        return std::make_unique<CrudHandlerFactory>(spec);
      });
}