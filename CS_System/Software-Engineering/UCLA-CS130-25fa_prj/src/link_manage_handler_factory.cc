#include "link_manage_handler_factory.h"

#include "handler_registry.h"
#include "handler_types.h"
#include "logger.h"
#include "link_manager_provider.h"
#include "link_manage_request_handler.h"

LinkManageHandlerFactory::LinkManageHandlerFactory(const HandlerSpec& spec) {
  if (auto it = spec.options.find("data_path"); it != spec.options.end()) {
    data_path_ = it->second;
    manager_ = LinkManagerProvider::GetOrCreate(data_path_);
  }
}

std::unique_ptr<RequestHandler>
LinkManageHandlerFactory::create(const std::string& location,
                                 const std::string& /*url*/) {
  return std::make_unique<LinkManageRequestHandler>(location, manager_);
}

void RegisterLinkManageHandlerFactory() {
  HandlerRegistry::Register(
      handler_types::LINK_MANAGE_HANDLER,
      [](const HandlerSpec& spec) -> std::unique_ptr<RequestHandlerFactory> {
        if (spec.options.find("data_path") == spec.options.end()) {
          Logger::getInstance().logError("dispatcher: link_manage missing 'data_path'");
          return nullptr;
        }
        return std::make_unique<LinkManageHandlerFactory>(spec);
      });
}

