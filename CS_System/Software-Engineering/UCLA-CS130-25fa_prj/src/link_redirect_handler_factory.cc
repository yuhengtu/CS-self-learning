#include "link_redirect_handler_factory.h"

#include "handler_registry.h"
#include "handler_types.h"
#include "logger.h"
#include "link_manager_provider.h"
#include "link_redirect_request_handler.h"

LinkRedirectHandlerFactory::LinkRedirectHandlerFactory(const HandlerSpec& spec) {
  if (auto it = spec.options.find("data_path"); it != spec.options.end()) {
    data_path_ = it->second;
    manager_ = LinkManagerProvider::GetOrCreate(data_path_);
  }
}

std::unique_ptr<RequestHandler>
LinkRedirectHandlerFactory::create(const std::string& location,
                                 const std::string& /*url*/) {
  return std::make_unique<LinkRedirectRequestHandler>(location, manager_);
}

void RegisterLinkRedirectHandlerFactory() {
  HandlerRegistry::Register(
      handler_types::LINK_REDIRECT_HANDLER,
      [](const HandlerSpec& spec) -> std::unique_ptr<RequestHandlerFactory> {
        if (spec.options.find("data_path") == spec.options.end()) {
          Logger::getInstance().logError("dispatcher: link_redirect missing 'data_path'");
          return nullptr;
        }
        return std::make_unique<LinkRedirectHandlerFactory>(spec);
      });
}

