#include "analytics_handler_factory.h"

#include "analytics_request_handler.h"
#include "handler_registry.h"
#include "handler_types.h"
#include "link_manager_provider.h"
#include "logger.h"

AnalyticsHandlerFactory::AnalyticsHandlerFactory(const HandlerSpec& spec) {
  if (auto it = spec.options.find("data_path"); it != spec.options.end()) {
    data_path_ = it->second;
    manager_ = LinkManagerProvider::GetOrCreate(data_path_);
  }
}

std::unique_ptr<RequestHandler>
AnalyticsHandlerFactory::create(const std::string& location,
                                const std::string& /*url*/) {
  if (!manager_) {
    return nullptr;
  }
  return std::make_unique<AnalyticsRequestHandler>(location, manager_);
}

void RegisterAnalyticsHandlerFactory() {
  HandlerRegistry::Register(
      handler_types::ANALYTICS_HANDLER,
      [](const HandlerSpec& spec) -> std::unique_ptr<RequestHandlerFactory> {
        if (spec.options.find("data_path") == spec.options.end()) {
          Logger::getInstance().logError("dispatcher: analytics missing 'data_path'");
          return nullptr;
        }
        return std::make_unique<AnalyticsHandlerFactory>(spec);
      });
}
