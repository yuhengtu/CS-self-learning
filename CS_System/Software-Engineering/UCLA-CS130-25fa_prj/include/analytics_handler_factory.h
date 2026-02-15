#ifndef ANALYTICS_HANDLER_FACTORY_H
#define ANALYTICS_HANDLER_FACTORY_H

#include <memory>
#include <string>

#include "link_manager_interface.h"
#include "request_handler_factory.h"
#include "server_config.h"

class AnalyticsHandlerFactory : public RequestHandlerFactory {
public:
  explicit AnalyticsHandlerFactory(const HandlerSpec& spec);
  std::unique_ptr<RequestHandler> create(const std::string& location,
                                         const std::string& url) override;

private:
  std::string data_path_;
  std::shared_ptr<LinkManagerInterface> manager_;
};

void RegisterAnalyticsHandlerFactory();

#endif  // ANALYTICS_HANDLER_FACTORY_H
