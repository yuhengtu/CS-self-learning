#ifndef LINK_MANAGE_HANDLER_FACTORY_H
#define LINK_MANAGE_HANDLER_FACTORY_H

#include <memory>
#include <string>

#include "request_handler_factory.h"
#include "server_config.h"
#include "link_manager_interface.h"

class LinkManageHandlerFactory : public RequestHandlerFactory {
public:
  explicit LinkManageHandlerFactory(const HandlerSpec& spec);
  std::unique_ptr<RequestHandler> create(const std::string& location,
                                         const std::string& url) override;
private:
  std::string data_path_;
  std::shared_ptr<LinkManagerInterface> manager_;
};

void RegisterLinkManageHandlerFactory();

#endif // LINK_MANAGE_HANDLER_FACTORY_H

