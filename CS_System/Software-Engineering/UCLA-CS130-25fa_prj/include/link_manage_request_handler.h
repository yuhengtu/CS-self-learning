#ifndef LINK_MANAGE_REQUEST_HANDLER_H
#define LINK_MANAGE_REQUEST_HANDLER_H

#include <memory>
#include <string>

#include "request_handler.h"
#include "link_manager_interface.h"

class LinkManageRequestHandler : public RequestHandler {
public:
  LinkManageRequestHandler(std::string mount_prefix,
                           std::shared_ptr<LinkManagerInterface> manager);

  std::unique_ptr<response> handle_request(const request& req) override;
  std::string name() const override { return "link_manage"; }

private:
  std::string mount_prefix_;
  std::shared_ptr<LinkManagerInterface> manager_;

  std::unique_ptr<response> HandlePost(const request& req);
  std::unique_ptr<response> HandleGet(const std::string& code, const request& req);
  std::unique_ptr<response> HandlePut(const std::string& code, const request& req);
  std::unique_ptr<response> HandleDelete(const std::string& code, const request& req);
};

#endif // LINK_MANAGE_REQUEST_HANDLER_H
