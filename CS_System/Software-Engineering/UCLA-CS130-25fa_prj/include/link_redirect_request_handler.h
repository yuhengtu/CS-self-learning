#ifndef LINK_REDIRECT_REQUEST_HANDLER_H
#define LINK_REDIRECT_REQUEST_HANDLER_H

#include <memory>
#include <string>

#include "request_handler.h"
#include "link_manager_interface.h"

class LinkRedirectRequestHandler : public RequestHandler {
public:
  LinkRedirectRequestHandler(std::string mount_prefix,
                           std::shared_ptr<LinkManagerInterface> manager);

  std::unique_ptr<response> handle_request(const request& req) override;
  std::string name() const override { return "link_redirect"; }

private:
  std::string mount_prefix_;
  std::shared_ptr<LinkManagerInterface> manager_;
};

#endif // LINK_REDIRECT_REQUEST_HANDLER_H
