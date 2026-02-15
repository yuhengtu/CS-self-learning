#ifndef ANALYTICS_REQUEST_HANDLER_H
#define ANALYTICS_REQUEST_HANDLER_H

#include <memory>
#include <string>

#include "link_manager_interface.h"
#include "request_handler.h"

class AnalyticsRequestHandler : public RequestHandler {
public:
  AnalyticsRequestHandler(std::string mount_prefix,
                          std::shared_ptr<LinkManagerInterface> manager);

  std::unique_ptr<response> handle_request(const request& req) override;
  std::string name() const override { return "analytics"; }

private:
  std::string mount_prefix_;
  std::shared_ptr<LinkManagerInterface> manager_;

  std::unique_ptr<response> HandleCodeQuery(const std::string& code);
  std::unique_ptr<response> HandleTopQuery(const std::string& count_str);
};

#endif  // ANALYTICS_REQUEST_HANDLER_H
