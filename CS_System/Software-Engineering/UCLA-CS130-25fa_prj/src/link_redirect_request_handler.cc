#include "link_redirect_request_handler.h"

#include <boost/json.hpp>
#include <sstream>
#include "logger.h"

#include "response_builder.h"

namespace json = boost::json;

static std::string TrimPrefix(const std::string& s, const std::string& prefix) {
  if (s.rfind(prefix, 0) == 0) return s.substr(prefix.size());
  return s;
}

LinkRedirectRequestHandler::LinkRedirectRequestHandler(
    std::string mount_prefix,
    std::shared_ptr<LinkManagerInterface> manager)
    : mount_prefix_(std::move(mount_prefix)), manager_(std::move(manager)) {}

std::unique_ptr<response> LinkRedirectRequestHandler::handle_request(const request& req) {
  Logger& log = Logger::getInstance();

  auto out = std::make_unique<response>();
  const std::string rel = TrimPrefix(req.uri, mount_prefix_);
  
  // /l endpoint only handling GET redirects, management operations in /api/link
  if (req.method != "GET") {
    log.logTrace("Invalid method " + req.method + " requested to /l endpoint");
    ResponseBuilder(405, "Method Not Allowed").withHeader("Allow", "GET").build(*out);
    return out;
  }

  std::string code;
  if (!rel.empty()) {
    if (rel[0] == '/') code = rel.substr(1);
    else code = rel;
  }
  else {
    log.logTrace("/l requested with no code");
    ResponseBuilder(400, "Bad Request").withBody("empty code").build(*out);
    return out;
  }

  // lightweight precheck, don't invoke the manager if invalid code
  if (!LinkManagerInterface::IsValidCode(code)) {
    log.logTrace("Invalid code " + code + " requested to /l endpoint");
    ResponseBuilder::createBadRequest("invalid code").build(*out);
    return out;
  }

  auto gr = manager_->Get(code);
  if (gr.status == LinkStatus::Ok && gr.record) { // happy case
    log.logTrace("Found valid code -> url mapping in fs");
    manager_->IncrementCodeVisits(code);
    manager_->IncrementVisits(code);
    ResponseBuilder(302, "Found")
        .withHeader("Location", gr.record->url)
        .build(*out);
    return out;
  }
  if (gr.status == LinkStatus::NotFound) { // code -> url mapping doesn't exist
    log.logTrace("Code " + code + " not found");
    ResponseBuilder::createNotFound().build(*out);
    return out;
  }
  if (gr.status == LinkStatus::Invalid) { 
    log.logTrace("Invalid code " + code + " requested to /l endpoint");
    ResponseBuilder::createBadRequest("Invalid code").build(*out);
    return out;
  }
  if (gr.status == LinkStatus::FsError) { // internal server error, flag in log
    log.logWarning("FsError when requesting for code " + code);
    ResponseBuilder::createInternalServerError("Filesystem error").build(*out);
    return out;
  }
  
  // should never get here
  log.logWarning("Impossible branch hit, building internal server error response");
  ResponseBuilder::createInternalServerError().build(*out);
  return out;
}
