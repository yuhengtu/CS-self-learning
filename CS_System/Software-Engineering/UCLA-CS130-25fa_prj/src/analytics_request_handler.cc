#include "analytics_request_handler.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <boost/json.hpp>
#include <sstream>
#include <vector>

#include "response_builder.h"

namespace json = boost::json;

namespace {
std::string TrimPrefix(const std::string& uri, const std::string& prefix) {
  if (uri.rfind(prefix, 0) == 0) {
    return uri.substr(prefix.size());
  }
  return uri;
}
}

AnalyticsRequestHandler::AnalyticsRequestHandler(
    std::string mount_prefix,
    std::shared_ptr<LinkManagerInterface> manager)
    : mount_prefix_(std::move(mount_prefix)), manager_(std::move(manager)) {}

std::unique_ptr<response> AnalyticsRequestHandler::handle_request(
    const request& req) {
  auto out = std::make_unique<response>();
  std::string rel = TrimPrefix(req.uri, mount_prefix_);
  if (!rel.empty() && rel.front() == '/') {
    rel = rel.substr(1);
  }
  if (rel.empty()) {
    ResponseBuilder::createBadRequest("missing analytics path").build(*out);
    return out;
  }

  if (rel.rfind("top", 0) == 0) {
    std::string count_str;
    if (rel.size() == 3) {
      ResponseBuilder::createBadRequest("missing leaderboard size").build(*out);
      return out;
    }
    if (rel[3] == '/') {
      count_str = rel.substr(4);
    } else {
      ResponseBuilder::createBadRequest("malformed analytics path").build(*out);
      return out;
    }
    return HandleTopQuery(count_str);
  }

  return HandleCodeQuery(rel);
}

std::unique_ptr<response> AnalyticsRequestHandler::HandleCodeQuery(
    const std::string& code) {
  auto out = std::make_unique<response>();
  if (!LinkManagerInterface::IsValidCode(code)) {
    ResponseBuilder::createBadRequest("invalid code").build(*out);
    return out;
  }
  auto gr = manager_->Get(code);
  if (gr.status == LinkStatus::NotFound) {
    ResponseBuilder::createNotFound().build(*out);
    return out;
  }
  if (gr.status != LinkStatus::Ok || !gr.record) {
    ResponseBuilder::createInternalServerError().build(*out);
    return out;
  }
  const std::uint64_t code_visits = gr.record->visits;
  std::uint64_t url_visits = 0;
  if (!manager_->GetUrlVisitCount(gr.record->url, &url_visits)) {
    ResponseBuilder::createInternalServerError().build(*out);
    return out;
  }
  json::object obj;
  obj["code"] = gr.record->code;
  obj["url"] = gr.record->url;
  obj["visits"] = code_visits;
  obj["url_visits"] = url_visits;
  ResponseBuilder::createOk()
      .withContentType("application/json")
      .withBody(json::serialize(obj))
      .build(*out);
  return out;
}

std::unique_ptr<response> AnalyticsRequestHandler::HandleTopQuery(
    const std::string& count_str) {
  auto out = std::make_unique<response>();
  if (count_str.empty()) {
    ResponseBuilder::createBadRequest("missing leaderboard size").build(*out);
    return out;
  }
  char* end = nullptr;
  long requested = std::strtol(count_str.c_str(), &end, 10);
  if (end == count_str.c_str() || requested <= 0) {
    ResponseBuilder::createBadRequest("invalid leaderboard size").build(*out);
    return out;
  }
  std::vector<std::pair<std::string, std::uint64_t>> stats;
  if (!manager_->GetAllUrlVisits(&stats)) {
    ResponseBuilder::createInternalServerError().build(*out);
    return out;
  }
  std::sort(stats.begin(), stats.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.second == rhs.second) {
                return lhs.first < rhs.first;
              }
              return lhs.second > rhs.second;
            });
  if (static_cast<size_t>(requested) < stats.size()) {
    stats.resize(static_cast<size_t>(requested));
  }
  json::array arr;
  for (const auto& entry : stats) {
    json::object obj;
    obj["url"] = entry.first;
    obj["visits"] = entry.second;
    arr.emplace_back(obj);
  }
  ResponseBuilder::createOk()
      .withContentType("application/json")
      .withBody(json::serialize(arr))
      .build(*out);
  return out;
}
