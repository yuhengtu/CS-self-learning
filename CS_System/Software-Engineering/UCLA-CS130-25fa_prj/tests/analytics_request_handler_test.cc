#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <boost/json.hpp>

#include "analytics_handler_factory.h"
#include "analytics_request_handler.h"
#include "fakes/link_manager_fake.h"
#include "handler_registry.h"
#include "handler_types.h"
#include "link_manager_interface.h"
#include "request.h"

#include <boost/filesystem.hpp>

namespace {
namespace json = boost::json;

class StubManager : public LinkManagerInterface {
 public:
  CreateResult Create(const LinkCreateParams&) override {
    return {LinkStatus::FsError, std::nullopt};
  }
  GetResult Get(const std::string&) override { return get_result; }
  UpdateResult Update(const std::string&, const LinkUpdateParams&) override {
    return {LinkStatus::FsError};
  }
  DeleteResult Delete(const std::string&) override {
    return {LinkStatus::FsError};
  }
  ResolveResult Resolve(const std::string&, bool) override {
    return {LinkStatus::FsError, std::nullopt};
  }
  bool IncrementCodeVisits(const std::string&) override { return false; }
  bool IncrementVisits(const std::string&) override { return false; }
  bool GetUrlVisitCount(const std::string&, std::uint64_t* count) override {
    if (!count) return false;
    *count = visit_count_value;
    return url_visit_success;
  }
  bool GetAllUrlVisits(
      std::vector<std::pair<std::string, std::uint64_t>>* stats) override {
    if (!stats) return false;
    if (!all_success) return false;
    *stats = all_visits;
    return true;
  }

  GetResult get_result{LinkStatus::Invalid, std::nullopt};
  bool url_visit_success = true;
  std::uint64_t visit_count_value = 0;
  bool all_success = true;
  std::vector<std::pair<std::string, std::uint64_t>> all_visits;
};

std::string MakeTempDir(const std::string& name) {
  namespace fs = boost::filesystem;
  fs::path dir = fs::temp_directory_path() / name;
  fs::create_directories(dir);
  return dir.string();
}

std::uint64_t ReadCount(const json::value& visits) {
  if (visits.is_uint64()) {
    return visits.as_uint64();
  }
  if (visits.is_int64()) {
    auto v = visits.as_int64();
    return v < 0 ? 0 : static_cast<std::uint64_t>(v);
  }
  return static_cast<std::uint64_t>(visits.as_double());
}

std::uint64_t ReadVisits(const json::object& obj) {
  return ReadCount(obj.at("visits"));
}

std::uint64_t ReadUrlVisits(const json::object& obj) {
  return ReadCount(obj.at("url_visits"));
}

request MakeReq(const std::string& uri) {
  request r;
  r.method = "GET";
  r.uri = uri;
  r.version = "1.1";
  return r;
}

std::string Content(const std::unique_ptr<response>& resp) {
  return resp ? resp->get_content() : std::string();
}

}  // namespace

TEST(AnalyticsRequestHandlerTest, CodeQueryReturnsVisits) {
  auto mgr = std::make_shared<LinkManagerFake>();
  auto created = mgr->Create(LinkCreateParams{"https://example.com"});
  ASSERT_EQ(created.status, LinkStatus::Ok);
  ASSERT_TRUE(created.code);
  mgr->IncrementCodeVisits(*created.code);
  mgr->IncrementVisits(*created.code);

  AnalyticsRequestHandler handler("/analytics", mgr);
  auto resp = handler.handle_request(MakeReq("/analytics/" + *created.code));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 200 OK\r\n");
  json::error_code ec;
  json::value val = json::parse(Content(resp), ec);
  ASSERT_FALSE(ec);
  ASSERT_TRUE(val.is_object());
  const auto& obj = val.as_object();
  EXPECT_STREQ(obj.at("code").as_string().c_str(), created.code->c_str());
  EXPECT_EQ(ReadVisits(obj), 1u);
  EXPECT_EQ(ReadUrlVisits(obj), 1u);
}

TEST(AnalyticsRequestHandlerTest, TopQueryReturnsSortedRecords) {
  auto mgr = std::make_shared<LinkManagerFake>();
  auto a = mgr->Create(LinkCreateParams{"https://a.com"});
  auto b = mgr->Create(LinkCreateParams{"https://b.com"});
  ASSERT_TRUE(a.code && b.code);
  mgr->IncrementVisits(*a.code);
  mgr->IncrementVisits(*a.code);
  mgr->IncrementVisits(*b.code);

  AnalyticsRequestHandler handler("/analytics", mgr);
  auto resp = handler.handle_request(MakeReq("/analytics/top/1"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 200 OK\r\n");
  json::error_code ec;
  json::value val = json::parse(Content(resp), ec);
  ASSERT_FALSE(ec);
  ASSERT_TRUE(val.is_array());
  auto arr = val.as_array();
  ASSERT_EQ(arr.size(), 1u);
  ASSERT_TRUE(arr[0].is_object());
  const auto& obj = arr[0].as_object();
  EXPECT_STREQ(obj.at("url").as_string().c_str(), "https://a.com");
  EXPECT_EQ(ReadVisits(obj), 2u);
}

TEST(AnalyticsRequestHandlerTest, MissingAnalyticsPathReturnsBadRequest) {
  auto mgr = std::make_shared<LinkManagerFake>();
  AnalyticsRequestHandler handler("/analytics", mgr);
  auto resp = handler.handle_request(MakeReq("/analytics"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 400 Bad Request\r\n");
}

TEST(AnalyticsRequestHandlerTest, TopQueryMissingSizeIsBadRequest) {
  auto mgr = std::make_shared<LinkManagerFake>();
  AnalyticsRequestHandler handler("/analytics", mgr);
  auto resp = handler.handle_request(MakeReq("/analytics/top"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 400 Bad Request\r\n");
}

TEST(AnalyticsRequestHandlerTest, TopQueryMalformedPathIsRejected) {
  auto mgr = std::make_shared<LinkManagerFake>();
  AnalyticsRequestHandler handler("/analytics", mgr);
  auto resp = handler.handle_request(MakeReq("/analytics/topabc"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 400 Bad Request\r\n");
}

TEST(AnalyticsRequestHandlerTest, TopQueryRejectsNonNumericCounts) {
  auto mgr = std::make_shared<LinkManagerFake>();
  AnalyticsRequestHandler handler("/analytics", mgr);
  auto resp = handler.handle_request(MakeReq("/analytics/top/not-a-number"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 400 Bad Request\r\n");
}

TEST(AnalyticsRequestHandlerTest, TopQueryRejectsZeroCount) {
  auto mgr = std::make_shared<LinkManagerFake>();
  AnalyticsRequestHandler handler("/analytics", mgr);
  auto resp = handler.handle_request(MakeReq("/analytics/top/0"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 400 Bad Request\r\n");
}

TEST(AnalyticsRequestHandlerTest, TopQueryHandlesManagerFailure) {
  auto mgr = std::make_shared<StubManager>();
  mgr->all_success = false;
  AnalyticsRequestHandler handler("/analytics", mgr);
  auto resp = handler.handle_request(MakeReq("/analytics/top/5"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 500 Internal Server Error\r\n");
}

TEST(AnalyticsRequestHandlerTest, TopQueryBreaksTiesAlphabetically) {
  auto mgr = std::make_shared<StubManager>();
  mgr->all_visits = {
      {"https://b.com", 2},
      {"https://a.com", 2},
      {"https://c.com", 1},
  };
  AnalyticsRequestHandler handler("/analytics", mgr);
  auto resp = handler.handle_request(MakeReq("/analytics/top/3"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 200 OK\r\n");
  json::error_code ec;
  json::value val = json::parse(Content(resp), ec);
  ASSERT_FALSE(ec);
  auto arr = val.as_array();
  ASSERT_EQ(arr.size(), 3u);
  EXPECT_STREQ(arr[0].as_object().at("url").as_string().c_str(),
               "https://a.com");
  EXPECT_STREQ(arr[1].as_object().at("url").as_string().c_str(),
               "https://b.com");
  EXPECT_STREQ(arr[2].as_object().at("url").as_string().c_str(),
               "https://c.com");
}

TEST(AnalyticsRequestHandlerTest, CodeQueryRejectsInvalidCode) {
  auto mgr = std::make_shared<LinkManagerFake>();
  AnalyticsRequestHandler handler("/analytics", mgr);
  auto resp = handler.handle_request(MakeReq("/analytics/invalid-$"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 400 Bad Request\r\n");
}

TEST(AnalyticsRequestHandlerTest, CodeQueryReturnsNotFound) {
  auto mgr = std::make_shared<StubManager>();
  mgr->get_result = {LinkStatus::NotFound, std::nullopt};
  AnalyticsRequestHandler handler("/analytics", mgr);
  auto resp = handler.handle_request(MakeReq("/analytics/abcd"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 404 Not Found\r\n");
}

TEST(AnalyticsRequestHandlerTest, CodeQueryHandlesManagerFailure) {
  auto mgr = std::make_shared<StubManager>();
  mgr->get_result = {LinkStatus::FsError, std::nullopt};
  AnalyticsRequestHandler handler("/analytics", mgr);
  auto resp = handler.handle_request(MakeReq("/analytics/abcd"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 500 Internal Server Error\r\n");
}

TEST(AnalyticsRequestHandlerTest, CodeQueryHandlesVisitLookupFailure) {
  auto mgr = std::make_shared<StubManager>();
  LinkRecord record;
  record.code = "abcd";
  record.url = "https://example.com";
  mgr->get_result = {LinkStatus::Ok, record};
  mgr->url_visit_success = false;
  AnalyticsRequestHandler handler("/analytics", mgr);
  auto resp = handler.handle_request(MakeReq("/analytics/abcd"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 500 Internal Server Error\r\n");
}

TEST(AnalyticsHandlerFactoryTest, CreatesHandlerWhenDataPathProvided) {
  HandlerSpec spec;
  spec.name = "analytics";
  spec.path = "/analytics";
  spec.type = handler_types::ANALYTICS_HANDLER;
  spec.options["data_path"] = MakeTempDir("analytics_factory_test");

  AnalyticsHandlerFactory factory(spec);
  auto handler = factory.create(spec.path, spec.path);
  ASSERT_NE(handler, nullptr);
  EXPECT_EQ(handler->name(), "analytics");
}

TEST(AnalyticsHandlerFactoryTest, MissingDataPathReturnsNullFactory) {
  HandlerSpec spec;
  spec.name = "analytics";
  spec.path = "/analytics";
  spec.type = handler_types::ANALYTICS_HANDLER;

  auto factory = HandlerRegistry::CreateFactory(spec);
  EXPECT_EQ(factory, nullptr);
}
