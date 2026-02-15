#include <gtest/gtest.h>
#include <boost/json.hpp>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

#include "link_manage_request_handler.h"
#include "fakes/link_manager_fake.h"
#include "request.h"

namespace json = boost::json;

static request MakeReq(const std::string& method, const std::string& uri, const std::string& body = "") {
  request r; r.method = method; r.uri = uri; r.version = "1.1"; r.body = body; return r;
}

static int StatusCode(response& resp) {
  std::istringstream iss(resp.get_status_line());
  std::string http; std::string status;
  iss >> http >> status;
  return std::stoi(status);
}

static std::string ExtractCode(response& resp) {
  json::error_code ec;
  auto v = json::parse(resp.get_content(), ec);
  if (ec || !v.is_object()) return {};
  auto* it = v.as_object().if_contains("code");
  if (!it || !it->is_string()) return {};
  return std::string(it->as_string().c_str());
}

static std::string ExtractUrl(response& resp) {
  json::error_code ec;
  auto v = json::parse(resp.get_content(), ec);
  if (ec || !v.is_object()) return {};
  auto* it = v.as_object().if_contains("url");
  if (!it || !it->is_string()) return {};
  return std::string(it->as_string().c_str());
}

static std::uint64_t ExtractVisits(response& resp) {
  json::error_code ec;
  auto v = json::parse(resp.get_content(), ec);
  if (ec || !v.is_object()) return 0;
  auto* it = v.as_object().if_contains("visits");
  if (!it) return 0;
  if (it->is_uint64()) return it->as_uint64();
  if (it->is_int64()) {
    auto val = it->as_int64();
    return val < 0 ? 0 : static_cast<std::uint64_t>(val);
  }
  return 0;
}

static bool ExtractPasswordProtected(response& resp) {
  json::error_code ec;
  auto v = json::parse(resp.get_content(), ec);
  if (ec || !v.is_object()) return false;
  auto* it = v.as_object().if_contains("password_protected");
  if (!it || !it->is_bool()) return false;
  return it->as_bool();
}

static std::string HeaderValue(response& resp, const std::string& name) {
  std::istringstream iss(resp.get_headers());
  std::string line;
  const std::string prefix = name + ": ";
  while (std::getline(iss, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.rfind(prefix, 0) == 0) {
      return line.substr(prefix.size());
    }
  }
  return {};
}

TEST(LinkManageRequestHandlerTest, CreateHappyPath) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("POST", "/link", "{\"url\":\"https://a.com\"}"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 200);
  std::string code = ExtractCode(*resp);
  ASSERT_FALSE(code.empty());
  auto get = mgr->Get(code);
  EXPECT_EQ(get.status, LinkStatus::Ok);
  ASSERT_TRUE(get.record.has_value());
  EXPECT_EQ(get.record->url, "https://a.com");
}

TEST(LinkManageRequestHandlerTest, CreateRejectsInvalidJson) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("POST", "/link", "{bad"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
}

TEST(LinkManageRequestHandlerTest, CreateRejectsMissingUrl) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("POST", "/link", "{\"foo\":1}"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
}

TEST(LinkManageRequestHandlerTest, CreateRejectsInvalidUrl) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  // URL with spaces should fail validation
  auto resp = handler.handle_request(MakeReq("POST", "/link", "{\"url\":\"not a url\"}"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
}

TEST(LinkManageRequestHandlerTest, CreateRejectsEmptyPassword) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("POST", "/link", "{\"url\":\"https://a.com\",\"password\":\"\"}"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
}

TEST(LinkManageRequestHandlerTest, BasePathGetIsMethodNotAllowed) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("GET", "/link"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 405);
  EXPECT_EQ(HeaderValue(*resp, "Allow"), "POST, GET, PUT, DELETE");
}

TEST(LinkManageRequestHandlerTest, GetHappyPath) {
  auto mgr = std::make_shared<LinkManagerFake>();
  auto created = mgr->Create(LinkCreateParams{"https://a.com"});
  ASSERT_EQ(created.status, LinkStatus::Ok);
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("GET", "/link/" + *created.code));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 200);
  EXPECT_EQ(ExtractUrl(*resp), "https://a.com");
  EXPECT_EQ(ExtractVisits(*resp), 0u);
  EXPECT_FALSE(ExtractPasswordProtected(*resp));
}

TEST(LinkManageRequestHandlerTest, GetInvalidCode) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("GET", "/link/bad!"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
}

TEST(LinkManageRequestHandlerTest, GetNotFound) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("GET", "/link/abc"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 404);
}

TEST(LinkManageRequestHandlerTest, GetProtectedLinkRequiresPassword) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto create = handler.handle_request(MakeReq("POST", "/link", "{\"url\":\"https://a.com\",\"password\":\"secret\"}"));
  ASSERT_NE(create, nullptr);
  EXPECT_EQ(StatusCode(*create), 200);
  const std::string code = ExtractCode(*create);
  ASSERT_FALSE(code.empty());

  auto unauth = MakeReq("GET", "/link/" + code);
  auto unauth_resp = handler.handle_request(unauth);
  ASSERT_NE(unauth_resp, nullptr);
  EXPECT_EQ(StatusCode(*unauth_resp), 403);

  auto wrong = MakeReq("GET", "/link/" + code);
  wrong.headers.emplace_back("Link-Password", "wrong");
  auto wrong_resp = handler.handle_request(wrong);
  ASSERT_NE(wrong_resp, nullptr);
  EXPECT_EQ(StatusCode(*wrong_resp), 403);

  auto auth = MakeReq("GET", "/link/" + code);
  auth.headers.emplace_back("Link-Password", "secret");
  auto auth_resp = handler.handle_request(auth);
  ASSERT_NE(auth_resp, nullptr);
  EXPECT_EQ(StatusCode(*auth_resp), 200);
  EXPECT_TRUE(ExtractPasswordProtected(*auth_resp));
}

TEST(LinkManageRequestHandlerTest, PutHappyPath) {
  auto mgr = std::make_shared<LinkManagerFake>();
  auto created = mgr->Create(LinkCreateParams{"https://a.com"});
  ASSERT_EQ(created.status, LinkStatus::Ok);
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("PUT", "/link/" + *created.code, "{\"url\":\"https://b.com\"}"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 200);
  auto get = mgr->Get(*created.code);
  ASSERT_TRUE(get.record.has_value());
  EXPECT_EQ(get.record->url, "https://b.com");
}

TEST(LinkManageRequestHandlerTest, PutRejectsInvalidJson) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("PUT", "/link/abc", "{bad"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
}

TEST(LinkManageRequestHandlerTest, PutRejectsInvalidUrl) {
  auto mgr = std::make_shared<LinkManagerFake>();
  auto created = mgr->Create(LinkCreateParams{"https://a.com"});
  ASSERT_EQ(created.status, LinkStatus::Ok);
  LinkManageRequestHandler handler("/link", mgr);
  // URL with spaces should fail validation
  auto resp = handler.handle_request(
      MakeReq("PUT", "/link/" + *created.code, "{\"url\":\"not a url\"}"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
}

TEST(LinkManageRequestHandlerTest, PutInvalidCode) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("PUT", "/link/ab$",
                                             "{\"url\":\"https://a.com\"}"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
}

TEST(LinkManageRequestHandlerTest, PutNotFound) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("PUT", "/link/abc", "{\"url\":\"https://b.com\"}"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 404);
}

TEST(LinkManageRequestHandlerTest, PutProtectedLinkRequiresPassword) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto create = handler.handle_request(MakeReq("POST", "/link", "{\"url\":\"https://a.com\",\"password\":\"secret\"}"));
  ASSERT_NE(create, nullptr);
  const std::string code = ExtractCode(*create);
  ASSERT_FALSE(code.empty());

  auto unauth = handler.handle_request(MakeReq("PUT", "/link/" + code, "{\"url\":\"https://b.com\"}"));
  ASSERT_NE(unauth, nullptr);
  EXPECT_EQ(StatusCode(*unauth), 403);
  auto rec = mgr->Get(code);
  ASSERT_TRUE(rec.record);
  EXPECT_EQ(rec.record->url, "https://a.com");

  auto auth_req = MakeReq("PUT", "/link/" + code, "{\"url\":\"https://b.com\"}");
  auth_req.headers.emplace_back("Link-Password", "secret");
  auto auth_resp = handler.handle_request(auth_req);
  ASSERT_NE(auth_resp, nullptr);
  EXPECT_EQ(StatusCode(*auth_resp), 200);
  rec = mgr->Get(code);
  ASSERT_TRUE(rec.record);
  EXPECT_EQ(rec.record->url, "https://b.com");
}

TEST(LinkManageRequestHandlerTest, DeleteHappyPath) {
  auto mgr = std::make_shared<LinkManagerFake>();
  auto created = mgr->Create(LinkCreateParams{"https://a.com"});
  ASSERT_EQ(created.status, LinkStatus::Ok);
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("DELETE", "/link/" + *created.code));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 200);
  auto get = mgr->Get(*created.code);
  EXPECT_EQ(get.status, LinkStatus::NotFound);
}

TEST(LinkManageRequestHandlerTest, DeleteInvalidCode) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("DELETE", "/link/!bad"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
}

TEST(LinkManageRequestHandlerTest, DeleteMissingCodeIsMethodNotAllowed) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("DELETE", "/link"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 405);
}

TEST(LinkManageRequestHandlerTest, DeleteNotFoundStillOk) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("DELETE", "/link/abc"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 200);
}

TEST(LinkManageRequestHandlerTest, DeleteProtectedLinkRequiresPassword) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto create = handler.handle_request(MakeReq("POST", "/link", "{\"url\":\"https://a.com\",\"password\":\"secret\"}"));
  ASSERT_NE(create, nullptr);
  const std::string code = ExtractCode(*create);
  ASSERT_FALSE(code.empty());

  auto unauth = handler.handle_request(MakeReq("DELETE", "/link/" + code));
  ASSERT_NE(unauth, nullptr);
  EXPECT_EQ(StatusCode(*unauth), 403);

  auto auth_req = MakeReq("DELETE", "/link/" + code);
  auth_req.headers.emplace_back("Link-Password", "secret");
  auto auth_resp = handler.handle_request(auth_req);
  ASSERT_NE(auth_resp, nullptr);
  EXPECT_EQ(StatusCode(*auth_resp), 200);
  auto get = mgr->Get(code);
  EXPECT_EQ(get.status, LinkStatus::NotFound);
}

TEST(LinkManageRequestHandlerTest, FullFlowRemainsWorking) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  auto create = handler.handle_request(MakeReq("POST", "/link", "{\"url\":\"https://a.com\"}"));
  ASSERT_NE(create, nullptr);
  std::string code = ExtractCode(*create);
  ASSERT_FALSE(code.empty());
  auto update = handler.handle_request(MakeReq("PUT", "/link/" + code, "{\"url\":\"https://b.com\"}"));
  EXPECT_EQ(StatusCode(*update), 200);
  auto get = handler.handle_request(MakeReq("GET", "/link/" + code));
  EXPECT_EQ(ExtractUrl(*get), "https://b.com");
  auto del = handler.handle_request(MakeReq("DELETE", "/link/" + code));
  EXPECT_EQ(StatusCode(*del), 200);
  auto get404 = handler.handle_request(MakeReq("GET", "/link/" + code));
  EXPECT_EQ(StatusCode(*get404), 404);
}

TEST(LinkManageRequestHandlerTest, UnsupportedMethodReturns405) {
  auto mgr = std::make_shared<LinkManagerFake>();
  auto created = mgr->Create(LinkCreateParams{"https://a.com"});
  ASSERT_EQ(created.status, LinkStatus::Ok);
  LinkManageRequestHandler handler("/link", mgr);
  auto resp = handler.handle_request(MakeReq("PATCH", "/link/" + *created.code));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 405);
  EXPECT_EQ(HeaderValue(*resp, "Allow"), "POST, GET, PUT, DELETE");
}

TEST(LinkManageRequestHandlerTest, CreateNormalizesSchemaLessUrl) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  
  // POST with "google.com" (no http://)
  auto resp = handler.handle_request(MakeReq("POST", "/link", "{\"url\":\"google.com\"}"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 200);
  
  std::string code = ExtractCode(*resp);
  ASSERT_FALSE(code.empty());
  
  // Verify manager received the normalized URL
  auto get = mgr->Get(code);
  EXPECT_EQ(get.status, LinkStatus::Ok);
  EXPECT_EQ(get.record->url, "http://google.com");
}

TEST(LinkManageRequestHandlerTest, UpdateNormalizesSchemaLessUrl) {
  auto mgr = std::make_shared<LinkManagerFake>();
  auto created = mgr->Create(LinkCreateParams{"http://old.com"});
  ASSERT_EQ(created.status, LinkStatus::Ok);
  
  LinkManageRequestHandler handler("/link", mgr);
  
  // PUT with "new.com" (no http://)
  auto resp = handler.handle_request(
      MakeReq("PUT", "/link/" + *created.code, "{\"url\":\"new.com\"}"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 200);
  
  auto get = mgr->Get(*created.code);
  EXPECT_EQ(get.status, LinkStatus::Ok);
  EXPECT_EQ(get.record->url, "http://new.com");
}

TEST(LinkManageRequestHandlerTest, CreateRejectsUnsupportedScheme) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkManageRequestHandler handler("/link", mgr);
  
  // Try creating with ftp:// scheme
  auto resp = handler.handle_request(MakeReq("POST", "/link", "{\"url\":\"ftp://files.example.com\"}"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
}

TEST(LinkManageRequestHandlerTest, UpdateRejectsUnsupportedScheme) {
  auto mgr = std::make_shared<LinkManagerFake>();
  auto created = mgr->Create(LinkCreateParams{"http://old.com"});
  ASSERT_EQ(created.status, LinkStatus::Ok);
  
  LinkManageRequestHandler handler("/link", mgr);
  
  // Try updating with ftp:// scheme
  auto resp = handler.handle_request(
      MakeReq("PUT", "/link/" + *created.code, "{\"url\":\"ftp://files.example.com\"}"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
  
  // Verify URL wasn't changed
  auto get = mgr->Get(*created.code);
  EXPECT_EQ(get.status, LinkStatus::Ok);
  EXPECT_EQ(get.record->url, "http://old.com");
}
