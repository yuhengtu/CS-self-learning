#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "link_redirect_request_handler.h"
#include "fakes/link_manager_fake.h"
#include "request.h"
#include "response.h"

static request MakeReq(const std::string& method,
                       const std::string& uri,
                       const std::string& body = "") {
  request r;
  r.method = method;
  r.uri = uri;
  r.version = "1.1";
  r.body = body;
  return r;
}

static int StatusCode(response& resp) {
  std::istringstream iss(resp.get_status_line());
  std::string http, status;
  iss >> http >> status;
  return std::stoi(status);
}

static std::string ExtractLocation(response& resp) {
  auto headers = resp.get_headers();
  std::istringstream iss(headers);
  std::string line;

  while (std::getline(iss, line)) {
    if (line.rfind("Location:", 0) == 0) {

      std::string v = line.substr(strlen("Location:"));

      // trim leading spaces
      while (!v.empty() && (v.front() == ' ' || v.front() == '\t'))
          v.erase(v.begin());

      // REMOVE trailing '\r'
      if (!v.empty() && v.back() == '\r')
          v.pop_back();

      return v;
    }
  }

  return {};
}

TEST(LinkRedirectRequestHandlerTest, MethodNotAllowed) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkRedirectRequestHandler handler("/l", mgr);

  auto resp = handler.handle_request(MakeReq("POST", "/l/abc"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 405);
}

TEST(LinkRedirectRequestHandlerTest, RejectsTrailingSlash) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkRedirectRequestHandler handler("/l", mgr);

  auto resp = handler.handle_request(MakeReq("GET", "/l/"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
}

TEST(LinkRedirectRequestHandlerTest, RejectsEmptyCode) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkRedirectRequestHandler handler("/l", mgr);

  auto resp = handler.handle_request(MakeReq("GET", "/l"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
}

TEST(LinkRedirectRequestHandlerTest, RejectsInvalidCode) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkRedirectRequestHandler handler("/l", mgr);

  auto resp = handler.handle_request(MakeReq("GET", "/l/#illegalcode!"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
}

TEST(LinkRedirectRequestHandlerTest, NotFound) {
  auto mgr = std::make_shared<LinkManagerFake>();
  LinkRedirectRequestHandler handler("/l", mgr);

  auto resp = handler.handle_request(MakeReq("GET", "/l/nonexistantcode"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 404);
}

TEST(LinkRedirectRequestHandlerTest, InvalidStatusTranslatesTo400) {
  class InvalidMgr : public LinkManagerInterface {
   public:
    GetResult Get(const std::string&) override {
      return {LinkStatus::Invalid, std::nullopt};
    }
    CreateResult Create(const LinkCreateParams&) override {
      return {LinkStatus::FsError, std::nullopt};
    }
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
    bool GetUrlVisitCount(const std::string&, std::uint64_t*) override {
      return false;
    }
    bool GetAllUrlVisits(std::vector<std::pair<std::string, std::uint64_t>>*) override {
      return false;
    }
  };
  auto mgr = std::make_shared<InvalidMgr>();
  LinkRedirectRequestHandler handler("/l", mgr);
  auto resp = handler.handle_request(MakeReq("GET", "/l/abcd"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 400);
}

TEST(LinkRedirectRequestHandlerTest, ManagerFsErrorReturns500) {
  class StubMgr : public LinkManagerInterface {
   public:
    CreateResult Create(const LinkCreateParams&) override {
      return {LinkStatus::FsError, std::nullopt};
    }
    GetResult Get(const std::string&) override {
      return {LinkStatus::FsError, std::nullopt};
    }
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
    bool GetUrlVisitCount(const std::string&, std::uint64_t*) override {
      return false;
    }
    bool GetAllUrlVisits(std::vector<std::pair<std::string, std::uint64_t>>*) override {
      return false;
    }
  };

  auto mgr = std::make_shared<StubMgr>();
  LinkRedirectRequestHandler handler("/l", mgr);

  auto resp = handler.handle_request(MakeReq("GET", "/l/abcde"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 500);
}

TEST(LinkRedirectRequestHandlerTest, RedirectHappyPath) {
  auto mgr = std::make_shared<LinkManagerFake>();
  auto created = mgr->Create(LinkCreateParams{"https://a.com"});
  ASSERT_EQ(created.status, LinkStatus::Ok);
  ASSERT_TRUE(created.code.has_value());

  LinkRedirectRequestHandler handler("/l", mgr);
  auto resp = handler.handle_request(MakeReq("GET", "/l/" + *created.code));

  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 302);
  EXPECT_EQ(ExtractLocation(*resp), "https://a.com");
}

TEST(LinkRedirectRequestHandlerTest, RedirectUpdatesCodeVisitCount) {
  auto mgr = std::make_shared<LinkManagerFake>();
  auto created = mgr->Create(LinkCreateParams{"https://a.com"});
  ASSERT_TRUE(created.code);

  LinkRedirectRequestHandler handler("/l", mgr);
  auto resp = handler.handle_request(MakeReq("GET", "/l/" + *created.code));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 302);

  auto gr = mgr->Get(*created.code);
  ASSERT_EQ(gr.status, LinkStatus::Ok);
  ASSERT_TRUE(gr.record);
  EXPECT_EQ(gr.record->visits, 1u);
}

TEST(LinkRedirectRequestHandlerTest, FullFlowRemainsWorking) {
  auto mgr = std::make_shared<LinkManagerFake>();

  auto created = mgr->Create(LinkCreateParams{"https://a.com"});
  ASSERT_EQ(created.status, LinkStatus::Ok);
  ASSERT_TRUE(created.code.has_value());
  std::string code = *created.code;

  LinkRedirectRequestHandler handler("/l", mgr);

  auto get1 = handler.handle_request(MakeReq("GET", "/l/" + code));
  ASSERT_NE(get1, nullptr);
  EXPECT_EQ(StatusCode(*get1), 302);
  EXPECT_EQ(ExtractLocation(*get1), "https://a.com");

  auto updated = mgr->Update(code, LinkUpdateParams{"https://b.com"});
  ASSERT_EQ(updated.status, LinkStatus::Ok);

  auto get2 = handler.handle_request(MakeReq("GET", "/l/" + code));
  ASSERT_NE(get2, nullptr);
  EXPECT_EQ(StatusCode(*get2), 302);
  EXPECT_EQ(ExtractLocation(*get2), "https://b.com");

  auto del = mgr->Delete(code);
  ASSERT_EQ(del.status, LinkStatus::Ok);

  auto get404 = handler.handle_request(MakeReq("GET", "/l/" + code));
  ASSERT_NE(get404, nullptr);
  EXPECT_EQ(StatusCode(*get404), 404);
}

TEST(LinkRedirectRequestHandlerTest, RedirectIncrementsCodeAndUrlCounters) {
  class CountingMgr : public LinkManagerInterface {
   public:
    mutable int code_increments = 0;
    mutable int url_increments = 0;
    LinkRecord record{"abcd", "https://a.com", 0};

    CreateResult Create(const LinkCreateParams&) override {
      return {LinkStatus::FsError, std::nullopt};
    }
    GetResult Get(const std::string&) override {
      return {LinkStatus::Ok, record};
    }
    UpdateResult Update(const std::string&, const LinkUpdateParams&) override {
      return {LinkStatus::FsError};
    }
    DeleteResult Delete(const std::string&) override {
      return {LinkStatus::FsError};
    }
    ResolveResult Resolve(const std::string&, bool) override {
      return {LinkStatus::FsError, std::nullopt};
    }
    bool IncrementCodeVisits(const std::string&) override {
      ++code_increments;
      return true;
    }
    bool IncrementVisits(const std::string&) override {
      ++url_increments;
      return true;
    }
    bool GetUrlVisitCount(const std::string&, std::uint64_t*) override {
      return false;
    }
    bool GetAllUrlVisits(std::vector<std::pair<std::string, std::uint64_t>>*) override {
      return false;
    }
  };

  auto mgr = std::make_shared<CountingMgr>();
  LinkRedirectRequestHandler handler("/l", mgr);
  auto resp = handler.handle_request(MakeReq("GET", "/l/abcd"));
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(StatusCode(*resp), 302);
  EXPECT_EQ(mgr->code_increments, 1);
  EXPECT_EQ(mgr->url_increments, 1);
}
