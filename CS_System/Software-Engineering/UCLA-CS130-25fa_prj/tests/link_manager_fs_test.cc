#include <gtest/gtest.h>
#include <boost/filesystem.hpp>
#include <cstdint>
#include <string>
#ifdef __unix__
#include <unistd.h>
#endif
#include <vector>

#include "link_manager.h"

namespace fs = boost::filesystem;

class LinkManagerFsTest : public ::testing::Test {
protected:
  fs::path tmp_root;

  void SetUp() override {
    tmp_root = fs::path("/tmp/link_manager_fs_test_") / std::to_string(::getpid());
    fs::remove_all(tmp_root);
    fs::create_directories(tmp_root);
  }
  void TearDown() override {
    fs::remove_all(tmp_root);
  }
};

TEST_F(LinkManagerFsTest, CreateAndResolve) {
  LinkManager mgr(tmp_root.string());

  auto cr = mgr.Create(LinkCreateParams{"https://example.org"});
  ASSERT_EQ(cr.status, LinkStatus::Ok);
  ASSERT_TRUE(cr.code.has_value());
  const std::string code = *cr.code;

  auto rr = mgr.Resolve(code, true);
  ASSERT_EQ(rr.status, LinkStatus::Ok);
  ASSERT_TRUE(rr.url.has_value());
  EXPECT_EQ(*rr.url, "https://example.org");
}

TEST_F(LinkManagerFsTest, GetReturnsRecord) {
  LinkManager mgr(tmp_root.string());
  auto cr = mgr.Create(LinkCreateParams{"https://example.org"});
  ASSERT_EQ(cr.status, LinkStatus::Ok);
  const std::string code = *cr.code;

  auto gr = mgr.Get(code);
  ASSERT_EQ(gr.status, LinkStatus::Ok);
  ASSERT_TRUE(gr.record.has_value());
  EXPECT_EQ(gr.record->code, code);
  EXPECT_EQ(gr.record->url, "https://example.org");
}

TEST_F(LinkManagerFsTest, UpdateHappyPath) {
  LinkManager mgr(tmp_root.string());
  auto cr = mgr.Create(LinkCreateParams{"https://one.com"});
  ASSERT_EQ(cr.status, LinkStatus::Ok);
  const std::string code = *cr.code;

  auto ur = mgr.Update(code, LinkUpdateParams{"https://two.com"});
  EXPECT_EQ(ur.status, LinkStatus::Ok);
  auto gr = mgr.Get(code);
  ASSERT_EQ(gr.status, LinkStatus::Ok);
  ASSERT_TRUE(gr.record.has_value());
  EXPECT_EQ(gr.record->url, "https://two.com");
}

TEST_F(LinkManagerFsTest, UpdateMissingReturnsNotFound) {
  LinkManager mgr(tmp_root.string());
  auto ur = mgr.Update("abc", LinkUpdateParams{"https://x.com"});
  EXPECT_EQ(ur.status, LinkStatus::NotFound);
}

TEST_F(LinkManagerFsTest, DeleteIdempotent) {
  LinkManager mgr(tmp_root.string());
  // Delete missing is Ok (idempotent)
  auto dr1 = mgr.Delete("nope");
  EXPECT_EQ(dr1.status, LinkStatus::Ok);

  auto cr = mgr.Create(LinkCreateParams{"https://d.com"});
  ASSERT_EQ(cr.status, LinkStatus::Ok);
  const std::string code = *cr.code;

  // Delete existing
  auto dr2 = mgr.Delete(code);
  EXPECT_EQ(dr2.status, LinkStatus::Ok);
  // Delete again (still Ok)
  auto dr3 = mgr.Delete(code);
  EXPECT_EQ(dr3.status, LinkStatus::Ok);

  // Resolve after delete -> NotFound
  auto rr = mgr.Resolve(code, true);
  EXPECT_EQ(rr.status, LinkStatus::NotFound);
}

TEST_F(LinkManagerFsTest, CreateRejectsInvalidUrl) {
  LinkManager mgr(tmp_root.string());
  auto cr = mgr.Create(LinkCreateParams{"ftp://nope"});
  EXPECT_EQ(cr.status, LinkStatus::Invalid);
}

TEST_F(LinkManagerFsTest, GetMissingReturnsNotFound) {
  LinkManager mgr(tmp_root.string());
  auto gr = mgr.Get("abc");
  EXPECT_EQ(gr.status, LinkStatus::NotFound);
}

TEST_F(LinkManagerFsTest, ResolveInvalidCodeReturnsInvalid) {
  LinkManager mgr(tmp_root.string());
  auto rr = mgr.Resolve("bad!", true); // '!' not allowed by IsValidCode
  EXPECT_EQ(rr.status, LinkStatus::Invalid);
}

TEST_F(LinkManagerFsTest, MultipleCreatesYieldUniqueCodes) {
  LinkManager mgr(tmp_root.string());
  auto c1 = mgr.Create(LinkCreateParams{"https://a.com"});
  auto c2 = mgr.Create(LinkCreateParams{"https://b.com"});
  auto c3 = mgr.Create(LinkCreateParams{"https://c.com"});
  ASSERT_EQ(c1.status, LinkStatus::Ok);
  ASSERT_EQ(c2.status, LinkStatus::Ok);
  ASSERT_EQ(c3.status, LinkStatus::Ok);
  ASSERT_TRUE(c1.code && c2.code && c3.code);
  EXPECT_NE(*c1.code, *c2.code);
  EXPECT_NE(*c1.code, *c3.code);
  EXPECT_NE(*c2.code, *c3.code);
}

TEST_F(LinkManagerFsTest, InvalidCodeFormatReturnsInvalid) {
  LinkManager mgr(tmp_root.string());
  auto gr = mgr.Get("bad!");
  EXPECT_EQ(gr.status, LinkStatus::Invalid);
  auto ur = mgr.Update("bad!", LinkUpdateParams{"https://x.com"});
  EXPECT_EQ(ur.status, LinkStatus::Invalid);
  auto dr = mgr.Delete("bad!");
  EXPECT_EQ(dr.status, LinkStatus::Invalid);
  auto rr = mgr.Resolve("bad!", true);
  EXPECT_EQ(rr.status, LinkStatus::Invalid);
}

TEST_F(LinkManagerFsTest, CreateRejectsLongUrlAndSpaces) {
  LinkManager mgr(tmp_root.string());
  std::string long_url(2050, 'a');
  long_url = "https://" + long_url;
  auto cr_long = mgr.Create(LinkCreateParams{long_url});
  EXPECT_EQ(cr_long.status, LinkStatus::Invalid);

  auto cr_space = mgr.Create(LinkCreateParams{"https://has space.com"});
  EXPECT_EQ(cr_space.status, LinkStatus::Invalid);
}

TEST_F(LinkManagerFsTest, IncrementVisitsAggregatesByUrl) {
  LinkManager mgr(tmp_root.string());
  auto a = mgr.Create(LinkCreateParams{"https://example.org"});
  auto b = mgr.Create(LinkCreateParams{"https://example.org"});
  ASSERT_TRUE(a.code && b.code);
  EXPECT_TRUE(mgr.IncrementVisits(*a.code));
  EXPECT_TRUE(mgr.IncrementVisits(*b.code));
  EXPECT_TRUE(mgr.IncrementVisits(*b.code));

  std::uint64_t count = 0;
  EXPECT_TRUE(mgr.GetUrlVisitCount("https://example.org", &count));
  EXPECT_EQ(count, 3u);

  std::vector<std::pair<std::string, std::uint64_t>> stats;
  EXPECT_TRUE(mgr.GetAllUrlVisits(&stats));
  ASSERT_EQ(stats.size(), 1u);
  EXPECT_EQ(stats[0].first, "https://example.org");
  EXPECT_EQ(stats[0].second, 3u);
}

TEST_F(LinkManagerFsTest, IncrementCodeVisitsPersistedInRecord) {
  std::string code;
  {
    LinkManager mgr(tmp_root.string());
    auto cr = mgr.Create(LinkCreateParams{"https://example.org"});
    ASSERT_EQ(cr.status, LinkStatus::Ok);
    ASSERT_TRUE(cr.code);
    code = *cr.code;
    EXPECT_TRUE(mgr.IncrementCodeVisits(code));
    EXPECT_TRUE(mgr.IncrementCodeVisits(code));
  }

  LinkManager mgr_again(tmp_root.string());
  auto gr = mgr_again.Get(code);
  ASSERT_EQ(gr.status, LinkStatus::Ok);
  ASSERT_TRUE(gr.record.has_value());
  EXPECT_EQ(gr.record->visits, 2u);
}
