#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "fakes/link_manager_fake.h"

TEST(LinkManagerFakeTest, CreateAndResolve) {
  LinkManagerFake mgr;
  LinkCreateParams p{"https://example.com"};
  auto cr = mgr.Create(p);
  ASSERT_EQ(cr.status, LinkStatus::Ok);
  ASSERT_TRUE(cr.code.has_value());

  auto rr = mgr.Resolve(*cr.code, /*increment=*/true);
  ASSERT_EQ(rr.status, LinkStatus::Ok);
  ASSERT_TRUE(rr.url.has_value());
  EXPECT_EQ(*rr.url, p.url);
}

TEST(LinkManagerFakeTest, InvalidCreate) {
  LinkManagerFake mgr;
  LinkCreateParams p{"ftp://not-allowed"};
  auto cr = mgr.Create(p);
  EXPECT_EQ(cr.status, LinkStatus::Invalid);
  EXPECT_FALSE(cr.code.has_value());
}

TEST(LinkManagerFakeTest, GetUpdateDeleteIdempotent) {
  LinkManagerFake mgr;
  // Delete on missing is Ok (idempotent)
  auto del_missing = mgr.Delete("nope");
  EXPECT_EQ(del_missing.status, LinkStatus::Ok);

  // Create
  auto cr = mgr.Create(LinkCreateParams{"https://one.com"});
  ASSERT_EQ(cr.status, LinkStatus::Ok);
  const std::string code = *cr.code;

  // Get
  auto gr = mgr.Get(code);
  ASSERT_EQ(gr.status, LinkStatus::Ok);
  ASSERT_TRUE(gr.record.has_value());
  EXPECT_EQ(gr.record->url, "https://one.com");

  // Update
  auto ur = mgr.Update(code, LinkUpdateParams{"https://two.com"});
  EXPECT_EQ(ur.status, LinkStatus::Ok);
  gr = mgr.Get(code);
  ASSERT_EQ(gr.status, LinkStatus::Ok);
  ASSERT_TRUE(gr.record.has_value());
  EXPECT_EQ(gr.record->url, "https://two.com");

  // Delete existing
  auto dr = mgr.Delete(code);
  EXPECT_EQ(dr.status, LinkStatus::Ok);

  // Delete again (idempotent)
  auto dr2 = mgr.Delete(code);
  EXPECT_EQ(dr2.status, LinkStatus::Ok);

  // Resolve after delete -> NotFound
  auto rr = mgr.Resolve(code, true);
  EXPECT_EQ(rr.status, LinkStatus::NotFound);
}

TEST(LinkManagerFakeTest, GetMissingReturnsNotFound) {
  LinkManagerFake mgr;
  auto gr = mgr.Get("abc");
  EXPECT_EQ(gr.status, LinkStatus::NotFound);
}

TEST(LinkManagerFakeTest, UpdateMissingReturnsNotFound) {
  LinkManagerFake mgr;
  auto ur = mgr.Update("abc", LinkUpdateParams{"https://x.com"});
  EXPECT_EQ(ur.status, LinkStatus::NotFound);
}

TEST(LinkManagerFakeTest, UpdateRejectsInvalidUrl) {
  LinkManagerFake mgr;
  auto cr = mgr.Create(LinkCreateParams{"https://ok.com"});
  ASSERT_EQ(cr.status, LinkStatus::Ok);
  auto ur = mgr.Update(*cr.code, LinkUpdateParams{"mailto:bad"});
  EXPECT_EQ(ur.status, LinkStatus::Invalid);
}

TEST(LinkManagerFakeTest, DeleteMissingIsOk) {
  LinkManagerFake mgr;
  auto dr = mgr.Delete("nope");
  EXPECT_EQ(dr.status, LinkStatus::Ok);
}

TEST(LinkManagerFakeTest, MultipleCreatesYieldUniqueCodes) {
  LinkManagerFake mgr;
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

TEST(LinkManagerFakeTest, InvalidCodeFormatReturnsInvalid) {
  LinkManagerFake mgr;
  auto gr = mgr.Get("bad!");
  EXPECT_EQ(gr.status, LinkStatus::Invalid);
  auto ur = mgr.Update("bad!", LinkUpdateParams{"https://x.com"});
  EXPECT_EQ(ur.status, LinkStatus::Invalid);
  auto dr = mgr.Delete("bad!");
  EXPECT_EQ(dr.status, LinkStatus::Invalid);
  auto rr = mgr.Resolve("bad!", true);
  EXPECT_EQ(rr.status, LinkStatus::Invalid);
}

TEST(LinkManagerFakeTest, CreateRejectsLongUrlAndSpaces) {
  LinkManagerFake mgr;
  std::string long_url(2050, 'a');
  long_url = "https://" + long_url;
  auto cr_long = mgr.Create(LinkCreateParams{long_url});
  EXPECT_EQ(cr_long.status, LinkStatus::Invalid);

  auto cr_space = mgr.Create(LinkCreateParams{"https://has space.com"});
  EXPECT_EQ(cr_space.status, LinkStatus::Invalid);
}

TEST(LinkManagerFakeTest, IncrementVisitsTracksUrlCounts) {
  LinkManagerFake mgr;
  auto first = mgr.Create(LinkCreateParams{"https://x.com"});
  auto second = mgr.Create(LinkCreateParams{"https://x.com"});
  ASSERT_EQ(first.status, LinkStatus::Ok);
  ASSERT_EQ(second.status, LinkStatus::Ok);
  ASSERT_TRUE(first.code && second.code);

  EXPECT_TRUE(mgr.IncrementVisits(*first.code));
  EXPECT_TRUE(mgr.IncrementVisits(*second.code));
  EXPECT_TRUE(mgr.IncrementVisits(*second.code));

  std::uint64_t count = 0;
  EXPECT_TRUE(mgr.GetUrlVisitCount("https://x.com", &count));
  EXPECT_EQ(count, 3u);

  std::vector<std::pair<std::string, std::uint64_t>> stats;
  EXPECT_TRUE(mgr.GetAllUrlVisits(&stats));
  ASSERT_EQ(stats.size(), 1u);
  EXPECT_EQ(stats[0].first, "https://x.com");
  EXPECT_EQ(stats[0].second, 3u);
}

TEST(LinkManagerFakeTest, IncrementCodeVisitsUpdatesRecord) {
  LinkManagerFake mgr;
  auto created = mgr.Create(LinkCreateParams{"https://example.com"});
  ASSERT_EQ(created.status, LinkStatus::Ok);
  ASSERT_TRUE(created.code);

  EXPECT_TRUE(mgr.IncrementCodeVisits(*created.code));
  EXPECT_TRUE(mgr.IncrementCodeVisits(*created.code));

  auto gr = mgr.Get(*created.code);
  ASSERT_EQ(gr.status, LinkStatus::Ok);
  ASSERT_TRUE(gr.record);
  EXPECT_EQ(gr.record->visits, 2u);
}

TEST(LinkManagerFakeTest, CreateStoresPasswordFields) {
  LinkManagerFake mgr;
  LinkCreateParams params{"https://example.com"};
  params.password_hash = "hash";
  params.password_salt = "salt";
  auto created = mgr.Create(params);
  ASSERT_EQ(created.status, LinkStatus::Ok);
  ASSERT_TRUE(created.code);
  auto gr = mgr.Get(*created.code);
  ASSERT_EQ(gr.status, LinkStatus::Ok);
  ASSERT_TRUE(gr.record);
  EXPECT_TRUE(gr.record->password_hash.has_value());
  EXPECT_TRUE(gr.record->password_salt.has_value());
  EXPECT_EQ(*gr.record->password_hash, "hash");
  EXPECT_EQ(*gr.record->password_salt, "salt");
}
