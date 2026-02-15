#include <gtest/gtest.h>
#include <boost/filesystem.hpp>
#ifdef __unix__
#include <unistd.h>
#endif

#include "link_manager_provider.h"

namespace fs = boost::filesystem;

class LinkManagerProviderTest : public ::testing::Test {
protected:
  fs::path tmp;
  void SetUp() override {
    tmp = fs::path("/tmp/link_manager_provider_") / std::to_string(::getpid());
    fs::remove_all(tmp);
    fs::create_directories(tmp);
  }
  void TearDown() override { fs::remove_all(tmp); }
};

TEST_F(LinkManagerProviderTest, SamePathReturnsSameInstance) {
  auto a = LinkManagerProvider::GetOrCreate(tmp.string());
  auto b = LinkManagerProvider::GetOrCreate(tmp.string());
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  EXPECT_EQ(a.get(), b.get());
}

TEST_F(LinkManagerProviderTest, NormalizedPathsShareInstance) {
  std::string p1 = tmp.string();
  std::string p2 = (tmp / "").string(); // trailing slash
  auto a = LinkManagerProvider::GetOrCreate(p1);
  auto b = LinkManagerProvider::GetOrCreate(p2);
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  EXPECT_EQ(a.get(), b.get());
}

TEST_F(LinkManagerProviderTest, DifferentPathsReturnDifferentInstances) {
  auto a = LinkManagerProvider::GetOrCreate(tmp.string());
  fs::path other = tmp.parent_path() / (tmp.filename().string() + "_other");
  fs::create_directories(other);
  auto b = LinkManagerProvider::GetOrCreate(other.string());
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  EXPECT_NE(a.get(), b.get());
}

TEST_F(LinkManagerProviderTest, WeakPtrAllowsRecreationAfterReset) {
  auto a = LinkManagerProvider::GetOrCreate(tmp.string());
  void* addr = a.get();
  // Drop the strong reference
  a.reset();
  // Create again; should be a different instance address
  auto b = LinkManagerProvider::GetOrCreate(tmp.string());
  ASSERT_TRUE(b);
  EXPECT_NE(addr, b.get());
}

