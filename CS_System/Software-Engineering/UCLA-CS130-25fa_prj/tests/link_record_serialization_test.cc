#include <gtest/gtest.h>
#include <boost/json.hpp>
#include <string>

#include "link_record_serialization.h"

namespace json = boost::json;

TEST(LinkRecordSerializationTest, RoundTripOk) {
  LinkRecord rec{"abc", "https://example.com", 42};
  std::string serialized = LinkRecordToJson(rec);
  auto parsed = LinkRecordFromJson(serialized, rec.code);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->code, rec.code);
  EXPECT_EQ(parsed->url, rec.url);
  EXPECT_EQ(parsed->visits, rec.visits);
  EXPECT_FALSE(parsed->password_hash.has_value());
  EXPECT_FALSE(parsed->password_salt.has_value());
}

TEST(LinkRecordSerializationTest, MissingUrlFieldIsError) {
  const std::string bad = "{\"not_url\":\"x\"}";
  auto parsed = LinkRecordFromJson(bad, "abc");
  EXPECT_FALSE(parsed.has_value());
}

TEST(LinkRecordSerializationTest, MissingVisitsDefaultsToZero) {
  const std::string body = "{\"url\":\"https://example.com\"}";
  auto parsed = LinkRecordFromJson(body, "abc");
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->visits, 0u);
}

TEST(LinkRecordSerializationTest, MalformedJsonIsError) {
  const std::string bad = "{\"url\":\"unterminated...";
  auto parsed = LinkRecordFromJson(bad, "abc");
  EXPECT_FALSE(parsed.has_value());
}

TEST(LinkRecordSerializationTest, SerializeWithCodeIncludesFields) {
  LinkRecord rec{"xyz", "https://example.com", 5};
  std::string serialized = LinkRecordToJsonWithCode(rec);
  json::error_code ec;
  auto value = json::parse(serialized, ec);
  ASSERT_FALSE(ec);
  ASSERT_TRUE(value.is_object());
  const auto& obj = value.as_object();
  EXPECT_STREQ(obj.at("code").as_string().c_str(), "xyz");
  EXPECT_STREQ(obj.at("url").as_string().c_str(), "https://example.com");
  const auto& visits_val = obj.at("visits");
  std::uint64_t visits = 0;
  if (visits_val.is_uint64()) {
    visits = visits_val.as_uint64();
  } else if (visits_val.is_int64()) {
    auto v = visits_val.as_int64();
    visits = v < 0 ? 0 : static_cast<std::uint64_t>(v);
  }
  EXPECT_EQ(visits, 5u);
  EXPECT_FALSE(obj.at("password_protected").as_bool());
}

TEST(LinkRecordSerializationTest, PasswordFieldsPersisted) {
  LinkRecord rec{"abc", "https://example.com", 1};
  rec.password_salt = "salt";
  rec.password_hash = "hash";
  std::string serialized = LinkRecordToJson(rec);
  json::error_code ec;
  auto value = json::parse(serialized, ec);
  ASSERT_FALSE(ec);
  ASSERT_TRUE(value.is_object());
  const auto& obj = value.as_object();
  EXPECT_STREQ(obj.at("password_hash").as_string().c_str(), "hash");
  EXPECT_STREQ(obj.at("password_salt").as_string().c_str(), "salt");
  auto parsed = LinkRecordFromJson(serialized, rec.code);
  ASSERT_TRUE(parsed.has_value());
  ASSERT_TRUE(parsed->password_hash.has_value());
  ASSERT_TRUE(parsed->password_salt.has_value());
  EXPECT_EQ(*parsed->password_hash, "hash");
  EXPECT_EQ(*parsed->password_salt, "salt");
}

TEST(LinkRecordSerializationTest, SerializeWithCodeMarksProtectedRecords) {
  LinkRecord rec{"xyz", "https://example.com", 5};
  rec.password_salt = "salt";
  rec.password_hash = "hash";
  std::string serialized = LinkRecordToJsonWithCode(rec);
  json::error_code ec;
  auto value = json::parse(serialized, ec);
  ASSERT_FALSE(ec);
  ASSERT_TRUE(value.is_object());
  const auto& obj = value.as_object();
  EXPECT_TRUE(obj.at("password_protected").as_bool());
}
