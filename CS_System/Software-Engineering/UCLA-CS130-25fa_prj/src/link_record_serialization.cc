#include "link_record_serialization.h"

#include <boost/json.hpp>

namespace json = boost::json;

std::string LinkRecordToJson(const LinkRecord& rec) {
  json::object obj;
  obj["url"] = rec.url;
  obj["visits"] = rec.visits;
  if (rec.password_hash) obj["password_hash"] = *rec.password_hash;
  if (rec.password_salt) obj["password_salt"] = *rec.password_salt;
  return json::serialize(obj);
}

std::string LinkRecordToJsonWithCode(const LinkRecord& rec) {
  json::object obj;
  obj["code"] = rec.code;
  obj["url"] = rec.url;
  obj["visits"] = rec.visits;
  obj["password_protected"] = rec.password_hash.has_value() && rec.password_salt.has_value();
  return json::serialize(obj);
}

std::optional<LinkRecord> LinkRecordFromJson(const std::string& s,
                                             const std::string& code) {
  json::error_code ec;
  json::value v = json::parse(s, ec);
  if (ec) return std::nullopt;
  if (!v.is_object()) return std::nullopt;
  const json::object& obj = v.as_object();
  auto url_it = obj.if_contains("url");
  if (!url_it || !url_it->is_string()) return std::nullopt;
  LinkRecord rec;
  rec.code = code;
  rec.url = std::string(url_it->as_string().c_str());
  rec.visits = 0;
  if (auto* visits_it = obj.if_contains("visits")) {
    if (visits_it->is_uint64()) {
      rec.visits = visits_it->as_uint64();
    } else if (visits_it->is_int64()) {
      auto v = visits_it->as_int64();
      rec.visits = v < 0 ? 0 : static_cast<std::uint64_t>(v);
    }
  }
  if (auto* hash_it = obj.if_contains("password_hash")) {
    if (hash_it->is_string()) {
      rec.password_hash = std::string(hash_it->as_string().c_str());
    }
  }
  if (auto* salt_it = obj.if_contains("password_salt")) {
    if (salt_it->is_string()) {
      rec.password_salt = std::string(salt_it->as_string().c_str());
    }
  }
  return rec;
}
