#include "link_manager.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <boost/json.hpp>
#ifdef __unix__
#include <unistd.h>
#endif
#include "base62.h"
#include "link_record_serialization.h"

namespace fs = boost::filesystem;

namespace {
constexpr std::uint64_t kCounterSeed = 15000000;
}

LinkManager::LinkManager(std::string data_path)
  : data_root_(fs::absolute(data_path)) {}

CreateResult LinkManager::Create(const LinkCreateParams& params) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!IsValidUrl(params.url)) {
    return {LinkStatus::Invalid, std::nullopt};
  }
  if (!EnsureDir(UrlsDir())) {
    return {LinkStatus::FsError, std::nullopt};
  }

  std::uint64_t counter = 0;
  if (!ReadCounter(&counter)) {
    return {LinkStatus::FsError, std::nullopt};
  }
  ++counter;
  if (!WriteCounter(counter)) {
    return {LinkStatus::FsError, std::nullopt};
  }
  const std::string code = base62::Encode(counter);
  const fs::path dst = CodePath(code);
  LinkRecord rec;
  rec.code = code;
  rec.url = params.url;
  rec.visits = 0;
  rec.password_hash = params.password_hash;
  rec.password_salt = params.password_salt;
  const std::string body = LinkRecordToJson(rec);
  if (!AtomicWrite(dst, body)) {
    return {LinkStatus::FsError, std::nullopt};
  }
  return {LinkStatus::Ok, code};
}

GetResult LinkManager::Get(const std::string& code) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!IsValidCode(code)) return {LinkStatus::Invalid, std::nullopt};
  const fs::path p = CodePath(code);
  if (!fs::exists(p) || !fs::is_regular_file(p)) {
    return {LinkStatus::NotFound, std::nullopt};
  }
  std::string body;
  if (!ReadFile(p, &body)) {
    return {LinkStatus::FsError, std::nullopt};
  }
  auto rec = LinkRecordFromJson(body, code);
  if (!rec) return {LinkStatus::FsError, std::nullopt};
  return {LinkStatus::Ok, *rec};
}

UpdateResult LinkManager::Update(const std::string& code, const LinkUpdateParams& params) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!IsValidCode(code)) return {LinkStatus::Invalid};
  if (!IsValidUrl(params.url)) return {LinkStatus::Invalid};
  const fs::path p = CodePath(code);
  if (!fs::exists(p) || !fs::is_regular_file(p)) {
    return {LinkStatus::NotFound};
  }
  // Read, update DTO, and write JSON
  std::string body;
  if (!ReadFile(p, &body)) return {LinkStatus::FsError};
  auto rec = LinkRecordFromJson(body, code);
  if (!rec) return {LinkStatus::FsError};
  rec->url = params.url;
  const std::string out = LinkRecordToJson(*rec);
  if (!AtomicWrite(p, out)) {
    return {LinkStatus::FsError};
  }
  return {LinkStatus::Ok};
}

DeleteResult LinkManager::Delete(const std::string& code) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!IsValidCode(code)) return {LinkStatus::Invalid};
  const fs::path p = CodePath(code);
  if (!fs::exists(p)) {
    return {LinkStatus::Ok}; // idempotent delete
  }
  boost::system::error_code ec;
  fs::remove(p, ec);
  if (ec) return {LinkStatus::FsError};
  return {LinkStatus::Ok};
}

ResolveResult LinkManager::Resolve(const std::string& code, bool /*increment*/) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!IsValidCode(code)) return {LinkStatus::Invalid, std::nullopt};
  const fs::path p = CodePath(code);
  if (!fs::exists(p) || !fs::is_regular_file(p)) {
    return {LinkStatus::NotFound, std::nullopt};
  }
  std::string body;
  if (!ReadFile(p, &body)) {
    return {LinkStatus::FsError, std::nullopt};
  }
  auto rec = LinkRecordFromJson(body, code);
  if (!rec) return {LinkStatus::FsError, std::nullopt};
  return {LinkStatus::Ok, rec->url};
}

bool LinkManager::IncrementCodeVisits(const std::string& code) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!IsValidCode(code)) return false;
  const fs::path p = CodePath(code);
  if (!fs::exists(p) || !fs::is_regular_file(p)) {
    return false;
  }
  std::string body;
  if (!ReadFile(p, &body)) return false;
  auto rec = LinkRecordFromJson(body, code);
  if (!rec) return false;
  ++rec->visits;
  return AtomicWrite(p, LinkRecordToJson(*rec));
}

bool LinkManager::IncrementVisits(const std::string& code) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!IsValidCode(code)) return false;
  const fs::path p = CodePath(code);
  if (!fs::exists(p) || !fs::is_regular_file(p)) {
    return false;
  }
  std::string body;
  if (!ReadFile(p, &body)) return false;
  auto rec = LinkRecordFromJson(body, code);
  if (!rec) return false;
  std::unordered_map<std::string, std::uint64_t> stats;
  if (!ReadUrlStats(&stats)) return false;
  ++stats[rec->url];
  return WriteUrlStats(stats);
}

bool LinkManager::GetUrlVisitCount(const std::string& url, std::uint64_t* count) {
  if (!count) return false;
  std::lock_guard<std::mutex> lk(mu_);
  std::unordered_map<std::string, std::uint64_t> stats;
  if (!ReadUrlStats(&stats)) return false;
  auto it = stats.find(url);
  *count = (it == stats.end()) ? 0 : it->second;
  return true;
}

bool LinkManager::GetAllUrlVisits(
    std::vector<std::pair<std::string, std::uint64_t>>* stats) {
  if (!stats) return false;
  std::lock_guard<std::mutex> lk(mu_);
  stats->clear();
  std::unordered_map<std::string, std::uint64_t> map;
  if (!ReadUrlStats(&map)) return false;
  stats->reserve(map.size());
  for (const auto& entry : map) {
    stats->emplace_back(entry.first, entry.second);
  }
  return true;
}

fs::path LinkManager::UrlsDir() const {
  return data_root_ / "urls";
}

fs::path LinkManager::CodePath(const std::string& code) const {
  return UrlsDir() / code; // extensionless; content is URL text
}

fs::path LinkManager::CounterPath() const {
  return UrlsDir() / ".counter";
}

bool LinkManager::EnsureDir(const fs::path& p) {
  boost::system::error_code ec;
  fs::create_directories(p, ec);
  return !ec;
}

bool LinkManager::AtomicWrite(const fs::path& path, const std::string& data) {
  // temp file in same directory
  fs::path tmp = path;
  tmp += ".tmp";
  tmp += "." + std::to_string(::getpid());
  std::ofstream ofs(tmp.string(), std::ios::out | std::ios::trunc | std::ios::binary);
  if (!ofs) return false;
  ofs << data;
  ofs.flush();
  ofs.close();
  boost::system::error_code ec;
  fs::rename(tmp, path, ec);
  return !ec;
}

bool LinkManager::ReadFile(const fs::path& path, std::string* out) {
  std::ifstream ifs(path.string(), std::ios::in | std::ios::binary);
  if (!ifs) return false;
  std::ostringstream ss;
  ss << ifs.rdbuf();
  *out = ss.str();
  return true;
}

bool LinkManager::ReadCounter(std::uint64_t* out) {
  const fs::path cp = CounterPath();
  if (!fs::exists(cp)) { *out = kCounterSeed; return true; }
  std::string s;
  if (!ReadFile(cp, &s)) return false;
  try {
    *out = static_cast<std::uint64_t>(std::stoull(s));
  } catch (...) {
    return false;
  }
  return true;
}

bool LinkManager::WriteCounter(std::uint64_t value) {
  const fs::path cp = CounterPath();
  return AtomicWrite(cp, std::to_string(value));
}

bool LinkManager::ReadUrlStats(std::unordered_map<std::string, std::uint64_t>* stats) {
  stats->clear();
  const fs::path path = UrlStatsPath();
  if (!fs::exists(path)) return true;
  std::string body;
  if (!ReadFile(path, &body)) return false;
  boost::json::error_code ec;
  boost::json::value val = boost::json::parse(body, ec);
  if (ec || !val.is_object()) return false;
  for (const auto& kv : val.as_object()) {
    const auto& value = kv.value();
    std::uint64_t visits = 0;
    if (value.is_uint64()) {
      visits = value.as_uint64();
    } else if (value.is_int64()) {
      auto v = value.as_int64();
      visits = v < 0 ? 0 : static_cast<std::uint64_t>(v);
    } else {
      continue;
    }
    (*stats)[kv.key_c_str()] = visits;
  }
  return true;
}

bool LinkManager::WriteUrlStats(const std::unordered_map<std::string, std::uint64_t>& stats) {
  boost::json::object obj;
  for (const auto& entry : stats) {
    obj[entry.first] = entry.second;
  }
  return AtomicWrite(UrlStatsPath(), boost::json::serialize(obj));
}

fs::path LinkManager::UrlStatsPath() const {
  return data_root_ / "url_stats.json";
}
