#include "link_manager_fake.h"

#include <algorithm>
#include "base62.h"

LinkManagerFake::LinkManagerFake() = default;

CreateResult LinkManagerFake::Create(const LinkCreateParams& params) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!LinkManagerInterface::IsValidUrl(params.url)) {
    return {LinkStatus::Invalid, std::nullopt};
  }
  // generate next code
  const std::string code = base62::Encode(++counter_);
  LinkRecord rec;
  rec.code = code;
  rec.url = params.url;
  rec.visits = 0;
  rec.password_hash = params.password_hash;
  rec.password_salt = params.password_salt;
  by_code_[code] = rec;
  url_counts_.emplace(params.url, 0);
  return {LinkStatus::Ok, code};
}

GetResult LinkManagerFake::Get(const std::string& code) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!LinkManagerInterface::IsValidCode(code)) {
    return {LinkStatus::Invalid, std::nullopt};
  }
  auto it = by_code_.find(code);
  if (it == by_code_.end()) return {LinkStatus::NotFound, std::nullopt};
  return {LinkStatus::Ok, it->second};
}

UpdateResult LinkManagerFake::Update(const std::string& code, const LinkUpdateParams& params) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!LinkManagerInterface::IsValidCode(code)) {
    return {LinkStatus::Invalid};
  }
  if (!LinkManagerInterface::IsValidUrl(params.url)) {
    return {LinkStatus::Invalid};
  }
  auto it = by_code_.find(code);
  if (it == by_code_.end()) return {LinkStatus::NotFound};
  it->second.url = params.url;
  return {LinkStatus::Ok};
}

DeleteResult LinkManagerFake::Delete(const std::string& code) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!LinkManagerInterface::IsValidCode(code)) {
    return {LinkStatus::Invalid};
  }
  auto it = by_code_.find(code);
  if (it == by_code_.end()) return {LinkStatus::Ok};
  by_code_.erase(it);
  return {LinkStatus::Ok};
}

ResolveResult LinkManagerFake::Resolve(const std::string& code, bool /*increment*/) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!LinkManagerInterface::IsValidCode(code)) {
    return {LinkStatus::Invalid, std::nullopt};
  }
  auto it = by_code_.find(code);
  if (it == by_code_.end()) return {LinkStatus::NotFound, std::nullopt};
  return {LinkStatus::Ok, it->second.url};
}

bool LinkManagerFake::IncrementCodeVisits(const std::string& code) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!LinkManagerInterface::IsValidCode(code)) return false;
  auto it = by_code_.find(code);
  if (it == by_code_.end()) return false;
  ++it->second.visits;
  return true;
}

bool LinkManagerFake::IncrementVisits(const std::string& code) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!LinkManagerInterface::IsValidCode(code)) return false;
  auto it = by_code_.find(code);
  if (it == by_code_.end()) return false;
  ++url_counts_[it->second.url];
  return true;
}

bool LinkManagerFake::GetUrlVisitCount(const std::string& url, std::uint64_t* count) {
  if (!count) return false;
  std::lock_guard<std::mutex> lk(mu_);
  auto it = url_counts_.find(url);
  *count = (it == url_counts_.end()) ? 0 : it->second;
  return true;
}

bool LinkManagerFake::GetAllUrlVisits(
    std::vector<std::pair<std::string, std::uint64_t>>* stats) {
  if (!stats) return false;
  std::lock_guard<std::mutex> lk(mu_);
  stats->clear();
  for (const auto& entry : url_counts_) {
    stats->emplace_back(entry.first, entry.second);
  }
  return true;
}
