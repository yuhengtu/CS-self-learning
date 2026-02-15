#include "link_manager_provider.h"

#include <boost/filesystem.hpp>

std::mutex LinkManagerProvider::mu_;
std::unordered_map<std::string, std::weak_ptr<LinkManagerInterface>> LinkManagerProvider::by_path_;

static std::string Normalize(const std::string& p) {
  boost::filesystem::path abs = boost::filesystem::absolute(p);
  return abs.lexically_normal().string();
}

std::shared_ptr<LinkManagerInterface> LinkManagerProvider::GetOrCreate(const std::string& data_path) {
  const std::string key = Normalize(data_path);
  std::lock_guard<std::mutex> lk(mu_);
  auto it = by_path_.find(key);
  if (it != by_path_.end()) {
    if (auto sp = it->second.lock()) return sp;
  }
  auto sp = std::make_shared<LinkManager>(key);
  by_path_[key] = sp;
  return sp;
}

