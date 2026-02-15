#ifndef LINK_MANAGER_H
#define LINK_MANAGER_H

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/filesystem.hpp>

#include "link_manager_interface.h"

// Filesystem-backed implementation of LinkManagerInterface.
class LinkManager : public LinkManagerInterface {
public:
  explicit LinkManager(std::string data_path);

  CreateResult Create(const LinkCreateParams& params) override;
  GetResult    Get(const std::string& code) override;
  UpdateResult Update(const std::string& code, const LinkUpdateParams& params) override;
  DeleteResult Delete(const std::string& code) override;
  ResolveResult Resolve(const std::string& code, bool increment) override;
  bool IncrementCodeVisits(const std::string& code) override;
  bool IncrementVisits(const std::string& code) override;
  bool GetUrlVisitCount(const std::string& url, std::uint64_t* count) override;
  bool GetAllUrlVisits(std::vector<std::pair<std::string, std::uint64_t>>* stats) override;

private:
  // Paths
  boost::filesystem::path UrlsDir() const;
  boost::filesystem::path CodePath(const std::string& code) const;
  boost::filesystem::path CounterPath() const;

  // IO helpers (return false on error)
  bool EnsureDir(const boost::filesystem::path& p);
  bool AtomicWrite(const boost::filesystem::path& path, const std::string& data);
  bool ReadFile(const boost::filesystem::path& path, std::string* out);

  bool ReadCounter(std::uint64_t* out);
  bool WriteCounter(std::uint64_t value);

  bool ReadUrlStats(std::unordered_map<std::string, std::uint64_t>* stats);
  bool WriteUrlStats(const std::unordered_map<std::string, std::uint64_t>& stats);
  boost::filesystem::path UrlStatsPath() const;

  // state
  const boost::filesystem::path data_root_;
  mutable std::mutex mu_;
};

#endif // LINK_MANAGER_H
