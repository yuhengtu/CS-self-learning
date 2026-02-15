#ifndef LINK_MANAGER_FAKE_H
#define LINK_MANAGER_FAKE_H

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "link_manager_interface.h"
#include "base62.h"

class LinkManagerFake : public LinkManagerInterface {
public:
  LinkManagerFake();

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
  std::mutex mu_;
  std::uint64_t counter_ = 0; // monotonic counter
  std::unordered_map<std::string, LinkRecord> by_code_;
  std::unordered_map<std::string, std::uint64_t> url_counts_;
};

#endif // LINK_MANAGER_FAKE_H
