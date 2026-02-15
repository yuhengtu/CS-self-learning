#ifndef LINK_MANAGER_INTERFACE_H
#define LINK_MANAGER_INTERFACE_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "link_manager_types.h"

// Abstract interface for the URL shortener manager.
// Handlers pass structured params; manager enforces invariants and persistence.
class LinkManagerInterface {
public:
  virtual ~LinkManagerInterface() = default;

  virtual CreateResult Create(const LinkCreateParams& params) = 0;
  virtual GetResult    Get(const std::string& code) = 0;
  virtual UpdateResult Update(const std::string& code,
                              const LinkUpdateParams& params) = 0;
  virtual DeleteResult Delete(const std::string& code) = 0;

  // Resolve returns the long URL. If increment is true, implementation may
  // atomically update internal counters (TODO for future analytics).
  virtual ResolveResult Resolve(const std::string& code,
                                bool increment) = 0;

  virtual bool IncrementCodeVisits(const std::string& code) = 0;
  virtual bool IncrementVisits(const std::string& code) = 0;
  virtual bool GetUrlVisitCount(const std::string& url, std::uint64_t* count) = 0;
  virtual bool GetAllUrlVisits(std::vector<std::pair<std::string, std::uint64_t>>* stats) = 0;

  // Shared validation helpers
  static bool IsValidUrl(const std::string& url) {
    if (url.empty() || url.size() > 2048) return false;
    // heuristic: must start with http:// or https:// and contain no spaces
    const bool scheme = (url.rfind("http://", 0) == 0) || (url.rfind("https://", 0) == 0);
    if (!scheme) return false;
    return url.find(' ') == std::string::npos;
  }
  static bool IsValidCode(const std::string& code) {
    if (code.empty() || code.size() > 32) return false;
    for (char c : code) {
      const bool is_digit = (c >= '0' && c <= '9');
      const bool is_upper = (c >= 'A' && c <= 'Z');
      const bool is_lower = (c >= 'a' && c <= 'z');
      if (!(is_digit || is_upper || is_lower)) return false;
    }
    return true;
  }
};

#endif // LINK_MANAGER_INTERFACE_H
