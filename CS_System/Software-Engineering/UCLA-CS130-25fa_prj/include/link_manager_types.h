#ifndef LINK_MANAGER_TYPES_H
#define LINK_MANAGER_TYPES_H

#include <cstdint>
#include <optional>
#include <string>

// Minimal DTO returned by the manager for reads
struct LinkRecord {
  std::string code;
  std::string url;
  std::uint64_t visits{0};
  std::optional<std::string> password_hash;
  std::optional<std::string> password_salt;
};

// Inputs parsed by handlers for Create/Update
struct LinkCreateParams {
  std::string url;
  std::optional<std::string> password_hash;
  std::optional<std::string> password_salt;
};

struct LinkUpdateParams {
  std::string url;
};

// Unified status for manager operations
enum class LinkStatus {
  Ok,
  NotFound,
  Invalid,
  FsError // (internal filesystem error)
};

// Result wrappers
struct CreateResult {
  LinkStatus status;
  std::optional<std::string> code;  // set on Ok
};

struct GetResult {
  LinkStatus status;
  std::optional<LinkRecord> record; // set on Ok
};

struct UpdateResult {
  LinkStatus status;
};

struct DeleteResult {
  LinkStatus status;
};

struct ResolveResult {
  LinkStatus status;
  std::optional<std::string> url;   // set on Ok
};

#endif // LINK_MANAGER_TYPES_H
