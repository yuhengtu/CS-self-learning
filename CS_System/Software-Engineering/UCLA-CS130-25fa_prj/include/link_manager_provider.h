#ifndef LINK_MANAGER_PROVIDER_H
#define LINK_MANAGER_PROVIDER_H

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "link_manager_interface.h"
#include "link_manager.h"

class LinkManagerProvider {
public:
  static std::shared_ptr<LinkManagerInterface> GetOrCreate(const std::string& data_path);

private:
  static std::mutex mu_;
  static std::unordered_map<std::string, std::weak_ptr<LinkManagerInterface>> by_path_;
};

#endif // LINK_MANAGER_PROVIDER_H

