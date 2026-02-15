#include "crud_manager.h"
#include "logger.h"

#include <boost/filesystem.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace fs = boost::filesystem;

CrudManager::CrudManager(const std::string& data_path) 
    : data_path_(fs::absolute(data_path).string()) {
}

std::optional<int> CrudManager::Create(const std::string& entity_type, const std::string& data) {
    std::lock_guard<std::mutex> lock(mutex_);

    fs::path entity_path = fs::path(data_path_) / entity_type;
    boost::system::error_code ec;

    // Ensure the directory for the entity type exists.
    fs::create_directories(entity_path, ec);
    if (ec) {
        Logger::getInstance().logError("CrudManager::Create: Failed to create directory " + entity_path.string() + ": " + ec.message());
        return std::nullopt;
    }

    int next_id = 0;
    if (fs::exists(entity_path) && fs::is_directory(entity_path)) {
        for (const auto& entry : fs::directory_iterator(entity_path)) {
            try {
                int id = std::stoi(entry.path().filename().string());
                if (id > next_id) {
                    next_id = id;
                }
            } catch (const std::invalid_argument&) {
                // Ignore files that are not valid integer IDs
            } catch (const std::out_of_range&) {
                // Ignore files with IDs that are too large
            }
        }
    }
    next_id++;

    fs::path file_path = entity_path / std::to_string(next_id);
    std::ofstream ofs(file_path.string(), std::ios::out | std::ios::trunc | std::ios::binary);
    if (!ofs) {
        Logger::getInstance().logError("CrudManager::Create: Failed to open file for writing " + file_path.string());
        return std::nullopt;
    }

    ofs << data;
    return next_id;
}

std::optional<std::string> CrudManager::Read(const std::string& entity_type, int id) {
    std::lock_guard<std::mutex> lock(mutex_);

    fs::path file_path = fs::path(data_path_) / entity_type / std::to_string(id);

    if (!fs::exists(file_path) || !fs::is_regular_file(file_path)) {
        return std::nullopt;
    }

    std::ifstream ifs(file_path.string(), std::ios::in | std::ios::binary);
    if (!ifs) {
        Logger::getInstance().logError("CrudManager::Read: Failed to open file " + file_path.string());
        return std::nullopt;
    }

    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

bool CrudManager::Update(const std::string& entity_type, int id, const std::string& data) {
    std::lock_guard<std::mutex> lock(mutex_);

    fs::path entity_path = fs::path(data_path_) / entity_type;
    boost::system::error_code ec;

    fs::create_directories(entity_path, ec);
    if (ec) {
        Logger::getInstance().logError("CrudManager::Update: Failed to create directory " + entity_path.string() + ": " + ec.message());
        return false;
    }

    fs::path file_path = entity_path / std::to_string(id);
    std::ofstream ofs(file_path.string(), std::ios::out | std::ios::trunc | std::ios::binary);
    if (!ofs) {
        Logger::getInstance().logError("CrudManager::Update: Failed to open file for writing " + file_path.string());
        return false;
    }

    ofs << data;
    return true;
}

bool CrudManager::Delete(const std::string& entity_type, int id) {
    std::lock_guard<std::mutex> lock(mutex_);

    fs::path file_path = fs::path(data_path_) / entity_type / std::to_string(id);

    if (!fs::exists(file_path)) {
        Logger::getInstance().logWarning("CrudManager::Delete: File does not exist " + file_path.string());
        return true; // File does not exist
    } 

    boost::system::error_code ec;
    if (!fs::remove(file_path, ec)) {
        // `remove` returns false if the file doesn't exist or on error.
        // We already checked for existence, so this is a real error.
        if (ec) {
            Logger::getInstance().logError("CrudManager::Delete: Failed to delete file " + file_path.string() + ": " + ec.message());
        }
        return false;
    }

    return true;
}

std::vector<int> CrudManager::List(const std::string& entity_type) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<int> ids;
    fs::path entity_path = fs::path(data_path_) / entity_type;

    if (!fs::exists(entity_path) || !fs::is_directory(entity_path)) {
        return ids; // Return empty list if directory doesn't exist.
    }

    for (const auto& entry : fs::directory_iterator(entity_path)) {
        if (fs::is_regular_file(entry.status())) {
            try {
                int id = std::stoi(entry.path().filename().string());
                ids.push_back(id);
            } catch (const std::invalid_argument&) {
                // Ignore non-integer filenames
            } catch (const std::out_of_range&) {
                // Ignore filenames that are too large to be an int
            }
        }
    }

    return ids;
}
