#include "static_request_handler.h"
#include "logger.h"
#include "response_builder.h"

#include <fstream>
#include <sstream>
#include <string>
#include <optional>
#include <memory>
#include <boost/filesystem.hpp>

namespace {
std::optional<std::string> join_fs_path(const std::string& root, const std::string& rel) {
    Logger& log = Logger::getInstance();

    if (root.empty()) return rel;
    if (rel.empty())  return root;
    if (root.back() == '/' && rel.front() == '/') return root + rel.substr(1);
    if (root.back() != '/' && rel.front() != '/') return root + "/" + rel;
    
    std::string joined = root + rel;

    try {
        namespace fs = boost::filesystem;
        fs::path root_path = fs::weakly_canonical(fs::path(root));
        fs::path full_path = fs::weakly_canonical(fs::path(joined));
        if (full_path.string().rfind(root_path.string(), 0) != 0) {
            log.logWarning("Path traversal attempt detected, requested: " + joined);
            return std::nullopt;
        }
        return full_path.string();
    } catch (const std::exception& e) {
        log.logError(e.what());
        throw;
    }
}
std::string get_extension(const std::string& path) {
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return "";
    return path.substr(pos + 1);
}
}

StaticRequestHandler::StaticRequestHandler(std::string mount_prefix, std::string root)
    : mount_prefix_(std::move(mount_prefix)), root_(std::move(root)) {}

std::string StaticRequestHandler::content_type_for_extension(const std::string& path) {
    const std::string ext = get_extension(path);
    if (ext == "html") return "text/html";
    if (ext == "css") return "text/css";
    if (ext == "js") return "text/javascript";
    if (ext == "jpg")  return "image/jpeg";
    if (ext == "zip")  return "application/zip";
    if (ext == "txt")  return "text/plain";
    return "";
}

std::unique_ptr<response> StaticRequestHandler::handle_request(const request& req) {
    Logger& log = Logger::getInstance();
    log.logTrace("static_request_handler: handling " + req.uri);

    // 1. Handler allocates the response
    auto out = std::make_unique<response>();

    if (req.uri.rfind(mount_prefix_, 0) != 0) {
        log.logWarning("static_request_handler: URI does not match mount prefix " + mount_prefix_);
        ResponseBuilder::createNotFound().build(*out);
        return out; // 2. Return 404 response
    }

    std::string rel = req.uri.substr(mount_prefix_.size());
    if (rel.empty() || rel == "/") {
        log.logWarning("static_request_handler: empty relative path after prefix");
        ResponseBuilder::createNotFound().build(*out);
        return out; // 2. Return 404 response
    }

    std::optional<std::string> full_path_opt;
    try {
        full_path_opt = join_fs_path(root_, rel);
    } catch (const std::exception& e) {
        log.logError("static_request_handler: filesystem error during path resolution: " + std::string(e.what()));
        ResponseBuilder::createInternalServerError().build(*out);
        return out; // 2. Return 500 response
    }
    
    if (!full_path_opt) {
        ResponseBuilder::createNotFound().build(*out);
        return out; // 2. Return 404 response
    }
    
    std::string full_path = *full_path_opt;
    std::string content_type = content_type_for_extension(full_path);
    if (content_type.empty()) {
        log.logWarning("static_request_handler: unsupported extension for " + full_path);
        ResponseBuilder::createNotFound().build(*out);
        return out; // 2. Return 404 response
    }

    std::ifstream file(full_path, std::ios::in | std::ios::binary);
    if (!file.good()) {
        log.logWarning("static_request_handler: file not found " + full_path);
        ResponseBuilder::createNotFound().build(*out);
        return out; // 2. Return 404 response
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string body = buffer.str();

    // 2. Build 200 OK response
    ResponseBuilder(200)
        .withContentType(content_type)
        .withBody(std::move(body))
        .build(*out);

    log.logTrace("static_request_handler: served " + full_path + " (" + content_type + ")");
    return out; // 3. Return 200 response
}