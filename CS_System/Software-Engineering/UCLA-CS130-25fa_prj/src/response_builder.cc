#include "response_builder.h"
#include "logger.h"

ResponseBuilder::ResponseBuilder(int status_code)
    : status_code_(status_code), reason_phrase_(getDefaultReasonPhrase(status_code)) {}

ResponseBuilder::ResponseBuilder(int status_code, const std::string& reason_phrase)
    : status_code_(status_code), reason_phrase_(reason_phrase) {}

ResponseBuilder ResponseBuilder::createOk() {
    return ResponseBuilder(200);
}

ResponseBuilder ResponseBuilder::createBadRequest() {
    return ResponseBuilder(400);
}

ResponseBuilder ResponseBuilder::createBadRequest(const std::string& message) {
    return ResponseBuilder(400).withBody(message);
}

ResponseBuilder ResponseBuilder::createNotFound() {
    return ResponseBuilder(404);
}

ResponseBuilder ResponseBuilder::createNotFound(const std::string& message) {
    return ResponseBuilder(404).withBody(message);
}

ResponseBuilder ResponseBuilder::createInternalServerError() {
    return ResponseBuilder(500);
}

ResponseBuilder ResponseBuilder::createInternalServerError(const std::string& message) {
    return ResponseBuilder(500).withBody(message);
}

ResponseBuilder& ResponseBuilder::withBody(const std::string& content) {
    body_ = content;
    return *this;
}

ResponseBuilder& ResponseBuilder::withBody(std::string&& content) {
    body_ = std::move(content);
    return *this;
}

ResponseBuilder& ResponseBuilder::withHeader(const std::string& name, const std::string& value) {
    headers_[name] = value;
    return *this;
}

ResponseBuilder& ResponseBuilder::withContentType(const std::string& type) {
    return withHeader("Content-Type", type);
}

ResponseBuilder& ResponseBuilder::withHttpVersion(const std::string& version) {
    http_version_ = version;
    return *this;
}

void ResponseBuilder::build(response& out) {
    Logger& log = Logger::getInstance();

    headers_["Content-Length"] = std::to_string(body_.size());
    
    auto it = headers_.find("Connection");
    if (it != headers_.end() and it->second != "close") {
        log.logWarning("Connection header set to '" + it->second + "' but only 'close' is supported, overriding.");
    }
    headers_["Connection"] = "close"; // Add to every response object
    
    std::string status_line = "HTTP/" + http_version_ + " " +
                              std::to_string(status_code_) + " " +
                              reason_phrase_ + "\r\n";
    
    std::string headers_str;
    for (const auto& [key, value] : headers_) {
        headers_str += key + ": " + value + "\r\n";
    }
    headers_str += "\r\n";
    
    out.set_status_line(status_line);
    out.set_headers(headers_str);
    out.set_content(body_);
    out.set_bufs();
}

std::string ResponseBuilder::getDefaultReasonPhrase(int code) const {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 500: return "Internal Server Error";
        default: return "Unknown";
    }
}

