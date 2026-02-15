#ifndef RESPONSE_BUILDER_H
#define RESPONSE_BUILDER_H

#include <string>
#include <map>
#include "response.h"

class ResponseBuilder {
public:
    explicit ResponseBuilder(int status_code);
    ResponseBuilder(int status_code, const std::string& reason_phrase);
    
    static ResponseBuilder createOk();
    static ResponseBuilder createBadRequest();
    static ResponseBuilder createBadRequest(const std::string& message);
    static ResponseBuilder createNotFound();
    static ResponseBuilder createNotFound(const std::string& message);
    static ResponseBuilder createInternalServerError();
    static ResponseBuilder createInternalServerError(const std::string& message);
    
    ResponseBuilder& withBody(const std::string& content);
    // Move semantics for body content 
    ResponseBuilder& withBody(std::string&& content);
    ResponseBuilder& withHeader(const std::string& name, const std::string& value);
    ResponseBuilder& withContentType(const std::string& type);
    ResponseBuilder& withHttpVersion(const std::string& version);
    
    void build(response& out);
    
private:
    int status_code_;
    std::string reason_phrase_;
    std::string body_;
    std::map<std::string, std::string> headers_;
    std::string http_version_ = "1.1"; // default HTTP version
    
    std::string getDefaultReasonPhrase(int code) const;
};

#endif // RESPONSE_BUILDER_H

