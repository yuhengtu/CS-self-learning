#include "gtest/gtest.h"
#include "response_builder.h"
#include "response.h"

TEST(ResponseBuilder, MinimalResponseWithStatusOnly) {
    response resp;
    ResponseBuilder(200).build(resp);
    
    EXPECT_EQ(resp.get_status_line(), "HTTP/1.1 200 OK\r\n");
    EXPECT_EQ(resp.get_content(), "");
    EXPECT_NE(resp.get_headers().find("Content-Length: 0"), std::string::npos);
}

TEST(ResponseBuilder, ResponseWithBody) {
    response resp;  
    std::string body = "Hello World";
    ResponseBuilder(200)
        .withBody(body)
        .build(resp);
    
    EXPECT_EQ(resp.get_content(), body);
    EXPECT_NE(resp.get_headers().find("Content-Length: " + std::to_string(body.size())), std::string::npos);
}

TEST(ResponseBuilder, ResponseWithContentType) {
    response resp;      
    std::string body = "{}";
    ResponseBuilder(200)
        .withContentType("application/json")
        .withBody(body)
        .build(resp);
    
    EXPECT_NE(resp.get_headers().find("Content-Type: application/json"), std::string::npos);
    EXPECT_EQ(resp.get_content(), body);
}

TEST(ResponseBuilder, ResponseWithCustomHeaders) {
    response resp;
    std::string header_name = "X-Custom";
    std::string header_value = "value";
    ResponseBuilder(200)
        .withHeader(header_name, header_value)
        .build(resp);
    
    EXPECT_NE(resp.get_headers().find(header_name + ": " + header_value), std::string::npos);
}

TEST(ResponseBuilder, ResponseWithCustomReasonPhrase) {
    response resp;
    std::string reason_phrase = "I'm a teapot";
    ResponseBuilder(418, reason_phrase).build(resp);
    
    EXPECT_EQ(resp.get_status_line(), "HTTP/1.1 418 " + reason_phrase + "\r\n");
}

TEST(ResponseBuilder, ResponseWithCustomHttpVersion) {
    response resp;
    std::string http_version = "1.0";
    ResponseBuilder(200)
        .withHttpVersion(http_version)
        .build(resp);
    
    EXPECT_EQ(resp.get_status_line(), "HTTP/" + http_version + " 200 OK\r\n");
}

TEST(ResponseBuilder, FluentChaining) {
    response resp;
    std::string body = "<h1>Test</h1>";
    std::string content_type = "text/html";
    std::string cache_control = "no-cache";
    ResponseBuilder(200)
        .withContentType(content_type)
        .withHeader("Cache-Control", cache_control)
        .withBody(body)
        .build(resp);
    
    EXPECT_EQ(resp.get_status_line(), "HTTP/1.1 200 OK\r\n");
    EXPECT_EQ(resp.get_content(), body);
    EXPECT_NE(resp.get_headers().find("Content-Type: " + content_type), std::string::npos);
    EXPECT_NE(resp.get_headers().find("Cache-Control: " + cache_control), std::string::npos);
}

TEST(ResponseBuilder, createOkFactory) {
    response resp;
    ResponseBuilder::createOk().build(resp);
    
    EXPECT_EQ(resp.get_status_line(), "HTTP/1.1 200 OK\r\n");
}

TEST(ResponseBuilder, createNotFoundFactory) {
    response resp;
    ResponseBuilder::createNotFound().build(resp);
    
    EXPECT_EQ(resp.get_status_line(), "HTTP/1.1 404 Not Found\r\n");
}

TEST(ResponseBuilder, createNotFoundFactoryWithMessage) {
    response resp;
    std::string message = "File not found";
    ResponseBuilder::createNotFound(message).build(resp);
    
    EXPECT_EQ(resp.get_status_line(), "HTTP/1.1 404 Not Found\r\n");
    EXPECT_EQ(resp.get_content(), message);
}

TEST(ResponseBuilder, createBadRequestFactory) {
    response resp;
    ResponseBuilder::createBadRequest().build(resp);
    
    EXPECT_EQ(resp.get_status_line(), "HTTP/1.1 400 Bad Request\r\n");
}

TEST(ResponseBuilder, createBadRequestFactoryWithMessage) {
    response resp;
    std::string message = "Invalid input";
    ResponseBuilder::createBadRequest(message).build(resp);
    
    EXPECT_EQ(resp.get_status_line(), "HTTP/1.1 400 Bad Request\r\n");
    EXPECT_EQ(resp.get_content(), message);
}

TEST(ResponseBuilder, createInternalServerErrorFactory) {
    response resp;
    ResponseBuilder::createInternalServerError().build(resp);
    
    EXPECT_EQ(resp.get_status_line(), "HTTP/1.1 500 Internal Server Error\r\n");
}

TEST(ResponseBuilder, createInternalServerErrorFactoryWithMessage) {
    response resp;
    std::string message = "Server error";
    ResponseBuilder::createInternalServerError(message).build(resp);
    
    EXPECT_EQ(resp.get_status_line(), "HTTP/1.1 500 Internal Server Error\r\n");
    EXPECT_EQ(resp.get_content(), message);
}

TEST(ResponseBuilder, FactoryMethodCanBeCustomized) {
    response resp;
    std::string message = "Custom message";
    std::string header_name = "X-Debug";
    std::string header_value = "info";
    ResponseBuilder::createNotFound(message)
        .withHeader(header_name, header_value)
        .build(resp);
    
    EXPECT_EQ(resp.get_content(), message);
    EXPECT_NE(resp.get_headers().find(header_name + ": " + header_value), std::string::npos);
}

TEST(ResponseBuilder, MoveSemantics) {
    response resp;
    std::string body = "Large content";
    std::string body_copy = body;

    ResponseBuilder(200)
        .withBody(std::move(body))
        .build(resp);
    
    // check body 
    EXPECT_EQ(resp.get_content(), body_copy);
}

