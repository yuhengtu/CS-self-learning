#include "gtest/gtest.h"
#include "response.h"

#include <string>
#include <vector>
#include <boost/asio.hpp>

TEST(Response, BufsSetCorrectly) {
    response r;

    std::string status = "HTTP/1.1 200 OK\r\n";
    std::string headers = "Content-Type: text/plain\r\n";
    std::string content = "some content";

    r.set_status_line(status);
    r.set_headers(headers);
    r.set_content(content);

    r.set_bufs();
    const auto& bufs = r.get_bufs();

    ASSERT_EQ(bufs.size(), 3);

    auto to_string = [](const auto& buf) { return std::string(static_cast<const char*>(buf.data()), buf.size()); };

    EXPECT_EQ(to_string(bufs[0]), status);
    EXPECT_EQ(to_string(bufs[1]), headers);
    EXPECT_EQ(to_string(bufs[2]), content);
}

TEST(Response, GettersReturnStoredValues) {
    response r;

    std::string status = "HTTP/1.1 201 Created\r\n";
    std::string headers = "Content-Type: application/json\r\nX-Test: 1\r\n";
    std::string content = "{\"ok\":true}";

    r.set_status_line(status);
    r.set_headers(headers);
    r.set_content(content);

    EXPECT_EQ(r.get_status_line(), status);
    EXPECT_EQ(r.get_headers(), headers);
    EXPECT_EQ(r.get_content(), content);
}

TEST(Response, StatusLineUnset) {
    response r;
    r.set_headers("Content-Type: text/plain\r\n");
    r.set_content("Hello");

    EXPECT_THROW({
        r.set_bufs();
    }, std::runtime_error);
}

TEST(Response, HeadersUnset) {
    response r;
    r.set_status_line("HTTP/1.1 200 OK\r\n");
    r.set_content("Hello");

    EXPECT_THROW({
        r.set_bufs();
    }, std::runtime_error);
}

TEST(Response, ContentUnset) {
    response r;
    r.set_status_line("HTTP/1.1 200 OK\r\n");
    r.set_headers("Content-Type: text/plain\r\n");

    EXPECT_THROW({
        r.set_bufs();
    }, std::runtime_error);
}

TEST(Response, EverythingUnset) {
    response r;

    EXPECT_THROW({
        r.set_bufs();
    }, std::runtime_error);
}

