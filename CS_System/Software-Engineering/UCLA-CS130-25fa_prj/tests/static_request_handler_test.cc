#include "gtest/gtest.h"

#include "static_request_handler.h"
#include "request_handler.h"
#include "response.h"
#include "request.h"

#include <string>
#include <string_view>
#include <vector>
#include <fstream>
#include <sstream>
#include <iterator>
#include <memory>
#include <boost/asio.hpp>

constexpr std::string_view kCrlf = "\r\n";
constexpr std::string_view kDoubleCrlf = "\r\n\r\n";
constexpr std::string_view kContentLengthHeader = "Content-Length: ";

// Flatten reply buffers to a single string
static std::string flatten(const std::unique_ptr<response>& resp) {
  std::string out;
  const auto& bufs = resp->get_bufs();
  for (const auto& b : bufs) {
    const char* p = static_cast<const char*>(b.data());
    out.append(p, p + b.size());
  }
  return out;
}

// Read an entire file (binary) into a std::string
static std::string read_all(const std::string& path) {
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  if (!ifs.good()) return {};
  std::ostringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

// Build a minimal request for a given URI
static request MakeRequest(const std::string& uri) {
  request req;
  req.method = "GET";
  req.uri = uri;
  req.version = "1.1";
  std::ostringstream raw;
  raw << "GET " << uri << " HTTP/1.1\r\n"
      << "Host: localhost\r\n"
      << "\r\n";
  req.raw = raw.str();
  return req;
}

class StaticHandlerFixturesTest : public ::testing::Test {
 protected:
  static constexpr const char* kMount = "/static";
  // CMake sets WORKING_DIRECTORY to ${CMAKE_CURRENT_SOURCE_DIR}/tests,
  // so these paths are relative to tests/
  static constexpr const char* kRootDir = "static_test_files";
};

// ---------------------- Success cases ----------------------

TEST_F(StaticHandlerFixturesTest, ServesHtml) {
  const std::string fname = "hello.html";
  const std::string expected = read_all(std::string(kRootDir) + "/" + fname);
  ASSERT_FALSE(expected.empty()) << "Missing fixture: " << fname;

  StaticRequestHandler handler(kMount, kRootDir);
  std::unique_ptr<response> resp; // To hold the result
  auto req = MakeRequest(std::string(kMount) + "/" + fname);

  resp = handler.handle_request(req); 
  ASSERT_NE(resp, nullptr);
  
  const std::string resp_str = flatten(resp);

  EXPECT_NE(resp_str.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
  EXPECT_NE(resp_str.find("Content-Type: text/html"), std::string::npos);

  const auto body_pos = resp_str.find(kDoubleCrlf);
  ASSERT_NE(body_pos, std::string::npos);
  const std::string body = resp_str.substr(body_pos + kDoubleCrlf.length());
  EXPECT_EQ(body, expected);

  const auto cl_pos = resp_str.find(kContentLengthHeader);
  ASSERT_NE(cl_pos, std::string::npos);
  const auto cl_end = resp_str.find(kCrlf, cl_pos);
  ASSERT_NE(cl_end, std::string::npos);
  const auto len = std::stoul(resp_str.substr(cl_pos + kContentLengthHeader.length(), 
                                               cl_end - (cl_pos + kContentLengthHeader.length())));
  EXPECT_EQ(len, expected.size());
}

TEST_F(StaticHandlerFixturesTest, ServesTxt) {
  const std::string fname = "note.txt";
  const std::string expected = read_all(std::string(kRootDir) + "/" + fname);
  ASSERT_FALSE(expected.empty()) << "Missing fixture: " << fname;

  StaticRequestHandler handler(kMount, kRootDir);
  std::unique_ptr<response> resp; // To hold the result
  auto req = MakeRequest(std::string(kMount) + "/" + fname);

  resp = handler.handle_request(req); 
  ASSERT_NE(resp, nullptr);
  
  const std::string resp_str = flatten(resp);

  EXPECT_NE(resp_str.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
  EXPECT_NE(resp_str.find("Content-Type: text/plain"), std::string::npos);

  const auto body_pos = resp_str.find(kDoubleCrlf);
  ASSERT_NE(body_pos, std::string::npos);
  const std::string body = resp_str.substr(body_pos + kDoubleCrlf.length());
  EXPECT_EQ(body, expected);
}

TEST_F(StaticHandlerFixturesTest, ServesJpgBinaryUnchanged) {
  const std::string fname = "img.jpg";
  const std::string expected = read_all(std::string(kRootDir) + "/" + fname);
  ASSERT_FALSE(expected.empty()) << "Missing fixture: " << fname;

  StaticRequestHandler handler(kMount, kRootDir);
  std::unique_ptr<response> resp; // To hold the result
  auto req = MakeRequest(std::string(kMount) + "/" + fname);

  resp = handler.handle_request(req); 
  ASSERT_NE(resp, nullptr);

  const std::string resp_str = flatten(resp);

  EXPECT_NE(resp_str.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
  EXPECT_NE(resp_str.find("Content-Type: image/jpeg"), std::string::npos);

  const auto body_pos = resp_str.find(kDoubleCrlf);
  ASSERT_NE(body_pos, std::string::npos);
  const std::string body = resp_str.substr(body_pos + kDoubleCrlf.length());
  EXPECT_EQ(body.size(), expected.size());
  EXPECT_TRUE(std::equal(body.begin(), body.end(), expected.begin()));
}

TEST_F(StaticHandlerFixturesTest, ServesZipBinaryUnchanged) {
  const std::string fname = "archive.zip";
  const std::string expected = read_all(std::string(kRootDir) + "/" + fname);
  ASSERT_FALSE(expected.empty()) << "Missing fixture: " << fname;

  StaticRequestHandler handler(kMount, kRootDir);
  std::unique_ptr<response> resp; // To hold the result
  auto req = MakeRequest(std::string(kMount) + "/" + fname);

  resp = handler.handle_request(req); 
  ASSERT_NE(resp, nullptr);

  const std::string resp_str = flatten(resp);

  EXPECT_NE(resp_str.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
  EXPECT_NE(resp_str.find("Content-Type: application/zip"), std::string::npos);

  const auto body_pos = resp_str.find(kDoubleCrlf);
  ASSERT_NE(body_pos, std::string::npos);
  const std::string body = resp_str.substr(body_pos + kDoubleCrlf.length());
  EXPECT_EQ(body.size(), expected.size());
  EXPECT_TRUE(std::equal(body.begin(), body.end(), expected.begin()));
}

// ---------------------- Error cases (404) ----------------------

TEST_F(StaticHandlerFixturesTest, MissingFileReturns404) {
  StaticRequestHandler handler(kMount, kRootDir);
  std::unique_ptr<response> resp; // To hold the result
  auto req = MakeRequest(std::string(kMount) + "/nope.html");

  resp = handler.handle_request(req); 
  ASSERT_NE(resp, nullptr);

  const std::string resp_str = flatten(resp);
  EXPECT_NE(resp_str.find("HTTP/1.1 404 Not Found\r\n"), std::string::npos);
}

TEST_F(StaticHandlerFixturesTest, UnsupportedExtensionReturns404WithoutTouchingDisk) {
  StaticRequestHandler handler(kMount, kRootDir);
  std::unique_ptr<response> resp; // To hold the result
  auto req = MakeRequest(std::string(kMount) + "/data.bin");

  resp = handler.handle_request(req); 
  ASSERT_NE(resp, nullptr);

  const std::string resp_str = flatten(resp);
  EXPECT_NE(resp_str.find("HTTP/1.1 404 Not Found\r\n"), std::string::npos);
}

TEST_F(StaticHandlerFixturesTest, MountPrefixButNoFileReturns404) {
  StaticRequestHandler handler(kMount, kRootDir);
  std::unique_ptr<response> resp; // To hold the result
  auto req = MakeRequest(std::string(kMount) + "/");

  resp = handler.handle_request(req); 
  ASSERT_NE(resp, nullptr);

  const std::string resp_str = flatten(resp);
  EXPECT_NE(resp_str.find("HTTP/1.1 404 Not Found\r\n"), std::string::npos);
}

TEST_F(StaticHandlerFixturesTest, PrefixMismatchReturns404) {
  StaticRequestHandler handler(kMount, kRootDir);
  std::unique_ptr<response> resp; // To hold the result
  auto req = MakeRequest("/public/hello.html");

  resp = handler.handle_request(req); 
  ASSERT_NE(resp, nullptr);

  const std::string resp_str = flatten(resp);
  EXPECT_NE(resp_str.find("HTTP/1.1 404 Not Found\r\n"), std::string::npos);
}

TEST_F(StaticHandlerFixturesTest, DotDotAttackThrows) {
  StaticRequestHandler handler(kMount, kRootDir);
  std::unique_ptr<response> resp; // To hold the result

  auto req = MakeRequest(std::string(kMount) + "/../../etc/passwd");

  resp = handler.handle_request(req); 
  ASSERT_NE(resp, nullptr);

  const std::string resp_str = flatten(resp);
  EXPECT_NE(resp_str.find("HTTP/1.1 404 Not Found\r\n"), std::string::npos);
}

TEST_F(StaticHandlerFixturesTest, EncodedTraversalReturns404) {
  StaticRequestHandler handler(kMount, kRootDir);
  std::unique_ptr<response> resp; // To hold the result

  auto req = MakeRequest(std::string(kMount) + "/..%2F..%2Fsecret.txt");

  resp = handler.handle_request(req); 
  ASSERT_NE(resp, nullptr);

  const std::string resp_str = flatten(resp);
  EXPECT_NE(resp_str.find("HTTP/1.1 404 Not Found\r\n"), std::string::npos);
}

TEST_F(StaticHandlerFixturesTest, MixedTraversalThrows) {
  StaticRequestHandler handler(kMount, kRootDir);
  std::unique_ptr<response> resp; // To hold the result

  auto req = MakeRequest(std::string(kMount) + "/images/../..//private/config.yaml");

  resp = handler.handle_request(req); 
  ASSERT_NE(resp, nullptr);

  const std::string resp_str = flatten(resp);
  EXPECT_NE(resp_str.find("HTTP/1.1 404 Not Found\r\n"), std::string::npos);
}