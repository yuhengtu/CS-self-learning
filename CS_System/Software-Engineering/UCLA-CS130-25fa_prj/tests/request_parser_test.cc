#include "gtest/gtest.h"
#include "request_parser.h"
#include "request.h"

class RequestParserTestPeer {
public:
    static void ForceState(RequestParser& parser, int state_value) {
        parser.m_state = static_cast<RequestParser::StateMachine>(state_value);
    }
};

// Shared fixture for RequestParser
class RequestParserTest : public ::testing::Test {
protected:
    void SetUp() override {
        parser.reset();
        req.reset();
    }

    void ResetState() {
        parser.reset();
        req.reset();
    }

    RequestParser parser;
    request req;
    RequestParser::Status parseAll(const std::string& data) {
        return parser.parse(req, data.c_str(), data.size());
    }
};

// ------------------ PROPER REQUEST TESTS ------------------

// Valid HTTP/1.1 request
TEST_F(RequestParserTest, ProperSimpleRequest) {
    std::string req_str =
        "GET /index.html HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(req_str), RequestParser::PROPER_REQUEST);
    EXPECT_EQ(this->req.method, "GET");
    EXPECT_EQ(this->req.uri, "/index.html");
    EXPECT_EQ(this->req.version, "1.1");
    ASSERT_EQ(this->req.headers.size(), 2u);
    EXPECT_EQ(this->req.headers[0].first, "Host");
    EXPECT_EQ(this->req.headers[0].second, "example.com");
    EXPECT_EQ(this->req.headers[1].first, "Connection");
    EXPECT_EQ(this->req.headers[1].second, "close");
}

// Multiple headers still valid
TEST_F(RequestParserTest, ProperMultipleHeaders) {
    std::string req_str =
        "GET / HTTP/1.1\r\n"
        "Header1: Value1\r\n"
        "Header2: Value2\r\n"
        "Header3: Value3\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(req_str), RequestParser::PROPER_REQUEST);
}

// Handles empty header value correctly
TEST_F(RequestParserTest, ProperEmptyHeaderValue) {
    std::string req_str =
        "GET / HTTP/1.1\r\n"
        "EmptyHeader: \r\n"
        "\r\n";
    EXPECT_EQ(parseAll(req_str), RequestParser::PROPER_REQUEST);
}

TEST_F(RequestParserTest, RawBufferMatchesInputAndBodyEmpty) {
    std::string req_str =
        "GET /echo HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "User-Agent: TestSuite\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(req_str), RequestParser::PROPER_REQUEST);
    EXPECT_EQ(this->req.raw, req_str);
    EXPECT_TRUE(this->req.body.empty());
}

// Body parsing with Content-Length in single buffer
TEST_F(RequestParserTest, ProperPostWithBodySingleBuffer) {
    std::string body = "{\"name\":\"Alice\"}";
    std::string req_str =
        "POST /api/users HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "\r\n" +
        body;
    EXPECT_EQ(parseAll(req_str), RequestParser::PROPER_REQUEST);
    EXPECT_EQ(this->req.method, "POST");
    EXPECT_EQ(this->req.uri, "/api/users");
    EXPECT_EQ(this->req.get_header_value("content-length"), std::to_string(body.size()));
    EXPECT_EQ(this->req.body, body);
}

// Body parsing across multiple buffers
TEST_F(RequestParserTest, ProperPostWithBodyAcrossBuffers) {
    std::string body = "{\"k\":1}";
    std::string headers =
        "POST /api/data HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "\r\n";
    // First parse only headers
    EXPECT_EQ(parseAll(headers), RequestParser::IN_PROGRESS);
    // Now feed the body
    EXPECT_EQ(parseAll(body), RequestParser::PROPER_REQUEST);
    EXPECT_EQ(this->req.body, body);
}

// Missing Content-Length should not block parsing and body remains empty
TEST_F(RequestParserTest, PostWithoutContentLengthTreatsNoBody) {
    std::string body = "{\"x\":true}";
    std::string req_str =
        "POST /api/no-cl HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "\r\n" +
        body; // body present but no Content-Length: we ignore per simple parser
    EXPECT_EQ(parseAll(req_str), RequestParser::PROPER_REQUEST);
    EXPECT_TRUE(this->req.body.empty());
}

// Explicit empty body with Content-Length: 0
TEST_F(RequestParserTest, ProperPostWithEmptyBodyContentLengthZero) {
    std::string req_str =
        "POST /api/empty HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: 0\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(req_str), RequestParser::PROPER_REQUEST);
    EXPECT_EQ(this->req.method, "POST");
    EXPECT_EQ(this->req.uri, "/api/empty");
    EXPECT_EQ(this->req.get_header_value("content-length"), "0");
    EXPECT_TRUE(this->req.body.empty());
}

// Large body across many buffers
TEST_F(RequestParserTest, ProperPostWithLargeBodyManyBuffers) {
    const std::size_t N = 100000; // 100 KB
    std::string body;
    body.reserve(N);
    for (std::size_t i = 0; i < N; ++i) body.push_back(static_cast<char>('a' + (i % 26)));

    std::string headers =
        "POST /api/large HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "\r\n";

    // Feed headers first
    EXPECT_EQ(parseAll(headers), RequestParser::IN_PROGRESS);

    // Feed body in small chunks
    std::size_t chunk = 1024;
    std::size_t offset = 0;
    while (offset + chunk < body.size()) {
        EXPECT_EQ(parseAll(body.substr(offset, chunk)), RequestParser::IN_PROGRESS);
        offset += chunk;
    }
    // Final chunk
    EXPECT_EQ(parseAll(body.substr(offset)), RequestParser::PROPER_REQUEST);
    EXPECT_EQ(this->req.body.size(), body.size());
    EXPECT_EQ(this->req.body, body);
}

// Extra data beyond Content-Length in same buffer should not be added to body
TEST_F(RequestParserTest, PostWithExtraDataBeyondContentLength) {
    std::string body = "{\"name\":\"Bob\"}";
    std::string extra = "GARBAGE-IGNORED";
    std::string req_str =
        "POST /api/users HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "\r\n" +
        body + extra; // parser should stop at Content-Length

    EXPECT_EQ(parseAll(req_str), RequestParser::PROPER_REQUEST);
    EXPECT_EQ(this->req.body, body);
    // Raw contains everything fed to parser; body must not include extra
    EXPECT_NE(this->req.raw.find(extra), std::string::npos);
}

// Negative Content-Length should be rejected as bad request
TEST_F(RequestParserTest, BadNegativeContentLength) {
    std::string body = "abc";
    std::string req_str =
        "POST /api/neg HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "Content-Length: -5\r\n"
        "\r\n" +
        body;
    EXPECT_EQ(parseAll(req_str), RequestParser::BAD_REQUEST);
}

// Non numeric Content-Length should be rejected as bad request
TEST_F(RequestParserTest, BadNonNumericContentLength) {
    std::string body = "abc";
    std::string req_str =
        "POST /api/nonnum HTTP/1.1\\r\\n"
        "Host: example.com\\r\\n"
        "Content-Length: abc\\r\\n"
        "\\r\\n" +
        body;
    EXPECT_EQ(parseAll(req_str), RequestParser::BAD_REQUEST);
}

TEST_F(RequestParserTest, HeaderLookupHelperCaseInsensitive) {
    std::string req_str =
        "GET /case HTTP/1.1\r\n"
        "Host: Example.COM\r\n"
        "X-Custom: Value\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(req_str), RequestParser::PROPER_REQUEST);
    EXPECT_EQ(this->req.get_header_value("host"), "Example.COM");
    EXPECT_EQ(this->req.get_header_value("HOST"), "Example.COM");
    EXPECT_EQ(this->req.get_header_value("x-custom"), "Value");
    EXPECT_EQ(this->req.get_header_value("missing"), ""); // try a header that doesn't exist
}

// Test that request::reset() properly clears all fields
TEST_F(RequestParserTest, RequestResetClearsAllFields) {
    std::string req_str =
        "POST /api/data HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "Content-Type: application/json\r\n"
        "\r\n";
    
    // Parse a request to populate all fields
    EXPECT_EQ(parseAll(req_str), RequestParser::PROPER_REQUEST);
    
    // Verify all fields are populated
    EXPECT_FALSE(this->req.method.empty());
    EXPECT_EQ(this->req.method, "POST");
    EXPECT_FALSE(this->req.uri.empty());
    EXPECT_EQ(this->req.uri, "/api/data");
    EXPECT_FALSE(this->req.version.empty());
    EXPECT_EQ(this->req.version, "1.1");
    EXPECT_FALSE(this->req.headers.empty());
    EXPECT_EQ(this->req.headers.size(), 2);
    EXPECT_FALSE(this->req.raw.empty());
    EXPECT_EQ(this->req.raw, req_str);
    // body is empty in this case but raw is not
    
    // Now reset the request
    this->req.reset();
    
    // Verify all fields are cleared
    EXPECT_TRUE(this->req.method.empty());
    EXPECT_TRUE(this->req.uri.empty());
    EXPECT_TRUE(this->req.version.empty());
    EXPECT_TRUE(this->req.headers.empty());
    EXPECT_TRUE(this->req.body.empty());
    EXPECT_TRUE(this->req.raw.empty());
}

// ------------------ IN PROGRESS TESTS ------------------

TEST_F(RequestParserTest, InProgressPartialRequestLine) {
    std::string partial = "GET /index.html HTTP/1.";
    EXPECT_EQ(parseAll(partial), RequestParser::IN_PROGRESS);
}

TEST_F(RequestParserTest, InProgressMissingFinalCRLF) {
    std::string partial =
        "GET / HTTP/1.1\r\n"
        "Host: example.com\r\n";
    EXPECT_EQ(parseAll(partial), RequestParser::IN_PROGRESS);
}

// ------------------ BAD REQUEST TESTS ------------------

// Invalid lowercase method
TEST_F(RequestParserTest, BadLowercaseMethod) {
    std::string bad = "get /index.html HTTP/1.1\r\n\r\n";
    EXPECT_EQ(parseAll(bad), RequestParser::BAD_REQUEST);
}

// Unsupported HTTP version (HTTP/1.0)
TEST_F(RequestParserTest, BadHTTP10Version) {
    std::string bad =
        "GET /index.html HTTP/1.0\r\n"
        "Host: example.com\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(bad), RequestParser::BAD_REQUEST);
}

// Invalid version format
TEST_F(RequestParserTest, BadInvalidVersionFormat) {
    std::string bad =
        "GET /index.html HTTP/2\r\n"
        "Host: example.com\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(bad), RequestParser::BAD_REQUEST);
}

// Missing HTTP version entirely
TEST_F(RequestParserTest, BadMissingHTTPVersion) {
    std::string bad = "GET /index.html\r\n\r\n";
    EXPECT_EQ(parseAll(bad), RequestParser::BAD_REQUEST);
}

// Missing colon in header
TEST_F(RequestParserTest, BadMissingColonInHeader) {
    std::string bad =
        "GET / HTTP/1.1\r\n"
        "Host example.com\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(bad), RequestParser::BAD_REQUEST);
}

// Newline in header name
TEST_F(RequestParserTest, BadHeaderNameNewline) {
    std::string bad =
        "GET / HTTP/1.1\r\n"
        "Host\nexample: test\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(bad), RequestParser::BAD_REQUEST);
}

// LF without CR
TEST_F(RequestParserTest, BadLFWithoutCR) {
    std::string bad =
        "1 started with a number not an uppercase letter";
    EXPECT_EQ(parseAll(bad), RequestParser::BAD_REQUEST);
}

// START --- EACH STATE IN FINITE STATE MACHINE FAILURE 
TEST_F(RequestParserTest, StartFail) {
    std::string s =
        "1GET /index.html HTTP/1.1\r\n" // Starts with '1'
        "Host: example.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(s), RequestParser::BAD_REQUEST);
}

TEST_F(RequestParserTest, MethodFail) {
    std::string s =
        "GE1T /index.html HTTP/1.1\r\n" // Has '1' in Method (not all uppercase letters)
        "Host: example.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(s), RequestParser::BAD_REQUEST);
}

TEST_F(RequestParserTest, UriFail) {
    std::string s =
        "GET /inde\rx.html HTTP/1.1\r\n" // CR in URI
        "Host: example.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(s), RequestParser::BAD_REQUEST);
}

TEST_F(RequestParserTest, HFail) {
    std::string s =
        "GET /index.html aTTP/1.1\r\n" // aTTP not HTTP
        "Host: example.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(s), RequestParser::BAD_REQUEST);
}

TEST_F(RequestParserTest, T1Fail) {
    std::string s =
        "GET /index.html HxTP/1.1\r\n" // HxTP not HTTP
        "Host: example.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(s), RequestParser::BAD_REQUEST);
}

TEST_F(RequestParserTest, T2Fail) {
    std::string s =
        "GET /index.html HT3P/1.1\r\n" // HT3P not HTTP
        "Host: example.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(s), RequestParser::BAD_REQUEST);
}

TEST_F(RequestParserTest, PFail) {
    std::string s =
        "GET /index.html HTT/1.1\r\n" // HTT not HTTP
        "Host: example.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(s), RequestParser::BAD_REQUEST);
}

TEST_F(RequestParserTest, SlashFail) {
    std::string s =
        "GET /index.html HTTP-1.1\r\n" // HTTP-1.1 not HTTP/1.1
        "Host: example.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(s), RequestParser::BAD_REQUEST);
}

TEST_F(RequestParserTest, VersionFail) {
    std::string s =
        "GET /index.html HTTP/1.3\r\n" // 1.3 not 1.1
        "Host: example.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(s), RequestParser::BAD_REQUEST);
}

TEST_F(RequestParserTest, RequestLineCRFail) {
    std::string s =
        "GET /index.html HTTP/1.1\r\r\n" // \r\r\n not \r\n
        "Host: example.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(s), RequestParser::BAD_REQUEST);
}

TEST_F(RequestParserTest, HeaderNameFail) {
    std::string s =
        "GET /index.html HTTP/1.1\r\n"
        "Ho\rst: example.com\r\n" // Ho\rst not Host
        "Connection: close\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(s), RequestParser::BAD_REQUEST);
}

TEST_F(RequestParserTest, HeaderColonFail) {
    std::string s =
        "GET /index.html HTTP/1.1\r\n"
        "Host:example.com\r\n" // Host:example.com not Host: example.com
        "Connection: close\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(s), RequestParser::BAD_REQUEST);
}

TEST_F(RequestParserTest, HeaderLineCRFail) {
    std::string s =
        "GET /index.html HTTP/1.1\r\n"
        "Host: example.com\rmorecontent\n" // .com\rmorecontent\n not .com\r\n
        "Connection: close\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(s), RequestParser::BAD_REQUEST);
}

TEST_F(RequestParserTest, EndCRFail) {
    std::string s =
        "GET /index.html HTTP/1.1\r\n"
        "Host: example.com\r\n" 
        "Connection: close\r\n"
        "\rmore stuff\n"; // \rmorestuff\n not \r\n
    EXPECT_EQ(parseAll(s), RequestParser::BAD_REQUEST);
}
// END --- EACH STATE IN FINITE STATE MACHINE FAILURE 

// ------------------ EDGE CASE TESTS ------------------

// No headers (just request line and blank line)
TEST_F(RequestParserTest, ProperNoHeaders) {
    std::string req_str =
        "GET / HTTP/1.1\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(req_str), RequestParser::PROPER_REQUEST);
}

// Header with symbols
TEST_F(RequestParserTest, ProperHeaderWithSymbols) {
    std::string req_str =
        "GET / HTTP/1.1\r\n"
        "X-Data: @#$%^&*()_+\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(req_str), RequestParser::PROPER_REQUEST);
}

// Empty header value followed by another header (HEADER_SPACE -> HEADER_LINE_CR -> HEADER_START loop)
TEST_F(RequestParserTest, ProperEmptyHeaderThenAnotherHeader) {
    std::string req_str =
        "GET / HTTP/1.1\r\n"
        "H1: \r\n"              // empty value
        "H2: v\r\n"             // another header after the empty one
        "\r\n";
    EXPECT_EQ(parseAll(req_str), RequestParser::PROPER_REQUEST);
}

// After a successful parse, feeding another request without resetting should fail.
// This hits the END_CR state's "else return BAD_REQUEST" branch with a non-'\n' char.
TEST_F(RequestParserTest, BadPipelinedSecondRequestWithoutReset) {
    // First, a valid request
    std::string ok =
        "GET / HTTP/1.1\r\n"
        "Host: x\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(ok), RequestParser::PROPER_REQUEST);

    // Now try to parse a new request on the same parser (state not reset)
    std::string second = "GET / HTTP/1.1\r\n\r\n";
    EXPECT_EQ(parseAll(second), RequestParser::BAD_REQUEST);

    // Now reset and ensure a fresh request succeeds.
    ResetState();
    std::string fresh =
        "GET /fresh HTTP/1.1\r\n"
        "Host: z\r\n"
        "\r\n";
    EXPECT_EQ(parseAll(fresh), RequestParser::PROPER_REQUEST);
}

// Version path: explicitly walk through '1' and '.' acceptance in VERSION state
TEST_F(RequestParserTest, ProperVersionCharactersAccepted) {
    // Feed in pieces to ensure VERSION consumes both '1' and '.' paths before CR
    std::string part1 = "GET / HTTP/1.";     // up to dot
    EXPECT_EQ(parseAll(part1), RequestParser::IN_PROGRESS);

    std::string part2 = "1\r";               // finish version with '1' then CR
    EXPECT_EQ(parseAll(part2), RequestParser::IN_PROGRESS);

    std::string finish = "\n\r\n";           // LF, then blank line
    EXPECT_EQ(parseAll(finish), RequestParser::PROPER_REQUEST);
}

// Evil test: force the parser into an impossible FSM state to cover default -> fail_impossible_state_()
TEST_F(RequestParserTest, ImpossibleStateTriggersFailGuard) {
    RequestParserTestPeer::ForceState(parser, 999);
    std::string data = "X"; // any byte; we just need to enter the switch once
    EXPECT_EQ(parseAll(data), RequestParser::BAD_REQUEST);
}
