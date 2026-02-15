#include "gtest/gtest.h"
#define private public
#include "config_parser.h"
#undef private
#include <sstream>
// Tests a simple config with a statement and a block with a few statements.
// Expects success.

class NginxConfigParserTestFixture: public testing::Test {
    protected:
        NginxConfigParserTestFixture() {}
        
        // Helper method to parse from a string
        bool ParseFromString(const std::string& config_string) {
            std::stringstream config_stream(config_string);
            return parser.Parse(&config_stream, &out_config);
        }
        
        NginxConfigParser parser;
        NginxConfig out_config;
};

TEST(NginxConfigParserTest, SimpleConfig) {
  NginxConfigParser parser;
  NginxConfig out_config;
  bool success = parser.Parse("test_config_1", &out_config);
  EXPECT_TRUE(success);
}
// Tests a simple nested config with one outerblock and two inner blocks on the same level.
// Expects success.
TEST_F(NginxConfigParserTestFixture, NestedConfig) {
    bool success = parser.Parse("test_config_2", &out_config);
    EXPECT_TRUE(success);
}
// ==================== SIMPLE STATEMENT TESTS ====================
// Test single token statement
TEST_F(NginxConfigParserTestFixture, SingleTokenStatement) {
    EXPECT_TRUE(ParseFromString("file;"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[0], "file");
}
// Test two token statement
TEST_F(NginxConfigParserTestFixture, TwoTokenStatement) {
    EXPECT_TRUE(ParseFromString("file list;"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_.size(), 2);
    EXPECT_EQ(out_config.statements_[0]->tokens_[0], "file");
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "list");
}
// Test multiple token statement
TEST_F(NginxConfigParserTestFixture, MultipleTokenStatement) {
    EXPECT_TRUE(ParseFromString("listen 80 default_server;"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_.size(), 3);
    EXPECT_EQ(out_config.statements_[0]->tokens_[0], "listen");
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "80");
    EXPECT_EQ(out_config.statements_[0]->tokens_[2], "default_server");
}
// Test multiple statements
TEST_F(NginxConfigParserTestFixture, MultipleStatements) {
    EXPECT_TRUE(ParseFromString("file list; baz qux;"));
    ASSERT_EQ(out_config.statements_.size(), 2);
    EXPECT_EQ(out_config.statements_[0]->tokens_[0], "file");
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "list");
    EXPECT_EQ(out_config.statements_[1]->tokens_[0], "baz");
    EXPECT_EQ(out_config.statements_[1]->tokens_[1], "qux");
}
// ==================== COMMENT TESTS ====================
// Test comment at beginning
TEST_F(NginxConfigParserTestFixture, CommentAtBeginning) {
    EXPECT_TRUE(ParseFromString("# this is a comment\nfile list;"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[0], "file");
}
// Test multiple comments
TEST_F(NginxConfigParserTestFixture, MultipleComments) {
    EXPECT_TRUE(ParseFromString("# comment 1\n# comment 2\nfile list;"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[0], "file");
}
// Test comment between statements
TEST_F(NginxConfigParserTestFixture, CommentBetweenStatements) {
    EXPECT_TRUE(ParseFromString("file list;\n# comment\nbaz qux;"));
    ASSERT_EQ(out_config.statements_.size(), 2);
    EXPECT_EQ(out_config.statements_[0]->tokens_[0], "file");
    // make sure that the middle comment statement is ignored, so next statement is baz
    EXPECT_EQ(out_config.statements_[1]->tokens_[0], "baz");
}
// Test empty comment
TEST_F(NginxConfigParserTestFixture, EmptyComment) {
    EXPECT_TRUE(ParseFromString("#\nfile list;"));
    ASSERT_EQ(out_config.statements_.size(), 1);
}
// Test comment inside block
TEST_F(NginxConfigParserTestFixture, CommentInsideBlock) {
    EXPECT_TRUE(ParseFromString("server {\n  # comment\n  listen 80;\n}"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    ASSERT_NE(out_config.statements_[0]->child_block_, nullptr);
    EXPECT_EQ(out_config.statements_[0]->child_block_->statements_.size(), 1);
}
// ==================== SIMPLE BLOCK TESTS ====================
// Test single empty block
TEST_F(NginxConfigParserTestFixture, EmptyBlock) {
    EXPECT_TRUE(ParseFromString("server {}"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[0], "server");
    // have a child block that's empty
    ASSERT_NE(out_config.statements_[0]->child_block_, nullptr);
    EXPECT_EQ(out_config.statements_[0]->child_block_->statements_.size(), 0);
}
// Test block with single statement
TEST_F(NginxConfigParserTestFixture, BlockWithStatement) {
    EXPECT_TRUE(ParseFromString("server { listen 80; }"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    ASSERT_NE(out_config.statements_[0]->child_block_, nullptr);
    EXPECT_EQ(out_config.statements_[0]->child_block_->statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->child_block_->statements_[0]->tokens_[0], "listen");
}
// Test double nested blocks
TEST_F(NginxConfigParserTestFixture, DoubleNestedBlock) {
    EXPECT_TRUE(ParseFromString("server { location / { root /data; } }"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    ASSERT_NE(out_config.statements_[0]->child_block_, nullptr);
    ASSERT_EQ(out_config.statements_[0]->child_block_->statements_.size(), 1);
    ASSERT_NE(out_config.statements_[0]->child_block_->statements_[0]->child_block_, nullptr);
    EXPECT_EQ(out_config.statements_[0]->child_block_->statements_[0]->child_block_->statements_.size(), 1);
}
// ==================== QUOTE TESTS ====================
// Test double quoted string
TEST_F(NginxConfigParserTestFixture, DoubleQuotedString) {
    EXPECT_TRUE(ParseFromString("file \"list\";"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_.size(), 2);
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "\"list\"");
}
// Test single quoted string
TEST_F(NginxConfigParserTestFixture, SingleQuotedString) {
    EXPECT_TRUE(ParseFromString("file 'list';"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "'list'");
}
// Test quoted string with spaces
TEST_F(NginxConfigParserTestFixture, QuotedStringWithSpaces) {
    EXPECT_TRUE(ParseFromString("file \"list baz\";"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "\"list baz\"");
}
// Test quoted string with special characters
TEST_F(NginxConfigParserTestFixture, QuotedStringWithSpecialChars) {
    EXPECT_TRUE(ParseFromString("file \"list;{}\";"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "\"list;{}\"");
}
// ==================== WHITESPACE TESTS ====================
// Test multiple spaces between tokens
TEST_F(NginxConfigParserTestFixture, MultipleSpaces) {
    EXPECT_TRUE(ParseFromString("file    list;"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_.size(), 2);
}
// Test tabs between tokens
TEST_F(NginxConfigParserTestFixture, TabsBetweenTokens) {
    EXPECT_TRUE(ParseFromString("file\t\tlist;"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_.size(), 2);
}
// Test newlines between statements
TEST_F(NginxConfigParserTestFixture, NewlinesBetweenStatements) {
    EXPECT_TRUE(ParseFromString("file list;\n\nbaz qux;"));
    ASSERT_EQ(out_config.statements_.size(), 2);
}
// ==================== INVALID CONFIG TESTS ====================
// Test missing semicolon
TEST_F(NginxConfigParserTestFixture, MissingSemicolon) {
    EXPECT_FALSE(ParseFromString("file list"));
}
// Test unclosed block
TEST_F(NginxConfigParserTestFixture, UnclosedBlock) {
    EXPECT_FALSE(ParseFromString("server { listen 80;"));
}
// Test unmatched close brace
TEST_F(NginxConfigParserTestFixture, UnmatchedCloseBrace) {
    EXPECT_FALSE(ParseFromString("server { } }"));
}
// Test multiple unmatched close braces
TEST_F(NginxConfigParserTestFixture, MultipleUnmatchedCloseBraces) {
    EXPECT_FALSE(ParseFromString("server { } } }"));
}
// Test missing open brace
TEST_F(NginxConfigParserTestFixture, MissingOpenBrace) {
    EXPECT_FALSE(ParseFromString("server listen 80; }"));
}
// Test unclosed single quote
TEST_F(NginxConfigParserTestFixture, UnclosedSingleQuote) {
    EXPECT_FALSE(ParseFromString("file 'list;"));
}
// Test unclosed double quote
TEST_F(NginxConfigParserTestFixture, UnclosedDoubleQuote) {
    EXPECT_FALSE(ParseFromString("file \"list;"));
}
// Test empty statement (just semicolon)
TEST_F(NginxConfigParserTestFixture, EmptyStatement) {
    EXPECT_FALSE(ParseFromString(";"));
}
// Test statement before block without token
TEST_F(NginxConfigParserTestFixture, BlockWithoutToken) {
    EXPECT_FALSE(ParseFromString("{ listen 80; }"));
}
// ==================== QUOTE ESCAPING TESTS ====================
// Test escaped single quote
TEST_F(NginxConfigParserTestFixture, EscapedSingleQuote) {
    // `file 'don\'t';` should be valid because you escape the apostrophe
    EXPECT_TRUE(ParseFromString("file 'don\\'t';"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "'don't'");
}
// Test escaped double quote
TEST_F(NginxConfigParserTestFixture, EscapedDoubleQuote) {
    // `file "say \"hi\""";` should be valid because you escape the double quote
    EXPECT_TRUE(ParseFromString("file \"say \\\"hi\\\"\";"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "\"say \"hi\"\"");
}
// Test escaped backslash
TEST_F(NginxConfigParserTestFixture, EscapedBackslash) {
    // `file "path\\to\\file";` should be valid because you escape the backslash
    EXPECT_TRUE(ParseFromString("file \"path\\\\to\\\\file\";"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "\"path\\to\\file\"");
}
// Test that escaping the closing quote leaves string unclosed
TEST_F(NginxConfigParserTestFixture, EscapedQuotePreventsClosing) {
    EXPECT_FALSE(ParseFromString("file \"hi\\\";"));  // "hi\" - quote is escaped
}
// Test escaped single quote leaves string unclosed
TEST_F(NginxConfigParserTestFixture, EscapedSingleQuotePreventsClosing) {
    EXPECT_FALSE(ParseFromString("file 'can\\'t;"));  // 'can\'t (missing closing quote)
}
// Test quoted string must be followed by whitespace or semicolon
TEST_F(NginxConfigParserTestFixture, QuotedStringFollowedByToken) {
    EXPECT_FALSE(ParseFromString("file \"list\"baz;"));
}
// Test quoted string cannot be immediately followed by brace
TEST_F(NginxConfigParserTestFixture, QuotedStringFollowedByBrace) {
    EXPECT_FALSE(ParseFromString("file \"list\"{};"));
}
// Test quoted string followed by semicolon is valid
TEST_F(NginxConfigParserTestFixture, QuotedStringFollowedBySemicolon) {
    EXPECT_TRUE(ParseFromString("file \"list\";"));
}
// Test quoted string followed by whitespace then brace is valid
TEST_F(NginxConfigParserTestFixture, QuotedStringThenWhitespaceThenBrace) {
    EXPECT_TRUE(ParseFromString("server \"name\" { }"));
}

// ---------- NginxConfig/NginxConfigStatement::ToString tests ----------

// Empty config -> empty string
TEST(NginxConfigToString, EmptyConfig) {
    NginxConfig cfg;
    EXPECT_EQ(cfg.ToString(0), "");
}

// Simple statement, no indentation
TEST(NginxConfigStatementToString, SimpleNoIndent) {
    NginxConfigStatement s;
    s.tokens_.push_back("listen");
    s.tokens_.push_back("8080");
    EXPECT_EQ(s.ToString(0), "listen 8080;\n");
}

// Simple statement with indentation depth=2 (four spaces)
TEST(NginxConfigStatementToString, SimpleWithIndentDepthTwo) {
    NginxConfigStatement s;
    s.tokens_.push_back("root");
    s.tokens_.push_back("/var/www");
    EXPECT_EQ(s.ToString(2), "    root /var/www;\n");
}

// Block statement with one child
TEST(NginxConfigStatementToString, BlockWithChild) {
    // child block: { "listen 8080;" }
    std::unique_ptr<NginxConfig> child(new NginxConfig);
    std::unique_ptr<NginxConfigStatement> inner(new NginxConfigStatement);
    inner->tokens_.push_back("listen");
    inner->tokens_.push_back("8080");
    child->statements_.emplace_back(std::move(inner));

    // parent: "server { ... }"
    NginxConfigStatement parent;
    parent.tokens_.push_back("server");
    parent.child_block_.reset(child.release());

    // depth=0 should indent inner by two spaces and close brace aligned
    const std::string expected =
        "server {\n"
        "  listen 8080;\n"
        "}\n";
    EXPECT_EQ(parent.ToString(0), expected);
}

// Config concatenates statements, propagates depth to each
TEST(NginxConfigToString, MultipleStatementsAndDepthPropagation) {
    NginxConfig cfg;

    // "user nginx;"
    std::unique_ptr<NginxConfigStatement> s1(new NginxConfigStatement);
    s1->tokens_.push_back("user");
    s1->tokens_.push_back("nginx");
    cfg.statements_.emplace_back(std::move(s1));

    // "server { listen 8080; }"
    std::unique_ptr<NginxConfig> child(new NginxConfig);
    std::unique_ptr<NginxConfigStatement> inner(new NginxConfigStatement);
    inner->tokens_.push_back("listen");
    inner->tokens_.push_back("8080");
    child->statements_.emplace_back(std::move(inner));

    std::unique_ptr<NginxConfigStatement> block(new NginxConfigStatement);
    block->tokens_.push_back("server");
    block->child_block_.reset(child.release());
    cfg.statements_.emplace_back(std::move(block));

    const std::string expected_depth0 =
        "user nginx;\n"
        "server {\n"
        "  listen 8080;\n"
        "}\n";
    EXPECT_EQ(cfg.ToString(0), expected_depth0);

    const std::string expected_depth1 =
        "  user nginx;\n"
        "  server {\n"
        "    listen 8080;\n"
        "  }\n";
    EXPECT_EQ(cfg.ToString(1), expected_depth1);
}

// ==================== EXTRA EDGE/ERROR PATH TESTS (final) ====================

// 1) Empty file: forces START -> EOF error path in Parse()
TEST_F(NginxConfigParserTestFixture, EmptyFileIsInvalid) {
    EXPECT_FALSE(ParseFromString(""));
}

// 2) Lone closing brace at root: triggers END_BLOCK with stack.size() <= 1
TEST_F(NginxConfigParserTestFixture, CloseBlockWithoutOpen) {
    EXPECT_FALSE(ParseFromString("}"));
}

// 3) Comment at EOF with NO trailing newline: ParseToken returns EOF (not COMMENT),
//    so Parse() sees START->EOF and fails. Covers that EOF edge.
TEST_F(NginxConfigParserTestFixture, CommentAtEOFWithoutNewlineIsInvalid) {
    EXPECT_FALSE(ParseFromString("# just a comment without newline"));
}

// 4) Comment terminated by CR (not LF): exercises TOKEN_TYPE_COMMENT path on '\r'
TEST_F(NginxConfigParserTestFixture, CommentTerminatedByCR) {
    EXPECT_TRUE(ParseFromString("# c\rfile;"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[0], "file");
}

// 5) END_BLOCK after END_BLOCK (i.e., close an already-closed nested block):
//    valid sequence when blocks match, but then an extra '}' should fail—hits the
//    END_BLOCK guard where last_token_type is END_BLOCK for the first, then fails at root.
TEST_F(NginxConfigParserTestFixture, ExtraCloseAfterProperBlock) {
    EXPECT_FALSE(ParseFromString("server { listen 80; } }"));
}

// 6) EOF immediately after START_BLOCK (unclosed empty block): distinct from your existing
//    "UnclosedBlock" (which has a statement). This specifically covers last_token_type == START_BLOCK then EOF error.
TEST_F(NginxConfigParserTestFixture, EOFRightAfterStartBlockIsInvalid) {
    EXPECT_FALSE(ParseFromString("server {"));
}

// 7) File open failure path in Parse(const char*): exercise early return
TEST(NginxConfigParserOpenFile, NonexistentFileFailsToOpen) {
    NginxConfigParser parser;
    NginxConfig out_config;
    EXPECT_FALSE(parser.Parse("this_file_should_not_exist_987654.conf", &out_config));
}

// ===================== ADD-ON TESTS FOR FULL COVERAGE =====================
TEST_F(NginxConfigParserTestFixture, StartTokenCoveredIndirectlyThroughParseError) {
    // Lone semicolon from START triggers error path that prints START via TokenTypeAsString.
    EXPECT_FALSE(ParseFromString(";"));
}

// A single-line comment with NO trailing newline via stream: ParseToken()
// returns EOF from COMMENT state; Parse() sees START->EOF and fails.
TEST_F(NginxConfigParserTestFixture, CommentToEOFNoNewlineViaStream) {
    std::stringstream ss;
    ss << "# whole file is comment without newline";
    EXPECT_FALSE(parser.Parse(&ss, &out_config));
}

// Token immediately followed by '}' (no whitespace) stresses TOKEN_STATE_TOKEN_TYPE_NORMAL
// delimiter handling where '}' is ungot and returned later as END_BLOCK.
TEST_F(NginxConfigParserTestFixture, NormalTokenFollowedByRightBrace) {
    EXPECT_TRUE(ParseFromString("server { listen 80;}"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    ASSERT_NE(out_config.statements_[0]->child_block_, nullptr);
    ASSERT_EQ(out_config.statements_[0]->child_block_->statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->child_block_->statements_[0]->tokens_[0], "listen");
    EXPECT_EQ(out_config.statements_[0]->child_block_->statements_[0]->tokens_[1], "80");
}

// Empty block immediately followed by a root-level statement ensures END_BLOCK
// path (with last_token_type==END_BLOCK) continues parsing correctly.
TEST_F(NginxConfigParserTestFixture, EmptyBlockThenRootStatement) {
    EXPECT_TRUE(ParseFromString("server {} user nginx;"));
    ASSERT_EQ(out_config.statements_.size(), 2);
    EXPECT_EQ(out_config.statements_[0]->tokens_[0], "server");
    EXPECT_EQ(out_config.statements_[1]->tokens_[0], "user");
    EXPECT_EQ(out_config.statements_[1]->tokens_[1], "nginx");
}

// Two consecutive semicolons: first ends a statement, second appears with
// last_token_type == STATEMENT_END -> should fail.
TEST_F(NginxConfigParserTestFixture, DoubleSemicolonIsInvalid) {
    EXPECT_FALSE(ParseFromString("foo;;"));
}

// Semicolon as the first thing inside a block: invalid because there's no
// preceding NORMAL token to terminate.
TEST_F(NginxConfigParserTestFixture, SemicolonAsFirstTokenInBlockIsInvalid) {
    EXPECT_FALSE(ParseFromString("server { ; }"));
}

/*********** Post-'}' tokenization with/without whitespace ***********/

// No space between '}' and next root-level token should still parse.
TEST_F(NginxConfigParserTestFixture, EmptyBlockThenRootStatement_NoSpace) {
    EXPECT_TRUE(ParseFromString("server {}user nginx;"));
    ASSERT_EQ(out_config.statements_.size(), 2);
    EXPECT_EQ(out_config.statements_[0]->tokens_[0], "server");
    EXPECT_EQ(out_config.statements_[1]->tokens_[0], "user");
    EXPECT_EQ(out_config.statements_[1]->tokens_[1], "nginx");
}

/*********** Quotes: mixed, escapes, and preservation rules ***********/

// Single-quoted token containing double quotes (no special handling needed).
TEST_F(NginxConfigParserTestFixture, SingleQuotedContainsDoubleQuotes) {
    EXPECT_TRUE(ParseFromString("file 'he said \"hi\"';"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "'he said \"hi\"'");
}

// Double-quoted token with escaped single quotes: backslash should be
// consumed (since \" and \' and \\ are recognized in double-quote mode).
TEST_F(NginxConfigParserTestFixture, DoubleQuotedEscapedSingleQuote) {
    EXPECT_TRUE(ParseFromString("file \"it\\'s ok\";"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "\"it's ok\"");
}

// Single-quoted token with backslashes for paths; only \\ is a recognized
// escape in single-quote mode (adds a single backslash).
TEST_F(NginxConfigParserTestFixture, SingleQuotedBackslashesForPath) {
    EXPECT_TRUE(ParseFromString("file 'C\\\\path\\\\to\\\\file';"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "'C\\path\\to\\file'");
}

// Unknown escape in single quotes: backslash should be preserved literally.
TEST_F(NginxConfigParserTestFixture, SingleQuotedUnknownEscapePreserved) {
    EXPECT_TRUE(ParseFromString("file 'a\\x';"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "'a\\x'");
}

// Braces/semicolons inside single quotes must not affect parsing.
TEST_F(NginxConfigParserTestFixture, SingleQuotedBracesAndSemicolonLiteral) {
    EXPECT_TRUE(ParseFromString("file '{not_a_block; }';"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->tokens_[1], "'{not_a_block; }'");
}

/*********** Deeper nesting & sibling structure ***********/

// Three-level nesting with siblings at different levels.
TEST_F(NginxConfigParserTestFixture, TripleNestedBlocksWithSiblings) {
    const char* cfg =
        "a {\n"
        "  b {\n"
        "    c { }\n"
        "    d 1;\n"
        "  }\n"
        "  e 2;\n"
        "}\n";
    EXPECT_TRUE(ParseFromString(cfg));

    ASSERT_EQ(out_config.statements_.size(), 1);
    auto* a = out_config.statements_[0].get();
    ASSERT_NE(a->child_block_, nullptr);
    ASSERT_EQ(a->child_block_->statements_.size(), 2);

    auto* b_stmt = a->child_block_->statements_[0].get();
    ASSERT_NE(b_stmt->child_block_, nullptr);
    ASSERT_EQ(b_stmt->child_block_->statements_.size(), 2); // c{}, d 1;

    auto* c_stmt = b_stmt->child_block_->statements_[0].get();
    ASSERT_NE(c_stmt->child_block_, nullptr);
    EXPECT_EQ(c_stmt->child_block_->statements_.size(), 0); // empty c{}

    auto* d_stmt = b_stmt->child_block_->statements_[1].get();
    ASSERT_TRUE(d_stmt->child_block_ == nullptr);
    ASSERT_EQ(d_stmt->tokens_.size(), 2);
    EXPECT_EQ(d_stmt->tokens_[0], "d");
    EXPECT_EQ(d_stmt->tokens_[1], "1");

    auto* e_stmt = a->child_block_->statements_[1].get();
    ASSERT_TRUE(e_stmt->child_block_ == nullptr);
    ASSERT_EQ(e_stmt->tokens_.size(), 2);
    EXPECT_EQ(e_stmt->tokens_[0], "e");
    EXPECT_EQ(e_stmt->tokens_[1], "2");
}

/*********** Comments & CRLF handling ***********/

// Inline comment after a statement should be ignored up to EOL.
TEST_F(NginxConfigParserTestFixture, CommentAfterStatementSameLine) {
    EXPECT_TRUE(ParseFromString("file list; # ignore me\nuser nginx;"));
    ASSERT_EQ(out_config.statements_.size(), 2);
    EXPECT_EQ(out_config.statements_[0]->tokens_[0], "file");
    EXPECT_EQ(out_config.statements_[1]->tokens_[0], "user");
}

// Windows-style CRLF newlines should be treated as whitespace separators.
TEST_F(NginxConfigParserTestFixture, CRLFLineEndings) {
    EXPECT_TRUE(ParseFromString("file list;\r\nuser nginx;\r\n"));
    ASSERT_EQ(out_config.statements_.size(), 2);
    EXPECT_EQ(out_config.statements_[0]->tokens_[0], "file");
    EXPECT_EQ(out_config.statements_[1]->tokens_[0], "user");
}

/*********** TokenTypeAsString direct coverage (public via test include) ***********/

TEST(NginxConfigParserTokenNames, TokenTypeAsStringMappings) {
    NginxConfigParser p;
    EXPECT_STREQ(p.TokenTypeAsString(NginxConfigParser::TOKEN_TYPE_START), "TOKEN_TYPE_START");
    EXPECT_STREQ(p.TokenTypeAsString(NginxConfigParser::TOKEN_TYPE_NORMAL), "TOKEN_TYPE_NORMAL");
    EXPECT_STREQ(p.TokenTypeAsString(NginxConfigParser::TOKEN_TYPE_START_BLOCK), "TOKEN_TYPE_START_BLOCK");
    EXPECT_STREQ(p.TokenTypeAsString(NginxConfigParser::TOKEN_TYPE_END_BLOCK), "TOKEN_TYPE_END_BLOCK");
    EXPECT_STREQ(p.TokenTypeAsString(NginxConfigParser::TOKEN_TYPE_COMMENT), "TOKEN_TYPE_COMMENT");
    EXPECT_STREQ(p.TokenTypeAsString(NginxConfigParser::TOKEN_TYPE_STATEMENT_END), "TOKEN_TYPE_STATEMENT_END");
    EXPECT_STREQ(p.TokenTypeAsString(NginxConfigParser::TOKEN_TYPE_EOF), "TOKEN_TYPE_EOF");
    EXPECT_STREQ(p.TokenTypeAsString(NginxConfigParser::TOKEN_TYPE_ERROR), "TOKEN_TYPE_ERROR");
}

TEST(NginxConfigParserTokenNames, UnknownTokenTypeFallback) {
    NginxConfigParser p;
    EXPECT_STREQ(p.TokenTypeAsString(static_cast<NginxConfigParser::TokenType>(999)),
                 "Unknown token type");
}

// Token immediately followed by '{' with no space is allowed (block opener).
TEST_F(NginxConfigParserTestFixture, NoSpaceBeforeStartBlock) {
    EXPECT_TRUE(ParseFromString("server{listen 80;}"));
    ASSERT_EQ(out_config.statements_.size(), 1);
    ASSERT_NE(out_config.statements_[0]->child_block_, nullptr);
    ASSERT_EQ(out_config.statements_[0]->child_block_->statements_.size(), 1);
    EXPECT_EQ(out_config.statements_[0]->child_block_->statements_[0]->tokens_[0], "listen");
    EXPECT_EQ(out_config.statements_[0]->child_block_->statements_[0]->tokens_[1], "80");
}
