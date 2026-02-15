#include "gtest/gtest.h"

#include <sstream>

#include "config_parser.h"
#include "server_config.h"
#include "handler_types.h"

namespace {

bool ParseConfig(const std::string& text, NginxConfig* out) {
    NginxConfigParser parser;
    std::istringstream stream(text);
    return parser.Parse(&stream, out);
}

}  // namespace

TEST(ServerConfigBuilderTest, ParsesPortAndHandler) {
    const std::string text = R"(
        server {
            listen 9000;
            location /echo {
                handler echo;
            }
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    ASSERT_TRUE(ServerConfig::FromTokenizedConfig(tokenized, &config, nullptr));

    EXPECT_EQ(config.port, 9000);
    ASSERT_EQ(config.handlers.size(), 1u);
    EXPECT_EQ(config.handlers[0].name, "/echo");
    EXPECT_EQ(config.handlers[0].path, "/echo");
    EXPECT_EQ(config.handlers[0].type, handler_types::ECHO_HANDLER);
}

TEST(ServerConfigBuilderTest, NoHandlers_ParserReturnsEmptyHandlers_DispatcherInjects404Later) {
    const std::string text = R"(
        server {
            listen 8080;
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    ASSERT_TRUE(ServerConfig::FromTokenizedConfig(tokenized, &config, nullptr));

    EXPECT_EQ(config.port, 8080);
    // New expectation: parser does not invent any handlers.
    EXPECT_TRUE(config.handlers.empty());
}

TEST(ServerConfigBuilderTest, RejectsDuplicatePaths) {
    const std::string text = R"(
        server {
            listen 8081;
            location /dup {
                handler echo;
            }
            location /dup {
                handler echo;
            }
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;
    EXPECT_FALSE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_FALSE(error.empty());
}

TEST(ServerConfigBuilderTest, RejectsMissingPath) {
    // location with no path argument: "location { ... }"
    const std::string text = R"(
        server {
            listen 8000;
            location {
                handler echo;
            }
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;
    EXPECT_FALSE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_FALSE(error.empty());
}

TEST(ServerConfigBuilderTest, RejectsListenWithoutValue) {
    const std::string text = R"(
        server {
            listen;
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;
    EXPECT_FALSE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_FALSE(error.empty());
}

TEST(ServerConfigBuilderTest, RejectsListenWithExtraTokens) {
    const std::string text = R"(
        server {
            listen 8080 extra;
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;
    EXPECT_FALSE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_FALSE(error.empty());
}

TEST(ServerConfigBuilderTest, RejectsWhenNoListenDirective) {
    const std::string text = R"(
        server {
            location /echo {
                handler echo;
            }
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;
    EXPECT_FALSE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_FALSE(error.empty());
}

TEST(ServerConfigBuilderTest, RejectsNonNumericListenPort) {
    const std::string text = R"(
        server {
            listen abc;
            location /echo {
                handler echo;
            }
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;
    EXPECT_FALSE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_FALSE(error.empty());
}

// Implementation accepts any integer from stoi() (no range checks here).
// We verify it simply parses whatever stoi returns.

TEST(ServerConfigBuilderTest, AcceptsNegativePortValue_NoRangeValidationHere) {
    const std::string text = R"(
        server {
            listen -1;
            location /echo {
                handler echo;
            }
        }
    )";
    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;  // unused on success
    ASSERT_TRUE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_EQ(config.port, -1);
    EXPECT_TRUE(error.empty());  // remains empty because success doesn't write it
}

TEST(ServerConfigBuilderTest, AcceptsLargePortValue_NoRangeValidationHere) {
    const std::string text = R"(
        server {
            listen 70000;
            location /echo {
                handler echo;
            }
        }
    )";
    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    // Pass nullptr for error to document that success doesn't touch it anyway.
    ASSERT_TRUE(ServerConfig::FromTokenizedConfig(tokenized, &config, nullptr));
    EXPECT_EQ(config.port, 70000);
}

TEST(ServerConfigBuilderTest, RejectsHandlerMissingType) {
    // handler directive present but missing type value: "handler;"
    const std::string text = R"(
        server {
            listen 8080;
            location /x {
                handler;
            }
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;
    EXPECT_FALSE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_FALSE(error.empty());
}

TEST(ServerConfigBuilderTest, RejectsHandlerMissingInLocation) {
    // location has no handler directive at all -> spec.type remains empty.
    const std::string text = R"(
        server {
            listen 8080;
            location /x {
                # no handler directive
            }
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;
    EXPECT_FALSE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_FALSE(error.empty());
}

TEST(ServerConfigBuilderTest, RejectsPathWithoutLeadingSlash) {
    const std::string text = R"(
        server {
            listen 8080;
            location echo {   # missing leading '/'
                handler echo;
            }
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;
    EXPECT_FALSE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_FALSE(error.empty());
}

TEST(ServerConfigBuilderTest, ParsesMultipleHandlersAndOptions) {
    // Also ensure default echo is NOT added when at least one handler exists
    const std::string text = R"(
        server {
            listen 9090;
            location /echo {
                handler echo;
            }
            location /static {
                handler static;
                root /var/www;  # option captured into HandlerSpec.options
            }
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;
    ASSERT_TRUE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_TRUE(error.empty());

    EXPECT_EQ(config.port, 9090);
    ASSERT_EQ(config.handlers.size(), 2u);

    // Echo
    EXPECT_EQ(config.handlers[0].name, "/echo");
    EXPECT_EQ(config.handlers[0].path, "/echo");
    EXPECT_EQ(config.handlers[0].type, handler_types::ECHO_HANDLER);
    EXPECT_TRUE(config.handlers[0].options.empty());

    // Static with option
    EXPECT_EQ(config.handlers[1].name, "/static");
    EXPECT_EQ(config.handlers[1].path, "/static");
    EXPECT_EQ(config.handlers[1].type, handler_types::STATIC_HANDLER);
    ASSERT_TRUE(config.handlers[1].options.count("root") > 0);
    EXPECT_EQ(config.handlers[1].options.at("root"), "/var/www");
}

TEST(ServerConfigBuilderTest, SuccessDoesNotWriteErrorOutParam) {
    const std::string text = R"(
        server {
            listen 8082;
            location /echo {
                handler echo;
            }
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;

    // Seed a non-empty string and confirm it is NOT modified on success.
    std::string error = "pre-seeded";
    ASSERT_TRUE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));

    EXPECT_EQ(config.port, 8082);
    ASSERT_EQ(config.handlers.size(), 1u);

    // Implementation only writes to `error` on failure; on success it leaves it untouched.
    EXPECT_EQ(error, "pre-seeded");
}

TEST(ServerConfigBuilderTest, RejectsNullOutPointer_AndWritesError) {
    const std::string text = R"(
        server {
            listen 8080;
            location /echo {
                handler echo;
            }
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    std::string error = "";
    EXPECT_FALSE(ServerConfig::FromTokenizedConfig(tokenized, /*out=*/nullptr, &error));  // null out*
    EXPECT_FALSE(error.empty());  // should be populated: "output pointer is null"
}

TEST(ServerConfigBuilderTest, RejectsLocationMissingBodyBlock) {
    // 'location /x;' present but with no '{ ... }' child block
    const std::string text = R"(
        server {
            listen 8080;
            location /x;
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;
    EXPECT_FALSE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_FALSE(error.empty());
}

TEST(ServerConfigBuilderTest, RejectsHandlerWithExtraIdentifierTokens) {
    // The builder expects exactly one identifier after 'handler' inside a location
    const std::string text = R"(
        server {
            listen 8080;
            location /echo {
                handler echo extra;
            }
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;
    EXPECT_FALSE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_FALSE(error.empty());
}

TEST(ServerConfigBuilderTest, IgnoresUnknownDirectivesAndStillBuilds) {
    // Unknown directive inside server block should be ignored; result still succeeds
    const std::string text = R"(
        server {
            listen 9091;
            # unknown directive (ignored by builder)
            frobnicate on;

            location /echo {
                handler echo;
            }
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;
    ASSERT_TRUE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(config.port, 9091);
    ASSERT_EQ(config.handlers.size(), 1u);
    EXPECT_EQ(config.handlers[0].path, "/echo");
    EXPECT_EQ(config.handlers[0].type, handler_types::ECHO_HANDLER);
}

TEST(ServerConfigBuilderTest, OptionTakesFirstValue_ExtraTokensIgnored) {
    // For non 'path'/'type' keys, tokens.size() >= 2 -> options[key] = tokens[1]
    // Extra tokens after the first value are ignored by the implementation.
    const std::string text = R"(
        server {
            listen 9092;
            location /static {
                handler static;
                root /var/www extra tokens are ignored;
            }
        }
    )";

    NginxConfig tokenized;
    ASSERT_TRUE(ParseConfig(text, &tokenized));

    ServerConfig config;
    std::string error;
    ASSERT_TRUE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
    EXPECT_TRUE(error.empty());
    ASSERT_EQ(config.handlers.size(), 1u);
    const auto& spec = config.handlers[0];
    ASSERT_TRUE(spec.options.count("root") > 0);
    EXPECT_EQ(spec.options.at("root"), "/var/www"); // only first value is kept
}

TEST(ServerConfigBuilderTest, TopLevelServerBlockErrorsPopulateErrorString) {
    // Case 1: Multiple top-level server blocks -> "multiple top-level server blocks are not supported"
    {
        const std::string text = R"(
            server {
                listen 8080;
            }
            server {
                listen 8081;
            }
        )";

        NginxConfig tokenized;
        ASSERT_TRUE(ParseConfig(text, &tokenized));

        ServerConfig config;
        std::string error;
        EXPECT_FALSE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
        EXPECT_FALSE(error.empty());
        EXPECT_NE(error.find("multiple top-level server blocks are not supported"),
                  std::string::npos);
    }

    // Case 2: No top-level server block at all -> "no top-level server block found"
    {
        const std::string text = R"(
            # No 'server { ... }' block here, just a top-level listen
            listen 8080;
        )";

        NginxConfig tokenized;
        ASSERT_TRUE(ParseConfig(text, &tokenized));

        ServerConfig config;
        std::string error;
        EXPECT_FALSE(ServerConfig::FromTokenizedConfig(tokenized, &config, &error));
        EXPECT_FALSE(error.empty());
        EXPECT_NE(error.find("no top-level server block found"), std::string::npos);
    }
}
