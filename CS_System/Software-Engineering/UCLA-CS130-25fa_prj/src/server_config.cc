#include "server_config.h"

#include <set>
#include <sstream>
#include <string_view>

#include "config_parser.h"
#include "handler_types.h"

// Note: written with GenAI assistance

namespace {

constexpr std::string_view kListenDirective = "listen";
constexpr std::string_view kHandlerDirective = "handler";
constexpr std::string_view kPathDirective = "path";
constexpr std::string_view kTypeDirective = "type";
constexpr int kExpectedDirectiveTokens = 2;  // keyword + value
constexpr std::string_view kLocationDirective = "location";
constexpr std::string_view kServerDirective = "server";

std::string BuildError(const std::string& message) {
    return "ServerConfigBuilder: " + message;
}

bool EnsurePathValid(const std::string& path, std::string* error) {
    if (path.empty() || path[0] != '/') {
        if (error) {
            *error = BuildError("handler path must start with '/'");
        }
        return false;
    }
    return true;
}

}  // namespace

bool ServerConfig::FromTokenizedConfig(const NginxConfig& tokenized_config,
                                       ServerConfig* out,
                                       std::string* error) {
    if (!out) {
        if (error) {
            *error = BuildError("output pointer is null");
        }
        return false;
    }

    ServerConfig result;
    bool port_set = false;
    std::set<std::string> paths;

    const NginxConfig* server_scope = nullptr;
    for (const auto& statement : tokenized_config.statements_) {
        if (statement->tokens_.empty()) {
            continue;
        }
        const std::string& directive = statement->tokens_[0];
        if (directive == kServerDirective && statement->child_block_) {
            if (server_scope != nullptr) {
                if (error) {
                    *error = BuildError("multiple top-level server blocks are not supported");
                }
                return false;
            }
            server_scope = statement->child_block_.get();
        }
    }

    if (!server_scope) {
        if (error) {
            *error = BuildError("no top-level server block found");
        }
        return false;
    }

    for (const auto& statement : server_scope->statements_) {
        if (statement->tokens_.empty()) {
            continue;
        }
        const std::string& directive = statement->tokens_[0];

        if (directive == kListenDirective) {
            if (statement->tokens_.size() != kExpectedDirectiveTokens) {
                if (error) {
                    *error = BuildError("listen directive expects exactly one port value");
                }
                return false;
            }
            try {
                result.port = std::stoi(statement->tokens_[1]);
                port_set = true;
            } catch (const std::exception&) {
                if (error) {
                    *error = BuildError("listen directive has invalid port value");
                }
                return false;
            }

        } else if (directive == kLocationDirective) {
            if (statement->tokens_.size() != kExpectedDirectiveTokens) {
                if (error) {
                    *error = BuildError("handler directive expects exactly one identifier");
                }
                return false;
            }
            if (!statement->child_block_) {
                if (error) {
                    *error = BuildError("handler block missing body");
                }
                return false;
            }

            HandlerSpec spec;
            spec.path = statement->tokens_[1];
            spec.name = spec.path;

            for (const auto& child : statement->child_block_->statements_) {
                if (child->tokens_.empty()) {
                    continue;
                }
                const std::string& key = child->tokens_[0];

                if (key == kHandlerDirective) {
                    if (child->tokens_.size() != kExpectedDirectiveTokens) {
                        if (error) {
                            *error = BuildError("handler type expects exactly one value");
                        }
                        return false;
                    }
                    spec.type = child->tokens_[1];
                } else if (child->tokens_.size() >= 2) {
                    spec.options[key] = child->tokens_[1];
                }
            }

            if (!EnsurePathValid(spec.path, error)) {
                return false;
            }
            if (spec.type.empty()) {
                if (error) {
                    *error = BuildError("handler type must be specified");
                }
                return false;
            }
            if (!paths.insert(spec.path).second) {
                if (error) {
                    *error = BuildError("duplicate handler path: " + spec.path);
                }
                return false;
            }

            result.handlers.push_back(std::move(spec));

        } else {
            continue;
        }
    }

    if (!port_set) {
        if (error) {
            *error = BuildError("no listen directive found");
        }
        return false;
    }

    *out = std::move(result);
    return true;
}
