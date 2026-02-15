#ifndef SERVER_CONFIG_H
#define SERVER_CONFIG_H

#include <map>
#include <string>
#include <vector>

class NginxConfig;

struct HandlerSpec {
    std::string name;
    std::string path;
    std::string type;
    std::map<std::string, std::string> options;
};

struct ServerConfig {
    int port = 0;
    std::vector<HandlerSpec> handlers;

    static bool FromTokenizedConfig(const NginxConfig& tokenized_config,
                                    ServerConfig* out,
                                    std::string* error);
};

#endif // SERVER_CONFIG_H
