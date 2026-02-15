#ifndef HANDLER_TYPES_H
#define HANDLER_TYPES_H

#include <string>

namespace handler_types {
    extern const std::string ECHO_HANDLER;
    extern const std::string STATIC_HANDLER;
    extern const std::string NOT_FOUND_HANDLER;
    extern const std::string CRUD_HANDLER;
    extern const std::string HEALTH_HANDLER;
    extern const std::string SLEEP_HANDLER;
    // URL shortener management handler
    extern const std::string LINK_MANAGE_HANDLER;
    extern const std::string LINK_REDIRECT_HANDLER;
    extern const std::string ANALYTICS_HANDLER;
}

#endif // HANDLER_TYPES_H
