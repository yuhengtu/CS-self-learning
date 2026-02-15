#ifndef LOGGER_H
#define LOGGER_H

#include <boost/asio.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/signals2.hpp>
#include <memory>

#include "request.h"

using boost::asio::ip::tcp;

class Logger {
public:
    static Logger& getInstance();

    void init();

    void logServerInitialization();

    void logTrace(const std::string& message);
    void logDebug(const std::string& message);
    void logWarning(const std::string& message);
    void logError(const std::string& message);

    void logTraceHTTPRequest(const request& http_request);
    void logConnectionDetails(tcp::socket& sock);

    void logSignal();

private:
    Logger();
    static std::unique_ptr<Logger> instance_m;
    boost::log::sources::severity_logger<boost::log::trivial::severity_level> logger_m;
};

#endif  // LOGGER_H
