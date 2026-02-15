#include "logger.h"

#include <boost/log/utility/setup/console.hpp>
#include <boost/asio/signal_set.hpp>
#include <sstream>
#include <iostream>
#include <memory>

namespace logging = boost::log;
namespace keywords = boost::log::keywords;
using namespace boost::log::trivial;

// Init singleton
std::unique_ptr<Logger> Logger::instance_m = nullptr;

Logger::Logger() {
    init();
    logging::add_common_attributes();
    logging::add_console_log(std::cout, keywords::format = "[%TimeStamp%]:[%ThreadID%]:%Message%");
}

// Accessor returns a raw pointer, but ownership is managed internally
Logger& Logger::getInstance() {
    if (!instance_m) {
        instance_m.reset(new Logger());
    }
    return *instance_m;
}

void Logger::init() {
    const int k10MB = 10 * 1024 * 1024;
    logging::add_file_log(
        keywords::file_name = "../log/LOG_%N.log",
        keywords::rotation_size = k10MB,
        keywords::time_based_rotation = boost::log::sinks::file::rotation_at_time_point(0, 0, 0),
        keywords::format = "[%TimeStamp%]:[%ThreadID%]:%Message%",
        keywords::auto_flush = true
    );
}

void Logger::logServerInitialization() {
    BOOST_LOG_SEV(logger_m, trace) << "Trace: Server started";
}

void Logger::logTrace(const std::string& message) {
    BOOST_LOG_SEV(logger_m, trace) << "Trace: " << message;
}

void Logger::logDebug(const std::string& message) {
    BOOST_LOG_SEV(logger_m, debug) << "Debug: " << message;
}

void Logger::logWarning(const std::string& message) {
    BOOST_LOG_SEV(logger_m, warning) << "Warning: " << message;
}

void Logger::logError(const std::string& message) {
    BOOST_LOG_SEV(logger_m, error) << "Error: " << message;
}

void Logger::logTraceHTTPRequest(const request& http_request) {
    std::stringstream stream;
    stream << "Trace: Incoming HTTP Request:\n" << http_request.raw;
    BOOST_LOG_SEV(logger_m, trace) << stream.str();
}

void Logger::logSignal() {
    BOOST_LOG_SEV(logger_m, warning) << "Warning: Shutting down the server...";
}

void Logger::logConnectionDetails(tcp::socket& sock){
    std::stringstream stream;
    stream << "Trace: Incoming connection from IP Address " << sock.remote_endpoint().address().to_string()
    << " and port " << sock.remote_endpoint().port();
    BOOST_LOG_SEV(logger_m, trace) << stream.str();
}
