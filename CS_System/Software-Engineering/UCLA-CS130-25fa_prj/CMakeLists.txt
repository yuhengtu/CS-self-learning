cmake_minimum_required(VERSION 3.10.0)
project(semiconductor-savants)

# Make sure developers do not run cmake in the main project directory, to keep
# build artifacts from becoming clutter
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed.
    Please make a new directory (called a build directory) and run CMake from there.
    You may need to remove CMakeCache.txt." )
endif()

# Turn on debug builds if we are building in a devel shell
if (CMAKE_BUILD_TYPE STREQUAL "" AND "$ENV{DEVEL_SHELL}" STREQUAL "1")
    message(STATUS "Setting debug build type by default in devel shell")
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Output binaries to a sub directory "bin"
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable cmake testing
include(CTest)
enable_testing()

# Enable GoogleTest
include(GoogleTest)
add_subdirectory(/usr/src/googletest googletest)

# Enable Boost
# Use static libraries so binaries can be deployed without a full boost install
set(Boost_USE_STATIC_LIBS ON)
# Add log and log_setup for Boost.Log
find_package(Boost 1.75 REQUIRED COMPONENTS system log_setup log thread filesystem json)
find_package(OpenSSL REQUIRED)
message(STATUS "Boost version: ${Boost_VERSION}")

include_directories(include)

# ===================================================================
# 1. LIBRARY DEFINITIONS
# Define all our libraries first.
# ===================================================================

add_library(logger_lib src/logger.cc)
target_link_libraries(logger_lib
    PUBLIC
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

add_library(request_lib src/request.cc)

add_library(response_lib src/response.cc)

add_library(response_builder_lib src/response_builder.cc)

add_library(config_parser_lib
    src/config_parser.cc
    src/server_config.cc
    src/handler_types.cc
)

add_library(request_parser_lib src/request_parser.cc)

add_library(request_handler_lib
    src/echo_request_handler.cc
    src/dispatcher.cc
    src/handler_types.cc
    src/handler_registry.cc
    src/echo_handler_factory.cc     
    src/static_handler_factory.cc
    src/static_request_handler.cc
    src/not_found_request_handler.cc
    src/not_found_handler_factory.cc
    src/crud_handler_factory.cc
    src/crud_request_handler.cc
    src/health_request_handler.cc
    src/health_handler_factory.cc
    src/sleep_request_handler.cc
    src/sleep_handler_factory.cc
    src/link_manage_request_handler.cc
    src/link_manage_handler_factory.cc
    src/link_redirect_request_handler.cc
    src/link_redirect_handler_factory.cc
    src/analytics_request_handler.cc
    src/analytics_handler_factory.cc
)

add_library(crud_manager_lib
    src/crud_manager.cc
)

# URL shortener: filesystem-backed manager + serialization helpers
add_library(link_manager_lib
    src/link_manager.cc
    src/link_record_serialization.cc
    src/link_manager_provider.cc
)


add_library(session_lib src/session.cc)

add_library(server_lib src/server.cc)


# ===================================================================
# 2. EXECUTABLE DEFINITIONS
# Define all executables (the server + tests)
# ===================================================================

add_executable(webserver src/_main.cc)

add_executable(config_parser_test tests/config/config_parser_test.cc)
add_executable(request_parser_test tests/request_parser_test.cc)
add_executable(server_config_test tests/server_config_test.cc)
add_executable(dispatcher_test tests/dispatcher_test.cc)
add_executable(response_test tests/response_test.cc)
add_executable(static_request_handler_test tests/static_request_handler_test.cc)
add_executable(echo_request_handler_test tests/echo_request_handler_test.cc)
add_executable(echo_handler_factory_test tests/echo_handler_factory_test.cc)
add_executable(static_handler_factory_test tests/static_handler_factory_test.cc)
add_executable(handler_registry_test tests/handler_registry_test.cc)
add_executable(logger_test tests/logger_test.cc)
add_executable(response_builder_test tests/response_builder_test.cc)
add_executable(not_found_handler_test tests/not_found_handler_test.cc)
add_executable(session_test tests/session_test.cc)
add_executable(server_test tests/server_test.cc)
add_executable(crud_manager_test tests/crud_manager_test.cc)
add_executable(crud_request_handler_test tests/crud_request_handler_test.cc)
add_executable(crud_handler_factory_test tests/crud_handler_factory_test.cc)
add_executable(health_handler_factory_test tests/health_handler_factory_test.cc)
add_executable(health_request_handler_test tests/health_request_handler_test.cc)
add_executable(sleep_handler_factory_test tests/sleep_handler_factory_test.cc)
add_executable(sleep_request_handler_test tests/sleep_request_handler_test.cc)
add_executable(link_manager_fake_test tests/link_manager_fake_test.cc tests/fakes/link_manager_fake.cc)
add_executable(link_manager_fs_test tests/link_manager_fs_test.cc)
add_executable(link_record_serialization_test tests/link_record_serialization_test.cc)
add_executable(link_manage_request_handler_test
    tests/link_manage_request_handler_test.cc
    tests/fakes/link_manager_fake.cc
)
add_executable(analytics_request_handler_test
    tests/analytics_request_handler_test.cc
    tests/fakes/link_manager_fake.cc
)
add_executable(link_manager_provider_test tests/link_manager_provider_test.cc)
add_executable(link_redirect_handler_factory_test tests/link_redirect_handler_factory_test.cc)
add_executable(link_redirect_request_handler_test 
    tests/link_redirect_request_handler_test.cc
    tests/fakes/link_manager_fake.cc
)
# ===================================================================
# 3. TARGET LINKING
# Link all our targets now that they are defined.
# ===================================================================

# --- Library Links ---

target_link_libraries(server_lib
    PRIVATE
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(session_lib PUBLIC request_lib)
target_link_libraries(request_parser_lib PUBLIC request_lib)
target_link_libraries(response_builder_lib PUBLIC response_lib)
target_link_libraries(request_handler_lib 
    PUBLIC 
    request_lib 
    response_builder_lib
    # Static handler needs filesystem
    Boost::filesystem 
    crud_manager_lib
    # URL shortener deps
    link_manager_lib
    Boost::json
    OpenSSL::Crypto
    # Logger used by handler registry and others
    logger_lib
)

target_link_libraries(crud_manager_lib
    PUBLIC
    Boost::filesystem
)

target_link_libraries(link_manager_lib
    PUBLIC
    Boost::filesystem
    Boost::json
)

# --- Server Executable Link ---

target_link_libraries(webserver 
    PRIVATE
    Boost::system 
    Boost::log_setup 
    Boost::log 
    server_lib 
    session_lib 
    request_parser_lib 
    response_lib
    response_builder_lib 
    request_handler_lib 
    config_parser_lib 
    logger_lib
)

# --- Test Executable Links ---

target_link_libraries(config_parser_test 
    PRIVATE
    config_parser_lib 
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(request_parser_test 
    PRIVATE
    request_parser_lib 
    gtest_main 
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(server_config_test
    PRIVATE
    config_parser_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(dispatcher_test 
    PRIVATE
    request_handler_lib
    response_lib
    response_builder_lib
    gtest_main 
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(response_test 
    PRIVATE
    response_lib 
    gtest_main 
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(static_request_handler_test
    PRIVATE
    request_handler_lib
    response_lib
    response_builder_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(echo_request_handler_test
    PRIVATE
    request_handler_lib
    response_lib
    response_builder_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(echo_handler_factory_test
    PRIVATE
    request_handler_lib
    response_lib
    response_builder_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(static_handler_factory_test
    PRIVATE
    request_handler_lib
    response_lib
    response_builder_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(handler_registry_test
    PRIVATE
    request_handler_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(logger_test
    PRIVATE
    logger_lib
    gtest_main
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(response_builder_test
    PRIVATE
    response_builder_lib
    response_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

# Add links for the new 404 handler test
target_link_libraries(not_found_handler_test
    PRIVATE
    request_handler_lib
    response_builder_lib
    response_lib
    request_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(session_test
    PRIVATE
    server_lib
    session_lib
    request_handler_lib
    config_parser_lib
    request_parser_lib
    response_builder_lib
    response_lib 
    request_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(server_test
    PRIVATE
    server_lib
    session_lib
    request_handler_lib
    config_parser_lib
    request_parser_lib
    response_builder_lib
    response_lib
    request_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(crud_manager_test
    PRIVATE
    crud_manager_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
    Boost::filesystem
)

target_link_libraries(crud_request_handler_test
    PRIVATE
    request_handler_lib
    response_builder_lib
    request_lib
    response_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(crud_handler_factory_test
    PRIVATE
    request_handler_lib
    response_lib
    response_builder_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

# === HEALTH HANDLER AND FACTORY ===
target_link_libraries(health_request_handler_test
    PRIVATE
    request_handler_lib
    response_builder_lib
    request_lib
    response_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(health_handler_factory_test
    PRIVATE
    request_handler_lib
    response_lib
    response_builder_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)
# === END HEALTH HANDLER AND FACTORY ===

target_link_libraries(sleep_handler_factory_test
    PRIVATE
    request_handler_lib
    response_lib
    response_builder_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(sleep_request_handler_test
    PRIVATE
    request_handler_lib
    response_builder_lib
    request_lib
    response_lib
    gtest_main
    logger_lib
    Boost::log_setup
    Boost::log
    Boost::thread
    Boost::system
)

target_link_libraries(link_manager_fake_test
    PRIVATE
    gtest_main
)

target_link_libraries(link_manager_fs_test
    PRIVATE
    link_manager_lib
    gtest_main
)

target_link_libraries(link_record_serialization_test
    PRIVATE
    link_manager_lib
    gtest_main
)

target_link_libraries(link_manage_request_handler_test
    PRIVATE
    request_handler_lib
    response_lib
    response_builder_lib
    gtest_main
    logger_lib
    Boost::json
)

target_link_libraries(analytics_request_handler_test
    PRIVATE
    request_handler_lib
    response_lib
    response_builder_lib
    gtest_main
    logger_lib
    Boost::json
)

target_link_libraries(link_redirect_handler_factory_test
    PRIVATE
    request_handler_lib
    response_lib
    response_builder_lib
    gtest_main
    logger_lib
    Boost::json
)

target_link_libraries(link_redirect_request_handler_test
    PRIVATE
    request_handler_lib
    response_lib
    response_builder_lib
    gtest_main
    logger_lib
    Boost::json
)

target_link_libraries(link_manager_provider_test
    PRIVATE
    link_manager_lib
    gtest_main
)
# ===================================================================
# 4. CTEST DISCOVERY
# ===================================================================

gtest_discover_tests(config_parser_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/config)
gtest_discover_tests(request_parser_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(server_config_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(dispatcher_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(response_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(static_request_handler_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(echo_request_handler_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(echo_handler_factory_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(static_handler_factory_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(handler_registry_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(logger_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(response_builder_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(not_found_handler_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(session_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(server_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(crud_manager_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(crud_request_handler_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(crud_handler_factory_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(health_request_handler_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(health_handler_factory_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(sleep_handler_factory_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(sleep_request_handler_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(link_manager_fake_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(link_manager_fs_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(link_record_serialization_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(link_manage_request_handler_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(analytics_request_handler_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(link_manager_provider_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(link_redirect_request_handler_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
gtest_discover_tests(link_redirect_handler_factory_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)

# Integration tests
add_test(NAME integration_tests
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/run_integration_tests.sh
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# In /build run 'cmake -DCMAKE_BUILD_TYPE=Coverage ..' and then 'make coverage' to generate report
include(cmake/CodeCoverageReportConfig.cmake)
generate_coverage_report(
    TARGETS 
        server_lib 
        session_lib 
        config_parser_lib 
        request_lib
        request_parser_lib 
        response_lib
        response_builder_lib 
        request_handler_lib
        logger_lib
        crud_manager_lib
        link_manager_lib
        webserver 
    TESTS 
        config_parser_test 
        request_parser_test 
        server_config_test
        dispatcher_test
        response_test
        response_builder_test
        echo_handler_factory_test
        static_handler_factory_test
        handler_registry_test
        static_request_handler_test
        echo_request_handler_test
        logger_test
        not_found_handler_test
        session_test
        server_test
        crud_manager_test
        crud_request_handler_test
        crud_handler_factory_test
        health_handler_factory_test
        health_request_handler_test
        sleep_handler_factory_test
        sleep_request_handler_test
        link_manager_fake_test
        link_manager_fs_test
        link_record_serialization_test
        link_manage_request_handler_test
        analytics_request_handler_test
        link_manager_provider_test
        link_redirect_handler_factory_test
        link_redirect_request_handler_test
)
