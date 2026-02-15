//
// async_tcp_echo_server.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2017 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <thread>
#include <vector>

#include "session.h"
#include "server.h"
#include "config_parser.h"
#include "server_config.h"
#include "logger.h"

using boost::asio::ip::tcp;

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: webserver <config_file>\n";
      return 1;
    }

    NginxConfigParser parser;
    NginxConfig config;
    if (!parser.Parse(argv[1], &config)) {
      std::cerr << "Failed to parse config file\n";
      return 1;
    }

    ServerConfig server_config;
    std::string error;
    if (!ServerConfig::FromTokenizedConfig(config, &server_config, &error)) {
      std::cerr << error << "\n";
      return 1;
    }

    boost::asio::io_service io_service;

    Server s(io_service, server_config);

    std::size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4;
    }

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back([&io_service]() {
            io_service.run();
        });
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
  }
  catch (std::exception& e)
  {
    Logger& log = Logger::getInstance();
    log.logError(std::string("main: Exception: ") + e.what() + "\n");
  }

  return 0;
}
