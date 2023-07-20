/*
 * Copyright 2023 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DaphneLogger.h"

#include <spdlog/spdlog.h>
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

#include <iostream>

/**
 * Available log levels taken from <spdlog/common.h> for your reference:
 *
 * trace = 0,
 * debug = 1,
 * info = 2,
 * warn = 3,
 * err = 4,
 * critical = 5,
 * off = 6,
 *
 * fallback_loggers takes {str:name, str:filename, int:level, str:pattern} initializers
 */
const std::vector<LogConfig> DaphneLogger::fallback_loggers = {
        {"default","", 6,"%^%L %H:%M:%S [%n]%$ %v"},
        {"compiler::cuda", "", 6, "%^[%n %L]:%$ %v" },
        {"runtime::cuda", "", 6, "%^[%n %L]:%$ %v" }
};

void DaphneLogger::createLoggers(const LogConfig& config) {

            auto logger = spdlog::get(config.name);
            if(not logger) {
                std::vector<spdlog::sink_ptr> sinks;
                if (!config.filename.empty()) {
                    sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(config.filename, true));
                }
                sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());

                logger = std::make_shared<spdlog::async_logger>(config.name, sinks.begin(), sinks.end(),
                                                                spdlog::thread_pool());
                auto level = static_cast<spdlog::level::level_enum>(config.level);
                logger->set_level(config.level >= level_limit ? level : spdlog::level::off);
                logger->set_pattern(config.format);
                loggers.push_back(logger);
                spdlog::register_logger(loggers.back());
                if (config.name == "default") {
                    default_logger = loggers.back();
                    spdlog::set_default_logger(loggers.back());
                }
            }
}

DaphneLogger::DaphneLogger(DaphneUserConfig& _config) : n_threads(1), queue_size(8192) {
    spdlog::init_thread_pool(queue_size, n_threads);
    try {
        level_limit = static_cast<spdlog::level::level_enum>(_config.log_level_limit);

        // user configured loggers
        for (const auto &config: _config.loggers) {
            createLoggers(config);
        }
        // compiled fallback loggers
        for (const auto &config: fallback_loggers) {
            createLoggers(config);
        }
    }
    catch (const spdlog::spdlog_ex &ex) {
        std::cout << "Log initialization failed: " << ex.what() << std::endl;
    }
    _config.log_ptr = this;
}
