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

#pragma once

#include <spdlog/async.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <api/cli/DaphneUserConfig.h>
#include <vector>

struct DaphneUserConfig;

class DaphneLogger {
    const static std::vector<LogConfig> fallback_loggers;
    std::shared_ptr<spdlog::logger> default_logger{};
    std::vector<std::shared_ptr<spdlog::logger>> loggers{};
    spdlog::level::level_enum level_limit{};
    int n_threads;
    int queue_size;

    void createLoggers(const LogConfig& config);

public:
    explicit DaphneLogger(DaphneUserConfig& config);

    [[maybe_unused]] std::vector<std::shared_ptr<spdlog::logger>>* getLoggers() {
        return &loggers;
    }

    [[maybe_unused]] std::shared_ptr<spdlog::logger> getDefaultLogger() { return default_logger; }

    // register loggers in shared libraries
    void registerLoggers() {
        for (const auto& logger : loggers) {
            if(not spdlog::get(logger->name())) {
                spdlog::register_logger(logger);
            }
        }
        spdlog::set_default_logger(default_logger);
    }
};

