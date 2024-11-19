/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <iostream>

#include "WorkerImpl.h"
#include "WorkerImplGRPCAsync.h"
#include "WorkerImplGRPCSync.h"
#include <parser/config/ConfigParser.h>

// global logger handle for this executable
static std::unique_ptr<DaphneLogger> daphneLogger;

int main(int argc, char *argv[]) {
    DaphneUserConfig user_config{};
    std::string configFile = "WorkerConfig.json";

    // initialize logging facility
    daphneLogger = std::make_unique<DaphneLogger>(user_config);
    auto log = daphneLogger->getDefaultLogger();

    if (argc == 3)
        configFile = argv[2];

    if (ConfigParser::fileExists(configFile)) {
        try {
            ConfigParser::readUserConfig(configFile, user_config);
        } catch (std::exception &e) {
            log->warn("Parser error while reading worker config from {}:\n{}", configFile, e.what());
        }
    }
    user_config.resolveLibDir();

    if (argc < 2 || argc > 3) {
        std::cout << "Usage: " << argv[0] << " <Address:Port> [ConfigFile]" << std::endl;
        exit(1);
    }
    auto addr = argv[1];

    std::unique_ptr<WorkerImpl> service;
    if (user_config.use_grpc_async)
        service = std::make_unique<WorkerImplGRPCAsync>(addr, user_config);
    else
        service = std::make_unique<WorkerImplGRPCSync>(addr, user_config);

    log->info(fmt::format("Started Distributed Worker on {}", addr));
    service->Wait();
    return 0;
}