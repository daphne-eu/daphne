/*
 * Copyright 2021 The DAPHNE Consortium
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

#include <runtime/local/context/DaphneContext.h>

#include <vector>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <memory>

// TODO: Separate implementation in a .cpp file?
class DistributedContext final : public IContext {
private:
    std::vector<std::string> workers;
public:
    DistributedContext() {

        // TODO: Get the list of distributed workers from daphne user config/cli arguments and
        // keep environmental variables optional.
        auto envVar = std::getenv("DISTRIBUTED_WORKERS");

        if (envVar == nullptr) {
            throw std::runtime_error("--distributed execution is set but EV DISTRIBUTED_WORKERS is empty");
        }

        std::string workersStr(envVar);
        std::string delimiter(",");

        size_t pos;
        while ((pos = workersStr.find(delimiter)) != std::string::npos) {
            workers.push_back(workersStr.substr(0, pos));
            workersStr.erase(0, pos + delimiter.size());
        }
        workers.push_back(workersStr);
    }
    ~DistributedContext() = default;

    static std::unique_ptr<IContext> createDistributedContext() {
        auto ctx = std::unique_ptr<DistributedContext>(new DistributedContext());
        return ctx;
    };

    void destroy() override {
        // Clean up
    };

    static DistributedContext* get(DaphneContext *ctx) { return dynamic_cast<DistributedContext*>(ctx->getDistributedContext()); };

    std::vector<std::string> getWorkers(){
        return workers;
    };
};