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

#define CATCH_CONFIG_MAIN // make catch2 generate a main-function
#include <catch.hpp>

#include <api/cli/DaphneUserConfig.h>
#include "runtime/local/kernels/CreateDaphneContext.h"
#ifdef USE_CUDA
    #include "runtime/local/kernels/CUDA/CreateCUDAContext.h"
#endif

#include "run_tests.h"

std::unique_ptr<DaphneContext> setupContextAndLogger() {
//    ToDo: Setting precompiled log level here as passing user config at runtime is not supported in our test suite
    auto loglevel = spdlog::level::off;
    user_config.log_level_limit = loglevel;
    user_config.loggers.push_back(LogConfig({"default", "daphne-tests.txt", static_cast<int>(loglevel),
            "\">>>>>>>>> %H:%M:%S %z %v\""}));
    if(not logger)
        logger = std::make_unique<DaphneLogger>(user_config);

    DaphneContext* dctx_;
    createDaphneContext(dctx_, reinterpret_cast<uint64_t>(&user_config));

#ifdef USE_CUDA
    CUDA::createCUDAContext(dctx_);
#endif

    return std::unique_ptr<DaphneContext>(dctx_);
}

// Nothing to do here, the individual test cases are in separate cpp-files.