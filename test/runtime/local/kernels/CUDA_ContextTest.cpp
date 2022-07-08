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


#include "runtime/local/kernels/CUDA/CreateCUDAContext.h"

#include <tags.h>

#include <catch.hpp>

TEST_CASE("CreateCUDAContext", TAG_KERNELS) {
    DaphneUserConfig user_config{};
    const size_t deviceID = 0; //ToDo: multi device support
    auto dctx = std::make_unique<DaphneContext>(user_config);
    CUDA::createCUDAContext(dctx.get());
    auto p = CUDAContext::get(dctx.get(), deviceID)->getDeviceProperties();
    CHECK(p);
}