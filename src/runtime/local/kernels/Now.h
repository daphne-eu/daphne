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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_NOW_H
#define SRC_RUNTIME_LOCAL_KERNELS_NOW_H

#include <runtime/local/context/DaphneContext.h>

#include <chrono>
#include <iostream>

#include <cstdint>

// ****************************************************************************
// Convenience function
// ****************************************************************************

int64_t now(DCTX(ctx)) {
    using clock = std::chrono::high_resolution_clock;

    if (clock::period::num != 1 || clock::period::den != 1000000000) {
        throw std::runtime_error(
            "now() expects std::chrono::high_resolution_clock to be in nano seconds");
    }
    return clock::now().time_since_epoch().count();
}

#endif //SRC_RUNTIME_LOCAL_KERNELS_NOW_H
