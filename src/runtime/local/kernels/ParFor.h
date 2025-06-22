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

#include <cstdio>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/Structure.h>

#ifndef SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H
#define SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H

// ****************************************************************************
// Convenience function
// ****************************************************************************
void parfor(int64_t from, int64_t to, int64_t step, void *inputs, void *func, DCTX(ctx)) {
    auto body = reinterpret_cast<void (*)(int64_t, void **)>(func);
    auto args = reinterpret_cast<void**>(inputs);
    for (int64_t i = from; i <= to; i += step) {
        //printf("[parforLoop] Iteration i = %ld\n", i);
        body(i, args);
    }
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H
