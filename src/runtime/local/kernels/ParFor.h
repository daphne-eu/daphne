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
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#ifndef SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H
#define SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H

// ****************************************************************************
// Convenience function
// ****************************************************************************
template <class DTRes> void parfor(DTRes **outputs, size_t numOutputs, int64_t from, int64_t to, int64_t step, void *inputs, void *func, DCTX(ctx)) {
    
    auto body = reinterpret_cast<void (*)(void**, void **, int64_t)>(func);
    auto ins = reinterpret_cast<void**>(inputs);
    
    auto ** outputs2 = new DTRes *[numOutputs];
    for (size_t i = 0; i < numOutputs; ++i)
        outputs2[i] = outputs[i];    
    
    auto outs = reinterpret_cast<void**>(outputs2); 

    for (int64_t i = from; i <= to; i += step) {
        printf("[parforLoop] Iteration i = %ld\n", i);
        body(outs, ins, i);
    }

    delete[] outputs2; // Clean up the allocated memory for outputs2
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H
