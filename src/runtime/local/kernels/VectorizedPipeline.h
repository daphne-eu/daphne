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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/vectorized/MTWrapper.h>
#include <ir/daphneir/Daphne.h>

#include <cassert>
#include <cstddef>

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes>
struct VectorizedPipeline {
    static void apply(DTRes *&resIn, bool* isScalar, Structure **inputs, size_t numInputs, size_t numOutputs, int64_t *outRows,
            int64_t *outCols, int64_t *splits, int64_t *combines, size_t numFuncs, void** fun, DCTX(ctx)) {
        assert(numOutputs == 1 && "FIXME: lowered to wrong kernel");

        auto wrapper = std::make_unique<MTWrapper<DTRes>>(0, numFuncs, ctx);

        std::vector<std::function<void(DTRes ***, Structure **, DCTX(ctx))>> funcs;
        for (auto i = 0ul; i < numFuncs; ++i) {
            funcs.emplace_back(std::function<void(DTRes ***, Structure **, DCTX(ctx))>(
                    reinterpret_cast<void (*)(DTRes ***, Structure **, DCTX(ctx))>(reinterpret_cast<void*>(fun[i]))));
        }

        DTRes **res[] = {&resIn};
        if(ctx->getUserConfig().vectorized_single_queue || numFuncs == 1) {
            wrapper->executeSingleQueue(funcs, res, isScalar, inputs, numInputs, numOutputs, outRows, outCols,
                    reinterpret_cast<VectorSplit *>(splits), reinterpret_cast<VectorCombine *>(combines), ctx, false);
        }
        else {
            wrapper->executeQueuePerDeviceType(funcs, res, isScalar, inputs, numInputs, numOutputs, outRows, outCols,
                    reinterpret_cast<VectorSplit *>(splits), reinterpret_cast<VectorCombine *>(combines), ctx, false);
        }
    }
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes>
void vectorizedPipeline(DTRes *&res, bool* isScalar, Structure **inputs, size_t numInputs, size_t numOutputs, int64_t *outRows,
        int64_t *outCols, int64_t *splits, int64_t *combines, size_t numFuncs, void** fun, DCTX(ctx)) {
    VectorizedPipeline<DTRes>::apply(res, isScalar, inputs, numInputs, numOutputs, outRows, outCols, splits, combines, numFuncs,
        fun, ctx);
}

// TODO: use variable args
template<class DTRes>
void vectorizedPipeline(DTRes *&res1, DTRes *&res2, bool* isScalar, Structure **inputs, size_t numInputs, size_t numOutputs,
        int64_t *outRows, int64_t *outCols, int64_t *splits, int64_t *combines, size_t numFuncs, void** fun, DCTX(ctx)){
    assert(numOutputs == 2 && "FIXME: lowered to wrong kernel");

    auto wrapper = std::make_unique<MTWrapper<DTRes>>(0, numFuncs, ctx);

    std::vector<std::function<void(DTRes ***, Structure **, DCTX(ctx))>> funcs;
    for (auto i = 0ul; i < numFuncs; ++i) {
        funcs.emplace_back(std::function<void(DTRes ***, Structure **, DCTX(ctx))>(reinterpret_cast<void (*)(DTRes ***,
                Structure **, DCTX(ctx))>(reinterpret_cast<void*>(fun[i]))));
    }

//    DTRes **res[] = {&res1, &res2};
    DTRes*** res = new DTRes**[2];
    res[0] = &res1;
    res[1] = &res2;
    if(ctx->getUserConfig().vectorized_single_queue || numFuncs == 1) {
        wrapper->executeSingleQueue(funcs, res, isScalar, inputs, numInputs, numOutputs, outRows, outCols,
                reinterpret_cast<VectorSplit *>(splits), reinterpret_cast<VectorCombine *>(combines), ctx, false);
    }
    else {
        wrapper->executeQueuePerDeviceType(funcs, res, isScalar, inputs, numInputs, numOutputs, outRows, outCols,
                reinterpret_cast<VectorSplit *>(splits), reinterpret_cast<VectorCombine *>(combines), ctx, false);
    }
}
