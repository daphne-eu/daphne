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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_VECTORIZEDPIPELINE_H
#define SRC_RUNTIME_LOCAL_KERNELS_VECTORIZEDPIPELINE_H

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
    static void apply(DTRes *&resIn,
                      Structure **inputs,
                      size_t numInputs,
                      size_t numOutputs,
                      int64_t *outRows,
                      int64_t *outCols,
                      int64_t *splits,
                      int64_t *combines,
                      void *fun,
                      DCTX(ctx)) {
        auto wrapper = std::make_unique<MTWrapper<DTRes>>();
        auto function =
            std::function<void(DTRes ***, Structure **)>(
                reinterpret_cast<void (*)(DTRes ***, Structure **)>(fun));
        assert(numOutputs == 1 && "FIXME: lowered to wrong kernel");
        DTRes **res[] = {&resIn};
        wrapper->execute(function,
            res,
            inputs,
            numInputs,
            numOutputs,
            outRows,
            outCols,
            reinterpret_cast<VectorSplit *>(splits),
            reinterpret_cast<VectorCombine *>(combines),
            false);
    }
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes>
void vectorizedPipeline(DTRes *&res,
                        Structure **inputs,
                        size_t numInputs,
                        size_t numOutputs,
                        int64_t *outRows,
                        int64_t *outCols,
                        int64_t *splits,
                        int64_t *combines,
                        void *fun,
                        DCTX(ctx))
{
    VectorizedPipeline<DTRes>::apply(res,
        inputs,
        numInputs,
        numOutputs,
        outRows,
        outCols,
        splits,
        combines,
        fun,
        ctx);
}

// TODO: use variable args
template<class DTRes>
void vectorizedPipeline(DTRes *&res1,
                        DTRes *&res2,
                        Structure **inputs,
                        size_t numInputs,
                        size_t numOutputs,
                        int64_t *outRows,
                        int64_t *outCols,
                        int64_t *splits,
                        int64_t *combines,
                        void *fun,
                        DCTX(ctx)) {
    auto wrapper = std::make_unique<MTWrapper<DTRes>>();
    auto function =
        std::function<void(DTRes ***, Structure **)>(
            reinterpret_cast<void (*)(DTRes ***, Structure **)>(fun));
    assert(numOutputs == 2 && "FIXME: lowered to wrong kernel");
    DTRes **res[] = {&res1, &res2};
    wrapper->execute(function,
        res,
        inputs,
        numInputs,
        numOutputs,
        outRows,
        outCols,
        reinterpret_cast<VectorSplit *>(splits),
        reinterpret_cast<VectorCombine *>(combines),
        false);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

#endif //SRC_RUNTIME_LOCAL_KERNELS_VECTORIZEDPIPELINE_H