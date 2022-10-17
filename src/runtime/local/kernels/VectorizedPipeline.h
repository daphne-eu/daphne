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
#include <runtime/local/vectorized/MTWrapperCUDA.h>
#include <util/ILibCUDA.h>
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
    static void apply(DTRes ** outputs, size_t numOutputs, bool* isScalar, Structure **inputs, size_t numInputs, int64_t *outRows,
            int64_t *outCols, int64_t *splits, int64_t *combines, size_t numFuncs, void** fun, DCTX(ctx)) {
        auto wrapper = std::make_unique<MTWrapper<DTRes>>(numFuncs, ctx);

        std::vector<std::function<void(DTRes ***, Structure **, DCTX(ctx))>> funcs;
        for (auto i = 0ul; i < numFuncs; ++i) {
            funcs.emplace_back(std::function<void(DTRes ***, Structure **, DCTX(ctx))>(
                    reinterpret_cast<void (*)(DTRes ***, Structure **, DCTX(ctx))>(reinterpret_cast<void*>(fun[i]))));
        }

        // TODO Do we really need *** here, isn't ** enough?
        auto *** outputs2 = new DTRes**[numOutputs];
        for(size_t i = 0; i < numOutputs; i++)
            outputs2[i] = outputs + i;

// ToDo: Quick but ugly solution to get libCUDAKernels separated from libAllKernels

        if(numFuncs == 1) {
            wrapper->executeCpuQueues(funcs, outputs2, isScalar, inputs, numInputs, numOutputs, outRows, outCols,
                    reinterpret_cast<VectorSplit *>(splits), reinterpret_cast<VectorCombine *>(combines), ctx, false);
        }
#ifdef USE_CUDA
        else {

            std::cout << "instantiating CUDA vectorized executor" << std::endl;
            IVectorizedExecutor* cuda_vexec = create_cuda_vectorized_executor();
            //testing executor --> this is a noop atm
            cuda_vexec->executeQueuePerDeviceType(3, reinterpret_cast<void***>(outputs2), isScalar, inputs, numInputs, numOutputs, outRows, outCols,
                    reinterpret_cast<VectorSplit *>(splits), reinterpret_cast<VectorCombine *>(combines), ctx, false);
            std::cout << "unloading cuda vexec" << std::endl;

            // this one does the work for now (without GPU of course)
            wrapper->executeCpuQueues(funcs, outputs2, isScalar, inputs, numInputs, numOutputs, outRows, outCols,
                                      reinterpret_cast<VectorSplit *>(splits), reinterpret_cast<VectorCombine *>(combines), ctx, false);

            destroy_cuda_vectorized_executor(cuda_vexec);
        }
#endif
        delete[] outputs2;
    }
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes>
[[maybe_unused]] void vectorizedPipeline(DTRes ** outputs, size_t numOutputs, bool* isScalar, Structure **inputs,
        size_t numInputs, int64_t *outRows, int64_t *outCols, int64_t *splits, int64_t *combines, size_t numFuncs,
        void** fun, DCTX(ctx)) {
    VectorizedPipeline<DTRes>::apply(outputs, numOutputs, isScalar, inputs, numInputs, outRows, outCols, splits,
            combines, numFuncs, fun, ctx);
}
