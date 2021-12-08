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

template<class DTRes, class DTIn>
struct VectorizedPipeline
{
    static void apply(DTRes *&res,
                      DTIn **inputs,
                      size_t numInputs,
                      size_t numOutputs,
                      int64_t *outRows,
                      int64_t *outCols,
                      int64_t *splits,
                      int64_t *combines,
                      size_t numFuncs,
                      void** fun,
                      DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTIn>
void vectorizedPipeline(DTRes *&res,
                        DTIn **inputs,
                        size_t numInputs,
                        size_t numOutputs,
                        int64_t *outRows,
                        int64_t *outCols,
                        int64_t *splits,
                        int64_t *combines,
                        size_t numFuncs,
                        void** fun,
                        DCTX(ctx))
{
    VectorizedPipeline<DTRes, DTIn>::apply(res,
        inputs,
        numInputs,
        numOutputs,
        outRows,
        outCols,
        splits,
        combines,
        numFuncs,
        fun,
        ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<typename VT>
struct VectorizedPipeline<DenseMatrix<VT>, DenseMatrix<VT>>
{
    static void apply(DenseMatrix<VT> *&res,
                      DenseMatrix<VT> **inputs,
                      size_t numInputs,
                      size_t numOutputs,
                      int64_t *outRows,
                      int64_t *outCols,
                      int64_t *splits,
                      int64_t *combines,
                      size_t numFuncs,
                      void** fun,
                      DCTX(dctx))
    {
        auto wrapper = std::make_unique<MTWrapper<VT>>();
        std::vector<std::function<void(DenseMatrix<VT> ***, DenseMatrix<VT> **, DCTX(ctx))>> funcs;
        for (auto i = 0ul; i < numFuncs; ++i) {
            auto function = std::function<void(DenseMatrix<VT> ***, DenseMatrix<VT> **, DCTX(ctx))>(
                    reinterpret_cast<void (*)(DenseMatrix<VT> ***, DenseMatrix<VT> **, DCTX(ctx))>(
                    reinterpret_cast<void*>(fun[i])));
            funcs.push_back(function);
        }
        wrapper->execute(funcs,
            res,
            inputs,
            numInputs,
            numOutputs,
            outRows,
            outCols,
            reinterpret_cast<VectorSplit *>(splits),
            reinterpret_cast<VectorCombine *>(combines),
            dctx, false);
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_VECTORIZEDPIPELINE_H