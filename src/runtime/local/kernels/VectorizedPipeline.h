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
                      int64_t *splits,
                      int64_t *combines,
                      void *fun,
                      DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTIn>
void vectorizedPipeline(DTRes *&res,
                        DTIn **inputs,
                        size_t numInputs,
                        int64_t *splits,
                        int64_t *combines,
                        void *fun,
                        DCTX(ctx))
{
    VectorizedPipeline<DTRes, DTIn>::apply(res, inputs, numInputs, splits, combines, fun, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<>
struct VectorizedPipeline<DenseMatrix<double>, DenseMatrix<double>>
{
    static void apply(DenseMatrix<double> *&res,
                      DenseMatrix<double> **inputs,
                      size_t numInputs,
                      int64_t *splits,
                      int64_t *combines,
                      void *fun,
                      DCTX(ctx))
    {
        MTWrapper<double> *wrapper = new MTWrapper<double>();
        auto function =
            std::function<void(DenseMatrix<double> ***, DenseMatrix<double> **)>(
                reinterpret_cast<void (*)(DenseMatrix<double> ***, DenseMatrix<double> **)>(fun));
        wrapper->execute(function,
            res,
            inputs,
            numInputs,
            reinterpret_cast<VectorSplit *>(splits),
            reinterpret_cast<VectorCombine *>(combines),
            false);
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_VECTORIZEDPIPELINE_H