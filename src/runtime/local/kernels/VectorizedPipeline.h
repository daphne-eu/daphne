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
#include <runtime/local/datastructures/Handle.h>
#include <runtime/local/vectorized/MTWrapper.h>

#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

#include <runtime/distributed/worker/ProtoDataConverter.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct VectorizedPipeline
{
    static void
    apply(DTRes *&res, DTLhs *lhs, DTRhs *rhs, void *fun, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void vectorizedPipeline(DTRes *&res, DTLhs *lhs, DTRhs *rhs, void *fun, DCTX(ctx))
{
    VectorizedPipeline<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, fun, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<>
struct VectorizedPipeline<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>>
{
    static void apply(DenseMatrix<double> *&res,
                      DenseMatrix<double> *lhs,
                      DenseMatrix<double> *rhs,
                      void *fun, DCTX(ctx))
    {
        MTWrapper<double> *wrapper = new MTWrapper<double>();
        auto function =
            std::function<void(DenseMatrix<double> ***, DenseMatrix<double> **)>(
                reinterpret_cast<void (*)(DenseMatrix<double> ***, DenseMatrix<double> **)>(fun));
        wrapper->execute(function, res, lhs, rhs, false);
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_VECTORIZEDPIPELINE_H