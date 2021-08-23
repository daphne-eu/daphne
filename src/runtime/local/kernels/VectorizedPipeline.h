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
    apply(DTRes *&res, const DTLhs *lhs, const DTRhs *rhs, void *fun, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void vectorizedPipeline(DTRes *&res, const DTLhs *lhs, const DTRhs *rhs, void *fun, DCTX(ctx))
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
                      const DenseMatrix<double> *lhs,
                      const DenseMatrix<double> *rhs,
                      void *fun, DCTX(ctx))
    {
        /*MTWrapper<double> *wrapper = new MTWrapper<double>(4);
        wrapper->execute([&](DenseMatrix<double> *lambRes, DenseMatrix<double> *lambLhs, DenseMatrix<double> *lambRhs)
        {

        }, res, lhs, rhs, false);*/
        void **outputs[] = {reinterpret_cast<void **>(&res)};
        void *inputs[] = {reinterpret_cast<void *>(const_cast<DenseMatrix<double> *>(lhs)),
                          reinterpret_cast<void *>(const_cast<DenseMatrix<double> *>(rhs))};
        auto function = reinterpret_cast<void (*)(void ***, void **)>(fun);
        function(outputs, inputs);
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_VECTORIZEDPIPELINE_H