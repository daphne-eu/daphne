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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H
#define SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H

#include "runtime/local/datastructures/Tensor.h"
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/UnaryOpCode.h>
#include <runtime/local/kernels/EwUnarySca.h>

#include <cstddef>
#include <vector>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct EwUnaryMat {
    static void apply(UnaryOpCode opCode, DTRes *& res, const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void ewUnaryMat(UnaryOpCode opCode, DTRes *& res, const DTArg * arg, DCTX(ctx)) {
    EwUnaryMat<DTRes, DTArg>::apply(opCode, res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct EwUnaryMat<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(UnaryOpCode opCode, DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
        
        const VT * valuesArg = arg->getValues();
        VT * valuesRes = res->getValues();
        
        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);
        
        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++)
                valuesRes[c] = func(valuesArg[c], ctx);
            valuesArg += arg->getRowSkip();
            valuesRes += res->getRowSkip();
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template<typename VT>
struct EwUnaryMat<Matrix<VT>, Matrix<VT>> {
    static void apply(UnaryOpCode opCode, Matrix<VT> *& res, const Matrix<VT> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                res->append(r, c, func(arg->get(r, c), ctx));
        res->finishAppend();
    }
};

// ----------------------------------------------------------------------------
// ContiguousTensor <- ContiguousTensor
// ----------------------------------------------------------------------------

template<typename VT>
struct EwUnaryMat<ContiguousTensor<VT>, ContiguousTensor<VT>> {
    static void apply(UnaryOpCode opCode, ContiguousTensor<VT> *& res, const ContiguousTensor<VT> * arg, DCTX(ctx)) {
        res = DataObjectFactory::create<ContiguousTensor<VT>>(arg->tensor_shape, InitCode::NONE);
        
        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);
        
        for(size_t i = 0; i < arg->total_element_count; i++) {
            res->data[i] = func(arg->data[i], ctx);
        }
    }
};

// ----------------------------------------------------------------------------
// ChunkedTensor <- ChunkedTensor
// ----------------------------------------------------------------------------

template<typename VT>
struct EwUnaryMat<ChunkedTensor<VT>, ChunkedTensor<VT>> {
    static void apply(UnaryOpCode opCode, ChunkedTensor<VT> *& res, const ChunkedTensor<VT> * arg, DCTX(ctx)) {
        res = DataObjectFactory::create<ChunkedTensor<VT>>(arg->tensor_shape, arg->chunk_shape, InitCode::NONE);
        
        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);
        
        for (size_t i=0; i < arg->total_chunk_count; i++) {
            auto current_chunk_ids = arg->getChunkIdsFromLinearChunkId(i);

            VT* current_chunk_ptr_res = res->getPtrToChunk(current_chunk_ids);
            VT* current_chunk_ptr_arg = arg->getPtrToChunk(current_chunk_ids);

            if (!arg->isPartialChunk(current_chunk_ids)) {
                for (size_t j=0; j < arg->chunk_element_count; j++) {
                    current_chunk_ptr_res[j] = func(current_chunk_ptr_arg[j], ctx);
                }
            } else {
                auto valid_id_bounds = arg->GetIdBoundsOfPartialChunk(current_chunk_ids);
                auto strides = arg->GetStridesOfPartialChunk(valid_id_bounds);
                auto valid_element_count = arg->GetElementCountOfPartialChunk(valid_id_bounds);

                for (size_t i=0; i < valid_element_count; i++) {
                    std::vector<size_t> element_ids;
                    element_ids.resize(arg->rank);

                    int64_t tmp = i;
                    for (int64_t j=arg->rank-1; j >= 0; j--) {
                        element_ids[j] = tmp / strides[j];
                        tmp = tmp % strides[j];
                    }
                    size_t lin_id = 0;
                    for (size_t j = 0; j < arg->rank; j++) {
                        lin_id += element_ids[j] * strides[j]; 
                    }

                    current_chunk_ptr_res[lin_id] = func(current_chunk_ptr_arg[lin_id], ctx);
                }
            }
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H
