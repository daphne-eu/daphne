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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_CONDMATSCASCA_H
#define SRC_RUNTIME_LOCAL_KERNELS_CONDMATSCASCA_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/datastructures/Matrix.h>

#include <vector>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTCond, class VTThen, class VTElse>
struct CondMatScaSca {
    static void apply(DTRes *& res, const DTCond * cond, VTThen thenVal, VTElse elseVal, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTCond, class VTThen, class VTElse>
void condMatScaSca(DTRes *& res, const DTCond * cond, VTThen thenVal, VTElse elseVal, DCTX(ctx)) {
    CondMatScaSca<DTRes, DTCond, VTThen, VTElse>::apply(res, cond, thenVal, elseVal, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, scalar, scalar
// ----------------------------------------------------------------------------

template<typename VTVal, typename VTCond>
struct CondMatScaSca<DenseMatrix<VTVal>, DenseMatrix<VTCond>, VTVal, VTVal> {
    static void apply(
        DenseMatrix<VTVal> *& res,
        const DenseMatrix<VTCond> * cond,
        VTVal thenVal,
        VTVal elseVal,
        DCTX(ctx)
    ) {
        const size_t numRows = cond->getNumRows();
        const size_t numCols = cond->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTVal>>(numRows, numCols, false);

        VTVal * valuesRes = res->getValues();
        const VTCond * valuesCond = cond->getValues();
        const size_t rowSkipRes = res->getRowSkip();
        const size_t rowSkipCond = cond->getRowSkip();

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++)
                valuesRes[c] = static_cast<bool>(valuesCond[c]) ? thenVal : elseVal;
            valuesRes += rowSkipRes;
            valuesCond += rowSkipCond;
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix, scalar, scalar
// ----------------------------------------------------------------------------

template<typename VTVal, typename VTCond>
struct CondMatScaSca<Matrix<VTVal>, Matrix<VTCond>, VTVal, VTVal> {
    static void apply(
        Matrix<VTVal> *& res,
        const Matrix<VTCond> * cond,
        VTVal thenVal,
        VTVal elseVal,
        DCTX(ctx)
    ) {
        const size_t numRows = cond->getNumRows();
        const size_t numCols = cond->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTVal>>(numRows, numCols, false);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                res->append(r, c, static_cast<bool>(cond->get(r, c)) ? thenVal : elseVal);
        res->finishAppend();
    }
};

// ----------------------------------------------------------------------------
// ContiguousTensor <- ContiguousTensor, scalar, scalar
// ----------------------------------------------------------------------------

template<typename VTVal, typename VTCond>
struct CondMatScaSca<ContiguousTensor<VTVal>, ContiguousTensor<VTCond>, VTVal, VTVal> {
    static void apply(
        ContiguousTensor<VTVal> *& res,
        const ContiguousTensor<VTCond> * cond,
        VTVal thenVal,
        VTVal elseVal,
        DCTX(ctx)
    ) {
        res = DataObjectFactory::create<ContiguousTensor<VTVal>>(cond->tensor_shape, InitCode::NONE);

        for (size_t i=0; i < cond->total_element_count; i++) {
            res->data[i] = static_cast<bool>(cond->data[i]) ? thenVal : elseVal;
        }
    }
};

// ----------------------------------------------------------------------------
// ChunkedTensor <- ChunkedTensor, scalar, scalar
// ----------------------------------------------------------------------------

template<typename VTVal, typename VTCond>
struct CondMatScaSca<ChunkedTensor<VTVal>, ChunkedTensor<VTCond>, VTVal, VTVal> {
    static void apply(
        ChunkedTensor<VTVal> *& res,
        const ChunkedTensor<VTCond> * cond,
        VTVal thenVal,
        VTVal elseVal,
        DCTX(ctx)
    ) {
        res = DataObjectFactory::create<ChunkedTensor<VTVal>>(cond->tensor_shape, InitCode::NONE);

        for (size_t i=0; i < cond->total_chunk_count; i++) {
            auto current_chunk_ids = cond->getChunkIdsFromLinearChunkId(i);

            VTVal* current_chunk_ptr_res = res->getPtrToChunk(current_chunk_ids);
            VTVal* current_chunk_ptr_cond = cond->getPtrToChunk(current_chunk_ids);

            if (!cond->isPartialChunk(current_chunk_ids)) {
                for (size_t j=0; j < cond->chunk_element_count; j++) {
                    current_chunk_ptr_res[j] = static_cast<bool>(current_chunk_ptr_cond[j]) ? thenVal : elseVal;
                }
            } else {
                auto valid_id_bounds = cond->GetIdBoundsOfPartialChunk(current_chunk_ids);
                auto strides = cond->GetStridesOfPartialChunk(valid_id_bounds);
                auto valid_element_count = cond->GetElementCountOfPartialChunk(valid_id_bounds);

                for (size_t i=0; i < valid_element_count; i++) {
                    std::vector<size_t> element_ids;
                    element_ids.resize(cond->rank);

                    int64_t tmp = i;
                    for (int64_t j=cond->rank-1; j >= 0; j--) {
                        element_ids[j] = tmp / strides[j];
                        tmp = tmp % strides[j];
                    }
                    size_t lin_id = 0;
                    for (size_t j = 0; j < cond->rank; j++) {
                        lin_id += element_ids[j] * strides[j]; 
                    }

                    current_chunk_ptr_res[lin_id] = static_cast<bool>(current_chunk_ptr_cond[lin_id]) ? thenVal : elseVal;
                }
            }

            res->chunk_materialization_flags[i] = true;
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_CONDMATSCASCA_H
