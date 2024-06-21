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
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/datastructures/Matrix.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename VTArg>
struct Fill {
    static void apply(DTRes *& res, VTArg arg, size_t numRows, size_t numCols, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, typename VTArg>
void fill(DTRes *& res, VTArg arg, size_t numRows, size_t numCols, DCTX(ctx)) {
    Fill<DTRes, VTArg>::apply(res, arg, numRows, numCols, ctx);
};

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename VTArg>
struct FillContiguousTensor {
    static void apply(DTRes*& res, VTArg arg, size_t* tensor_shape, size_t rank, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, typename VTArg>
void fillContiguousTensor(DTRes*& res, VTArg arg, size_t* tensor_shape, size_t rank, DCTX(ctx)) {
    FillContiguousTensor<DTRes, VTArg>::apply(res, arg, tensor_shape, rank, ctx);
};

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename VTArg>
struct FillChunkedTensor {
    static void apply(DTRes*& res, VTArg arg, size_t* tensor_shape, size_t* chunk_shape, size_t rank, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, typename VTArg>
void fillChunkedTensor(DTRes*& res, VTArg arg, size_t* tensor_shape, size_t* chunk_shape, size_t rank, DCTX(ctx)) {
    FillChunkedTensor<DTRes, VTArg>::apply(res, arg, tensor_shape, chunk_shape, rank, ctx);
};

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Fill<DenseMatrix<VT>, VT> {
    static void apply(DenseMatrix<VT> *& res, VT arg, size_t numRows, size_t numCols, DCTX(ctx)) {

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, arg == 0);

        if(arg != 0) {
            VT *valuesRes = res->getValues();
            for(auto i = 0ul; i < res->getNumItems(); ++i)
                valuesRes[i] = arg;
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Fill<Matrix<VT>, VT> {
    static void apply(Matrix<VT> *& res, VT arg, size_t numRows, size_t numCols, DCTX(ctx)) {
        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, arg == 0);

        if (arg != 0) {
            res->prepareAppend();
            for (size_t r = 0; r < numRows; ++r)
                for (size_t c = 0; c < numCols; ++c)
                    res->append(r, c, arg);
            res->finishAppend();
        }
    }
};

// ----------------------------------------------------------------------------
// ContiguousTensor
// ----------------------------------------------------------------------------

template<typename VT>
struct FillContiguousTensor<ContiguousTensor<VT>, VT> {
    static void apply(ContiguousTensor<VT>*& res, VT arg, size_t* tensor_shape, size_t rank, DCTX(ctx)) {
        res = DataObjectFactory::create<ContiguousTensor<VT>>(
            std::vector<size_t>(tensor_shape, tensor_shape + rank), InitCode::NONE);

        for (size_t i = 0; i < res->total_element_count; ++i) {
            res->data[i] = arg;
        }
    }
};

// ----------------------------------------------------------------------------
// ChunkedTensor
// ----------------------------------------------------------------------------

template<typename VT>
struct FillChunkedTensor<ChunkedTensor<VT>, VT> {
    static void apply(ChunkedTensor<VT>*& res, VT arg, size_t* tensor_shape, size_t* chunk_shape, size_t rank, DCTX(ctx)) {
        res = DataObjectFactory::create<ChunkedTensor<VT>>(std::vector<size_t>(tensor_shape, tensor_shape + rank),
                                                           std::vector<size_t>(chunk_shape, chunk_shape + rank),
                                                           InitCode::NONE);

        for (size_t i = 0; i < res->total_size_in_elements; ++i) {
            res->data[i] = arg;
        }
    }
};
