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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_INSERTROW_H
#define SRC_RUNTIME_LOCAL_KERNELS_INSERTROW_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <stdexcept>

#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTDst, class DTSrc>
struct InsertRow {
    static void apply(const DTDst * dst, const DTSrc * src, size_t rowLowerIncl, size_t rowUpperExcl, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTDst, class DTSrc>
void insertRow(const DTDst * dst, const DTSrc * src, size_t rowLowerIncl, size_t rowUpperExcl, DCTX(ctx)) {
    InsertRow<DTDst, DTSrc>::apply(dst, src, rowLowerIncl, rowUpperExcl, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct InsertRow<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> * dst, const DenseMatrix<VT> * src, size_t rowLowerIncl, size_t rowUpperExcl, DCTX(ctx)) {
        const size_t numRows = src->getNumRows();
        const size_t numCols = src->getNumCols();
        
        if(numRows != rowUpperExcl - rowLowerIncl)
            throw std::runtime_error("insertRow: the number of rows in target and source must match");
        if(numCols != dst->getNumCols())
            throw std::runtime_error("insertRow: the number of cols in target and source must match");
        
        VT * valuesDst = const_cast<VT *>(dst->getValues()) + numCols * rowLowerIncl;
        const VT * valuesSrc = src->getValues();
        const size_t rowSkipDst = dst->getRowSkip();
        const size_t rowSkipSrc = src->getRowSkip();
        
        if(rowSkipSrc == numCols && rowSkipDst == numCols)
            memcpy(valuesDst, valuesSrc, numRows * numCols * sizeof(VT));
        else
            for(size_t r = 0; r < numRows; r++) {
                memcpy(valuesDst, valuesSrc, numCols * sizeof(VT));
                valuesDst += rowSkipDst;
                valuesSrc += rowSkipSrc;
            }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_INSERTROW_H