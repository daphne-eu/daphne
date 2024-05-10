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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_ROWBIND_H
#define SRC_RUNTIME_LOCAL_KERNELS_ROWBIND_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <stdexcept>

#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTUp, class DTLow>
struct RowBind {
    static void apply(DTRes *& res, const DTUp * ups, const DTLow * lows, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTUp, class DTLow>
void rowBind(DTRes *& res, const DTUp * ups, const DTLow * lows, DCTX(ctx)) {
    RowBind<DTRes, DTUp, DTLow>::apply(res, ups, lows, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct RowBind<DenseMatrix<VT>, DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * ups, const DenseMatrix<VT> * lows, DCTX(ctx)) {
        const size_t numCols = ups->getNumCols();
        if(numCols != lows->getNumCols())
            throw std::runtime_error("ups and lows must have the same number of columns");
        
        const size_t numRowsUps = ups->getNumRows();
        const size_t numRowsLows = lows->getNumRows();
        const size_t numColsUps = ups->getNumCols();
        const size_t numColsLows = lows->getNumCols();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRowsUps + numRowsLows, numCols, false);
        
        
        const VT * valuesUps = ups->getValues();
        const VT * valuesLows = lows->getValues();
        VT * valuesRes = res->getValues();
        
        // TODO Take rowSkip into account. If ups/lows/res are views into
        // column segments of larger data objects, we must proceed row-by-row.
        memcpy(valuesRes, valuesUps, numColsUps * numRowsUps * sizeof(VT));
        memcpy(valuesRes + numRowsUps * numColsUps, valuesLows, numColsLows * numRowsLows * sizeof(VT));
        
    }
};

// ----------------------------------------------------------------------------
// Frame <- Frame, Frame
// ----------------------------------------------------------------------------

template<>
struct RowBind<Frame, Frame, Frame> {
    static void apply(Frame *& res, const Frame * ups, const Frame * lows, const DCTX(ctx)) {
        const size_t numCols = ups->getNumCols();
        const ValueTypeCode* schema = ups->getSchema();
        
        if(numCols != lows->getNumCols())
            throw std::runtime_error("ups and lows must have the same number of columns");
        for(size_t i = 0; i < numCols; i++) {
            if(schema[i] != lows->getSchema()[i])
                throw std::runtime_error("ups and lows must have the same schema");
            if(ups->getLabels()[i] != lows->getLabels()[i])
                throw std::runtime_error("ups and lows must have the same column names");
        }
        
        res = DataObjectFactory::create<Frame>(
                ups->getNumRows() + lows->getNumRows(), numCols,
                schema, ups->getLabels(), false
        );
        for(size_t i = 0; i < numCols; i++){
            const void * colUps = ups->getColumnRaw(i);
            const void * colLows = lows->getColumnRaw(i);
            uint8_t * colRes = reinterpret_cast<uint8_t *>(res->getColumnRaw(i));
            
            const size_t elemSize = ValueTypeUtils::sizeOf(schema[i]);
            memcpy(colRes, colUps, ups->getNumRows() * elemSize);
            memcpy(colRes + ups->getNumRows() * elemSize, colLows, lows->getNumRows() * elemSize);
        }
    }
};


// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix, CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct RowBind<CSRMatrix<VT>, CSRMatrix<VT>, CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *& res, const CSRMatrix<VT> * ups, const CSRMatrix<VT> * lows, DCTX(ctx)) {
        if(ups->getNumCols() != lows->getNumCols())
            throw std::runtime_error("ups and lows must have the same number of columns");

        auto upsRowOffsets = ups->getRowOffsets();
        auto lowsRowOffsets = lows->getRowOffsets();
        
        const size_t upsNumNonZeros = ups->getNumNonZeros();
        const size_t lowsNumNonZeros = lows->getNumNonZeros();
        
        size_t numRowsRes = ups->getNumRows() + lows->getNumRows();
        size_t numNonZerosRes = upsNumNonZeros + lowsNumNonZeros;

        if(!res)
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRowsRes, ups->getNumCols(), numNonZerosRes, false);

        auto resRowOffsets = res->getRowOffsets();
        
        // Ups
        size_t startOffset = upsRowOffsets[0];
        size_t offsetsSubsetLength = upsNumNonZeros;

        if(ups->isView())
            for(size_t rOffset = 0; rOffset < ups->getNumRows(); rOffset++)
                resRowOffsets[rOffset] = upsRowOffsets[rOffset] - startOffset;
        else
            memcpy(resRowOffsets, upsRowOffsets, ups->getNumRows() * sizeof(size_t));
        memcpy(res->getValues(), &ups->getValues()[startOffset], offsetsSubsetLength * sizeof(VT));
        memcpy(res->getColIdxs(), &ups->getColIdxs()[startOffset], offsetsSubsetLength * sizeof(size_t));

        // Lows
        size_t lowsTranslate = upsNumNonZeros;
        startOffset = lowsRowOffsets[0];
        offsetsSubsetLength = lowsNumNonZeros;

        for(size_t rOffset = 0; rOffset < lows->getNumRows(); rOffset++)
            resRowOffsets[rOffset + ups->getNumRows()] = lowsRowOffsets[rOffset] - startOffset + lowsTranslate;
        memcpy(&res->getValues()[lowsTranslate], &lows->getValues()[startOffset], offsetsSubsetLength * sizeof(VT));
        memcpy(&res->getColIdxs()[lowsTranslate], &lows->getColIdxs()[startOffset], offsetsSubsetLength * sizeof(size_t));

        res->getRowOffsets()[numRowsRes] = lowsTranslate + lowsNumNonZeros;
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_ROWBIND_H
