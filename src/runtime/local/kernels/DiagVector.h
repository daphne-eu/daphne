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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct DiagVector {
    static void apply(DTRes *& res, const DTArg * arg, DCTX(ctx)) = delete;
};


// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void diagVector(DTRes *& res, const DTArg * arg, DCTX(ctx)) {
    DiagVector<DTRes, DTArg>::apply(res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct DiagVector<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, DCTX(ctx)) {
        //------handling corner cases -------
        if (!arg) {
            throw std::runtime_error("arg must not be nullptr");
        }
        const size_t numRows = arg->getNumRows(); // number of rows
        if (numRows != arg->getNumCols()) {
            throw std::runtime_error("arg matrix should be square matrix");
        }
        if (numRows == 0) {
            throw std::runtime_error("arg matrix cannot be empty");
        }

        if(res==nullptr){
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, 1,  false);
        }
        const VT * allValues = arg->getValues();
        VT * allUpdatedValues = res->getValues();
        const size_t rowSize=arg->getRowSkip();
        for(size_t r = 0; r < numRows; r++){
            allUpdatedValues[r]=allValues[r+(r*rowSize)];
        }
    }
};


// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct DiagVector<DenseMatrix<VT>, CSRMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const CSRMatrix<VT> * arg, DCTX(ctx)) {
        //-------handling corner cases ---------
        if (arg == nullptr) {
            throw std::runtime_error("arg must not be nullptr");
        }
        const size_t numRows = arg->getNumRows(); // number of rows
        if (numRows != arg->getNumCols()) {
            throw std::runtime_error("arg matrix should be square matrix");
        }
        if (numRows == 0) {
            throw std::runtime_error("arg matrix cannot be empty");
        }
        if(res==nullptr){ 
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, 1,  false);
        }
        const VT *allValues= arg->getValues();
        const size_t *rowOffsets = arg->getRowOffsets();
        const size_t *colIndxs= arg->getColIdxs();
        VT * resValues = res->getValues();
        size_t startRowOffset;
        size_t endRowOffset;
        VT targetValue;
        for (size_t i =0 ; i< numRows;++i){
            startRowOffset = rowOffsets[i];
            endRowOffset = rowOffsets[i+1];
            targetValue = 0;
            //TODO perf binary search in row range for i
            for( size_t j = startRowOffset; j<endRowOffset; ++j ){
                if( colIndxs[j]==i) {
                    targetValue=allValues[j];
                    break;
                }
            }
            resValues[i]=targetValue;
        }
    }
};
