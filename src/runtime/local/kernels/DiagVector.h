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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_DIAGVECTOR_H
#define SRC_RUNTIME_LOCAL_KERNELS_DIAGVECTOR_H

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <string.h>
#include <cstddef>
#include <cassert>
#include <stdio.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTArg>
struct DiagVector {
    static void apply(DenseMatrix<typename DTArg::VT> *& res, const DTArg * arg) = delete;
};


// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTArg>
void diagVector(DenseMatrix<typename DTArg::VT> *& res, const DTArg * arg) {
    DiagVector<DTArg>::apply(res, arg);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct DiagVector<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg) {
        //------handling corner cases -------
        assert(arg!=nullptr&& "arg must not be nullptr"); // the arg matrix cannot be a nullptr
        const size_t numRows = arg->getNumRows(); // number of rows
        const size_t numCols = arg->getNumCols(); // number of columns
        assert(numRows==numCols && "arg matrix should be square matrix");
        assert(numRows!=0 && "arg matrix cannot be empty");
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
// CSRMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct DiagVector<CSRMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const CSRMatrix<VT> * arg) {
        //-------handling corner cases ---------
        assert(arg!=nullptr&& "arg must not be nullptr"); // the arg matrix cannot be a nullptr
        const size_t numRows = arg->getNumRows(); // number of rows
        const size_t numCols = arg->getNumCols(); // number of columns
        assert(numRows==numCols && "arg matrix should be square matrix");
        assert(numRows!=0 && "arg matrix cannot be empty");
        if(res==nullptr){ 
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, 1,  false);
        }
        const VT * allValues= arg->getValues();
        const size_t * rowOffsets = arg->getRowOffsets();
        const size_t * colIndxs= arg->getColIdxs();
        VT * resValues= res->getValues();
        size_t startRowOffset=0;
        size_t endRowOffset=0;
        VT targetValue=0;
        for (size_t i =0 ; i< numRows;++i){
            startRowOffset = rowOffsets[i];
            endRowOffset = rowOffsets[i+1];
            targetValue =0;
            for( size_t j =startRowOffset ;j<endRowOffset;++j){
                    if( colIndxs[j]==i){
                        targetValue=allValues[j];
                        break;
                    }
            }
            resValues[i]=targetValue;            
        }	
    }
};
#endif
