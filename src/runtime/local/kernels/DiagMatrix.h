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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_DIAGMATRIX_H
#define SRC_RUNTIME_LOCAL_KERNELS_DIAGMATRIX_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <stdexcept>

#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct DiagMatrix {
    static void apply(DTRes *& res, const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void diagMatrix(DTRes *& res, const DTArg * arg, DCTX(ctx)) {
    DiagMatrix<DTRes, DTArg>::apply(res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct DiagMatrix<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, DCTX(ctx)) {
        if (arg->getNumCols() != 1) {
            throw std::runtime_error(
                "DiagMatrix.h - parameter arg must be a column-matrix");
        }

        const size_t numRowsCols = arg->getNumRows();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRowsCols, numRowsCols, true);
        
        const VT * valuesArg = arg->getValues();
        VT * valuesRes = res->getValues();
        
        const size_t rowSkipArg = arg->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();
        
        for(size_t r = 0; r < numRowsCols; r++) {
            *valuesRes = *valuesArg;
            valuesArg += rowSkipArg;
            valuesRes += rowSkipRes + 1;
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct DiagMatrix<CSRMatrix<VT>, DenseMatrix<VT>> {
    static void apply(CSRMatrix<VT> *& res, const DenseMatrix<VT> * arg, DCTX(ctx)) {
        if(arg->getNumCols() != 1)
            throw std::runtime_error("parameter arg must be a column-matrix");

        const size_t numRowsCols = arg->getNumRows();
        if(res==nullptr){
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRowsCols, numRowsCols, numRowsCols, false);
        }

        const VT * valuesArg = arg->getValues();
        const size_t rowSkipArg = arg->getRowSkip();

        VT * valuesRes = res->getValues();
        size_t * colIdxsRes = res->getColIdxs();
        size_t * rowOffsetsRes = res->getRowOffsets();

        rowOffsetsRes[0] = 0;

        for(size_t r = 0, pos = 0; r < numRowsCols; r++) {
            if (*valuesArg) {
	        valuesRes[pos] = *valuesArg;
	        colIdxsRes[pos++] = r;
	    }
	    rowOffsetsRes[r + 1] = pos;
            valuesArg += rowSkipArg;
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct DiagMatrix<CSRMatrix<VT>, CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *& res, const CSRMatrix<VT> * arg, DCTX(ctx)) {
        if(arg->getNumCols() != 1)
            throw std::runtime_error("parameter arg must be a column-matrix");

        const size_t numRowsCols = arg->getNumRows();
        if(res==nullptr){
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRowsCols, numRowsCols, numRowsCols, false);
        }

        VT * valuesRes = res->getValues();
        size_t * colIdxsRes = res->getColIdxs();
        size_t * rowOffsetsRes = res->getRowOffsets();

	rowOffsetsRes[0] = 0;

        for(size_t r = 0, pos = 0; r < numRowsCols; r++) {
            if (arg->getNumNonZeros(r)) {
	        valuesRes[pos] = *(arg->getValues(r));
	        colIdxsRes[pos++] = r;
	    }
	    rowOffsetsRes[r + 1] = pos;
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_DIAGMATRIX_H
