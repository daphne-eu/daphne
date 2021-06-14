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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <cassert>
#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct DiagMatrix {
    static void apply(DTRes *& res, const DTArg * arg) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void diagMatrix(DTRes *& res, const DTArg * arg) {
    DiagMatrix<DTRes, DTArg>::apply(res, arg);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct DiagMatrix<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg) {
        assert((arg->getNumCols() == 1) && "parameter arg must be a column-matrix");
        
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

#endif //SRC_RUNTIME_LOCAL_KERNELS_DIAGMATRIX_H