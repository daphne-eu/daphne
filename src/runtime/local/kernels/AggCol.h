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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_AGGCOL_H
#define SRC_RUNTIME_LOCAL_KERNELS_AGGCOL_H

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <cassert>
#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct AggCol {
    static void apply(AggOpCode opCode, DTRes *& res, const DTArg * arg) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void aggCol(AggOpCode opCode, DTRes *& res, const DTArg * arg) {
    AggCol<DTRes, DTArg>::apply(opCode, res, arg);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct AggCol<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(AggOpCode opCode, DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(1, numCols, false);
        
        const VT * valuesArg = arg->getValues();
        VT * valuesRes = res->getValues();
        
        assert(AggOpCodeUtils::isPureBinaryReduction(opCode));
        
        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(AggOpCodeUtils::getBinaryOpCode(opCode));

        memcpy(valuesRes, valuesArg, numCols * sizeof(VT));
        
        for(size_t r = 1; r < numRows; r++) {
            valuesArg += arg->getRowSkip();
            for(size_t c = 0; c < numCols; c++)
                valuesRes[c] = func(valuesRes[c], valuesArg[c]);
        }
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct AggCol<DenseMatrix<VT>, CSRMatrix<VT>> {
    static void apply(AggOpCode opCode, DenseMatrix<VT> *& res, const CSRMatrix<VT> * arg) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(1, numCols, true);
        
        VT * valuesRes = res->getValues();
        
        assert(AggOpCodeUtils::isPureBinaryReduction(opCode));
        
        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(AggOpCodeUtils::getBinaryOpCode(opCode));

        const VT * valuesArg = arg->getValues(0);
        const size_t * colIdxsArg = arg->getColIdxs(0);
        
        const size_t numNonZeros = arg->getNumNonZeros();
        
        if(AggOpCodeUtils::isSparseSafe(opCode)) {
            for(size_t i = 0; i < numNonZeros; i++) {
                const size_t colIdx = colIdxsArg[i];
                valuesRes[colIdx] = func(valuesRes[colIdx], valuesArg[i]);
            }
        }
        else {
            size_t * hist = new size_t[numCols](); // initialized to zeros

            const size_t numNonZerosFirstRowArg = arg->getNumNonZeros(0);
            for(size_t i = 0; i < numNonZerosFirstRowArg; i++) {
                size_t colIdx = colIdxsArg[i];
                valuesRes[colIdx] = valuesArg[i];
                hist[colIdx]++;
            }

            if(arg->getNumRows() > 1) {
                for(size_t i = numNonZerosFirstRowArg; i < numNonZeros; i++) {
                    const size_t colIdx = colIdxsArg[i];
                    valuesRes[colIdx] = func(valuesRes[colIdx], valuesArg[i]);
                    hist[colIdx]++;
                }
                for(size_t c = 0; c < numCols; c++)
                    if(hist[c] < numRows)
                        valuesRes[c] = func(valuesRes[c], 0);
                delete[] hist;
            }
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_AGGCOL_H