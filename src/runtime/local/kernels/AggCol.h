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
#include <cmath>

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
        
        EwBinaryScaFuncPtr<VT, VT, VT> func;
        if(AggOpCodeUtils::isPureBinaryReduction(opCode))
            func = getEwBinaryScaFuncPtr<VT, VT, VT>(AggOpCodeUtils::getBinaryOpCode(opCode));
        else
            // TODO Setting the function pointer yields the correct result.
            // However, since MEAN and STDDEV are not sparse-safe, the program
            // does not take the same path for doing the summation, and is less
            // efficient.
            // for MEAN and STDDDEV, we need to sum
            func = getEwBinaryScaFuncPtr<VT, VT, VT>(AggOpCodeUtils::getBinaryOpCode(AggOpCode::SUM));

        memcpy(valuesRes, valuesArg, numCols * sizeof(VT));
        
        for(size_t r = 1; r < numRows; r++) {
            valuesArg += arg->getRowSkip();
            for(size_t c = 0; c < numCols; c++)
                valuesRes[c] = func(valuesRes[c], valuesArg[c]);
        }
        
        if(AggOpCodeUtils::isPureBinaryReduction(opCode))
            return;
        
        // The op-code is either MEAN or STDDEV.

        for(size_t c = 0; c < numCols; c++)
            valuesRes[c] /= numRows;

        if(opCode != AggOpCode::STDDEV)
            return;

        auto tmp = DataObjectFactory::create<DenseMatrix<VT>>(1, numCols, true);
        VT * valuesT = tmp->getValues();
        valuesArg = arg->getValues();

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                VT val = valuesArg[c] - valuesRes[c];
                valuesT[c] = valuesT[c] + val * val;
            }
            valuesArg += arg->getRowSkip();
        }

        for(size_t c = 0; c < numCols; c++) {
            valuesT[c] /= numRows;
            valuesT[c] = sqrt(valuesT[c]);
        }

        // TODO We could avoid copying by returning tmp and destroying res. But
        // that might be wrong if res was not nullptr initially.
        memcpy(valuesRes, valuesT, numCols * sizeof(VT));
        DataObjectFactory::destroy<DenseMatrix<VT>>(tmp);
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
        
        EwBinaryScaFuncPtr<VT, VT, VT> func;
        if(AggOpCodeUtils::isPureBinaryReduction(opCode))
            func = getEwBinaryScaFuncPtr<VT, VT, VT>(AggOpCodeUtils::getBinaryOpCode(opCode));
        else
            // TODO Setting the function pointer yields the correct result.
            // However, since MEAN and STDDEV are not sparse-safe, the program
            // does not take the same path for doing the summation, and is less
            // efficient.
            // for MEAN and STDDDEV, we need to sum
            func = getEwBinaryScaFuncPtr<VT, VT, VT>(AggOpCodeUtils::getBinaryOpCode(AggOpCode::SUM));

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
            }
            
            delete[] hist;
        }

        if(AggOpCodeUtils::isPureBinaryReduction(opCode))
            return;
        
        // The op-code is either MEAN or STDDEV.

        for(size_t c = 0; c < numCols; c++)
            valuesRes[c] /= arg->getNumRows();

        if(opCode != AggOpCode::STDDEV)
            return;

        auto tmp = DataObjectFactory::create<DenseMatrix<VT>>(1, numCols, true);
        VT * valuesT = tmp->getValues();

        size_t * nnzCol = new size_t[numCols](); // initialized to zeros
        for(size_t i = 0; i < numNonZeros; i++) {
            const size_t colIdx = colIdxsArg[i];
            VT val = valuesArg[i] - valuesRes[colIdx];
            valuesT[colIdx] = valuesT[colIdx] + val * val;
            nnzCol[colIdx]++;
        }

        for(size_t c = 0; c < numCols; c++) {
            // Take all zeros in the column into account.
            valuesT[c] += (valuesRes[c] * valuesRes[c]) * (numRows - nnzCol[c]);
            // Finish computation of stddev.
            valuesT[c] /= numRows;
            valuesT[c] = sqrt(valuesT[c]);
        }
        
        delete[] nnzCol;

        // TODO We could avoid copying by returning tmp and destroying res. But
        // that might be wrong if res was not nullptr initially.
        memcpy(valuesRes, valuesT, numCols * sizeof(VT));
        DataObjectFactory::destroy<DenseMatrix<VT>>(tmp);

    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_AGGCOL_H
