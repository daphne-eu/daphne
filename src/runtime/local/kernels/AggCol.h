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

#include <runtime/local/context/DaphneContext.h>
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
    static void apply(AggOpCode opCode, DTRes *& res, const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void aggCol(AggOpCode opCode, DTRes *& res, const DTArg * arg, DCTX(ctx)) {
    AggCol<DTRes, DTArg>::apply(opCode, res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
struct AggCol<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(AggOpCode opCode, DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(1, numCols, false);
        
        const VTArg * valuesArg = arg->getValues();
        VTRes * valuesRes = res->getValues();
        
        // TODO Merge the cases for IDXMIN and IDXMAX to avoid code duplication.
        if(opCode == AggOpCode::IDXMIN) {
            // Minimum values seen so far per column (initialize with first row of argument).
            auto tmp = DataObjectFactory::create<DenseMatrix<VTArg>>(1, numCols, false);
            VTArg * valuesTmp = tmp->getValues();
            memcpy(valuesTmp, valuesArg, numCols * sizeof(VTArg));

            // Positions at which the minimum values were found (initialize with zeros),
            // stored directly in the result.
            for (size_t c = 0; c < numCols; c++)
                valuesRes[c] = 0;

            // Scan over the remaining rows and update the minimum values and their positions accordingly.
            valuesArg += arg->getRowSkip();
            for(size_t r = 1; r < numRows; r++) {
                for(size_t c = 0; c < numCols; c++)
                    if(valuesArg[c] < valuesTmp[c]) {
                        valuesTmp[c] = valuesArg[c];
                        valuesRes[c] = r;
                    }
                valuesArg += arg->getRowSkip();
            }

            // Free the temporary minimum values.
            DataObjectFactory::destroy(tmp);
        }
        else if(opCode == AggOpCode::IDXMAX) {
            // Maximum values seen so far per column (initialize with first row of argument).
            auto tmp = DataObjectFactory::create<DenseMatrix<VTArg>>(1, numCols, false);
            VTArg * valuesTmp = tmp->getValues();
            memcpy(valuesTmp, valuesArg, numCols * sizeof(VTArg));

            // Positions at which the maximum values were found (initialize with zeros),
            // stored directly in the result.
            for (size_t c = 0; c < numCols; c++)
                valuesRes[c] = 0;

            // Scan over the remaining rows and update the maximum values and their positions accordingly.
            valuesArg += arg->getRowSkip();
            for(size_t r = 1; r < numRows; r++) {
                for(size_t c = 0; c < numCols; c++)
                    if(valuesArg[c] > valuesTmp[c]) {
                        valuesTmp[c] = valuesArg[c];
                        valuesRes[c] = r;
                    }
                valuesArg += arg->getRowSkip();
            }

            // Free the temporary maximum values.
            DataObjectFactory::destroy(tmp);
        }
        else {
            EwBinaryScaFuncPtr<VTRes, VTRes, VTRes> func;
            if(AggOpCodeUtils::isPureBinaryReduction(opCode))
                func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(opCode));
            else
                // TODO Setting the function pointer yields the correct result.
                // However, since MEAN and STDDEV are not sparse-safe, the program
                // does not take the same path for doing the summation, and is less
                // efficient.
                // for MEAN and STDDDEV, we need to sum
                func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(AggOpCode::SUM));

            // memcpy(valuesRes, valuesArg, numCols * sizeof(VTRes));
            // Can't memcpy because we might have different result type
            for (size_t c = 0; c < numCols; c++)
                valuesRes[c] = static_cast<VTRes>(valuesArg[c]);
            for(size_t r = 1; r < numRows; r++) {
                valuesArg += arg->getRowSkip();
                for(size_t c = 0; c < numCols; c++)
                    valuesRes[c] = func(valuesRes[c], static_cast<VTRes>(valuesArg[c]), ctx);
            }
            
            if(AggOpCodeUtils::isPureBinaryReduction(opCode))
                return;
            
            // The op-code is either MEAN or STDDEV or VAR.

            for(size_t c = 0; c < numCols; c++)
                valuesRes[c] /= numRows;

            if(opCode == AggOpCode::MEAN)
                return;

            auto tmp = DataObjectFactory::create<DenseMatrix<VTRes>>(1, numCols, true);
            VTRes * valuesT = tmp->getValues();
            valuesArg = arg->getValues();

            for(size_t r = 0; r < numRows; r++) {
                for(size_t c = 0; c < numCols; c++) {
                    VTRes val = static_cast<VTRes>(valuesArg[c]) - valuesRes[c];
                    valuesT[c] = valuesT[c] + val * val;
                }
                valuesArg += arg->getRowSkip();
            }

            for(size_t c = 0; c < numCols; c++) {
                valuesT[c] /= numRows;
                if (opCode == AggOpCode::STDDEV)
                    valuesT[c] = sqrt(valuesT[c]);
            }
            

            
            // TODO We could avoid copying by returning tmp and destroying res. But
            // that might be wrong if res was not nullptr initially.
            memcpy(valuesRes, valuesT, numCols * sizeof(VTRes));
            DataObjectFactory::destroy<DenseMatrix<VTRes>>(tmp);
        }
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
struct AggCol<DenseMatrix<VTRes>, CSRMatrix<VTArg>> {
    static void apply(AggOpCode opCode, DenseMatrix<VTRes> *& res, const CSRMatrix<VTArg> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(1, numCols, true);
        
        VTRes * valuesRes = res->getValues();
        
        EwBinaryScaFuncPtr<VTRes, VTRes, VTRes> func;
        if(AggOpCodeUtils::isPureBinaryReduction(opCode))
            func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(opCode));
        else
            // TODO Setting the function pointer yields the correct result.
            // However, since MEAN and STDDEV are not sparse-safe, the program
            // does not take the same path for doing the summation, and is less
            // efficient.
            // for MEAN and STDDDEV, we need to sum
            func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(AggOpCode::SUM));

        const VTArg * valuesArg = arg->getValues(0);
        const size_t * colIdxsArg = arg->getColIdxs(0);
        
        const size_t numNonZeros = arg->getNumNonZeros();
        
        if(AggOpCodeUtils::isSparseSafe(opCode)) {
            for(size_t i = 0; i < numNonZeros; i++) {
                const size_t colIdx = colIdxsArg[i];
                valuesRes[colIdx] = func(valuesRes[colIdx], static_cast<VTRes>(valuesArg[i]), ctx);
            }
        }
        else {
            size_t * hist = new size_t[numCols](); // initialized to zeros

            const size_t numNonZerosFirstRowArg = arg->getNumNonZeros(0);
            for(size_t i = 0; i < numNonZerosFirstRowArg; i++) {
                size_t colIdx = colIdxsArg[i];
                valuesRes[colIdx] = static_cast<VTRes>(valuesArg[i]);
                hist[colIdx]++;
            }

            if(arg->getNumRows() > 1) {
                for(size_t i = numNonZerosFirstRowArg; i < numNonZeros; i++) {
                    const size_t colIdx = colIdxsArg[i];
                    valuesRes[colIdx] = func(valuesRes[colIdx], static_cast<VTRes>(valuesArg[i]), ctx);
                    hist[colIdx]++;
                }
                for(size_t c = 0; c < numCols; c++)
                    if(hist[c] < numRows)
                        valuesRes[c] = func(valuesRes[c], VTRes(0), ctx);
            }
            
            delete[] hist;
        }

        if(AggOpCodeUtils::isPureBinaryReduction(opCode))
            return;
        
        // The op-code is either MEAN or STDDEV or VAR.

        for(size_t c = 0; c < numCols; c++)
            valuesRes[c] /= arg->getNumRows();

        if(opCode == AggOpCode::MEAN)
            return;

        auto tmp = DataObjectFactory::create<DenseMatrix<VTRes>>(1, numCols, true);
        VTRes * valuesT = tmp->getValues();

        size_t * nnzCol = new size_t[numCols](); // initialized to zeros
        for(size_t i = 0; i < numNonZeros; i++) {
            const size_t colIdx = colIdxsArg[i];
            VTRes val = static_cast<VTRes>(valuesArg[i]) - valuesRes[colIdx];
            valuesT[colIdx] = valuesT[colIdx] + val * val;
            nnzCol[colIdx]++;
        }

        for(size_t c = 0; c < numCols; c++) {
            // Take all zeros in the column into account.
            valuesT[c] += (valuesRes[c] * valuesRes[c]) * (numRows - nnzCol[c]);
            // Finish computation of stddev.
            valuesT[c] /= numRows;
            if (opCode == AggOpCode::STDDEV)
                valuesT[c] = sqrt(valuesT[c]);
        }
        
        delete[] nnzCol;

        // TODO We could avoid copying by returning tmp and destroying res. But
        // that might be wrong if res was not nullptr initially.
        memcpy(valuesRes, valuesT, numCols * sizeof(VTRes));
        DataObjectFactory::destroy<DenseMatrix<VTRes>>(tmp);

    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_AGGCOL_H