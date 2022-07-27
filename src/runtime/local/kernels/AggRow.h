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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_AGGROW_H
#define SRC_RUNTIME_LOCAL_KERNELS_AGGROW_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/AggAll.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct AggRow {
    static void apply(AggOpCode opCode, DTRes *& res, const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void aggRow(AggOpCode opCode, DTRes *& res, const DTArg * arg, DCTX(ctx)) {
    AggRow<DTRes, DTArg>::apply(opCode, res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
struct AggRow<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(AggOpCode opCode, DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, false);
        
        const VTArg * valuesArg = arg->getValues();
        VTRes * valuesRes = res->getValues();
        
        if(opCode == AggOpCode::IDXMIN) {
            for(size_t r = 0; r < numRows; r++) {
                VTArg minVal = valuesArg[0];
                size_t minValIdx = 0;
                for(size_t c = 1; c < numCols; c++)
                    if(valuesArg[c] < minVal) {
                        minVal = valuesArg[c];
                        minValIdx = c;
                    }
                *valuesRes = static_cast<VTRes>(minValIdx);
                valuesArg += arg->getRowSkip();
                valuesRes += res->getRowSkip();
            }
        }
        else if(opCode == AggOpCode::IDXMAX) {
            for(size_t r = 0; r < numRows; r++) {
                VTArg maxVal = valuesArg[0];
                size_t maxValIdx = 0;
                for(size_t c = 1; c < numCols; c++)
                    if(valuesArg[c] > maxVal) {
                        maxVal = valuesArg[c];
                        maxValIdx = c;
                    }
                *valuesRes = static_cast<VTRes>(maxValIdx);
                valuesArg += arg->getRowSkip();
                valuesRes += res->getRowSkip();
            }
        }
        else {
            EwBinaryScaFuncPtr<VTRes, VTArg, VTArg> func;    
            if(AggOpCodeUtils::isPureBinaryReduction(opCode))
                func = getEwBinaryScaFuncPtr<VTRes, VTArg, VTArg>(AggOpCodeUtils::getBinaryOpCode(opCode));
            else
                // TODO Setting the function pointer yields the correct result.
                // However, since MEAN and STDDEV are not sparse-safe, the program
                // does not take the same path for doing the summation, and is less
                // efficient.
                // for MEAN and STDDDEV, we need to sum
                func = getEwBinaryScaFuncPtr<VTRes, VTArg, VTArg>(AggOpCodeUtils::getBinaryOpCode(AggOpCode::SUM));

            for(size_t r = 0; r < numRows; r++) {
                VTRes agg = static_cast<VTRes>(*valuesArg);
                for(size_t c = 1; c < numCols; c++)
                    agg = func(agg, valuesArg[c], ctx);
                *valuesRes = static_cast<VTRes>(agg);
                valuesArg += arg->getRowSkip();
                valuesRes += res->getRowSkip();
            }

            if(AggOpCodeUtils::isPureBinaryReduction(opCode))
                return;

            // The op-code is either MEAN or STDDEV
            valuesRes = res->getValues();
            for(size_t r = 0; r < numRows; r++) {
                *valuesRes = (*valuesRes) / numCols;
                valuesRes += res->getRowSkip();
            }
            if(opCode == AggOpCode::MEAN)
                return;
            
            // else op-code is STDDEV
            // TODO STDDEV
            throw std::runtime_error("unsupported AggOpCode in AggRow for DenseMatrix");
        }
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
struct AggRow<DenseMatrix<VTRes>, CSRMatrix<VTArg>> {
    static void apply(AggOpCode opCode, DenseMatrix<VTRes> *& res, const CSRMatrix<VTArg> * arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, false);
        
        VTRes * valuesRes = res->getValues();
        
        if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {
        
            EwBinaryScaFuncPtr<VTRes, VTArg, VTArg> func = getEwBinaryScaFuncPtr<VTRes, VTArg, VTArg>(AggOpCodeUtils::getBinaryOpCode(opCode));

            const bool isSparseSafe = AggOpCodeUtils::isSparseSafe(opCode);
            const VTRes neutral = AggOpCodeUtils::template getNeutral<VTRes>(opCode);
        
            for(size_t r = 0; r < numRows; r++) {
                *valuesRes = AggAll<VTRes, CSRMatrix<VTArg>>::aggArray(
                        arg->getValues(r),
                        arg->getNumNonZeros(r),
                        numCols,
                        func,
                        isSparseSafe,
                        neutral,
                        ctx
                );
                valuesRes += res->getRowSkip();
            }
        }
        else { // The op-code is either MEAN or STDDEV
            // get sum for each row
            const VTRes neutral = VTRes(0);
            const bool isSparseSafe = true;
            EwBinaryScaFuncPtr<VTRes, VTArg, VTArg> func = getEwBinaryScaFuncPtr<VTRes, VTArg, VTArg>(AggOpCodeUtils::getBinaryOpCode(AggOpCode::SUM));
            for (size_t r = 0; r < numRows; r++){
                *valuesRes = AggAll<VTRes, CSRMatrix<VTArg>>::aggArray(
                    arg->getValues(r),
                    arg->getNumNonZeros(r),
                    numCols,
                    func,
                    isSparseSafe,
                    neutral,
                    ctx
                );
                if (opCode == AggOpCode::MEAN)
                    *valuesRes = *valuesRes / numCols;
                else
                    throw std::runtime_error("unsupported AggOpCode in AggRow for CSRMatrix");
                valuesRes += res->getRowSkip();
            }
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_AGGROW_H