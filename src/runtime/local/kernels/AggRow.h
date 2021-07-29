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

template<typename VT>
struct AggRow<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(AggOpCode opCode, DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, 1, false);
        
        const VT * valuesArg = arg->getValues();
        VT * valuesRes = res->getValues();
        
        assert(AggOpCodeUtils::isPureBinaryReduction(opCode));
        
        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(AggOpCodeUtils::getBinaryOpCode(opCode));

        for(size_t r = 0; r < numRows; r++) {
            VT agg = *valuesArg;
            for(size_t c = 1; c < numCols; c++)
                agg = func(agg, valuesArg[c], ctx);
            *valuesRes = agg;
            valuesArg += arg->getRowSkip();
            valuesRes += res->getRowSkip();
        }
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct AggRow<DenseMatrix<VT>, CSRMatrix<VT>> {
    static void apply(AggOpCode opCode, DenseMatrix<VT> *& res, const CSRMatrix<VT> * arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, 1, false);
        
        VT * valuesRes = res->getValues();
        
        assert(AggOpCodeUtils::isPureBinaryReduction(opCode));
        
        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(AggOpCodeUtils::getBinaryOpCode(opCode));

        const bool isSparseSafe = AggOpCodeUtils::isSparseSafe(opCode);
        const VT neutral = AggOpCodeUtils::template getNeutral<VT>(opCode);
        
        for(size_t r = 0; r < numRows; r++) {
            *valuesRes = AggAll<CSRMatrix<VT>>::aggArray(
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
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_AGGROW_H