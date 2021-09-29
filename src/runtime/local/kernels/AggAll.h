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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_AGGALL_H
#define SRC_RUNTIME_LOCAL_KERNELS_AGGALL_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct AggAll {
    static typename DT::VT apply(AggOpCode opCode, const DT * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
typename DT::VT aggAll(AggOpCode opCode, const DT * arg, DCTX(ctx)) {
    return AggAll<DT>::apply(opCode, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// scalar <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct AggAll<DenseMatrix<VT>> {
    static VT apply(AggOpCode opCode, const DenseMatrix<VT> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        const VT * valuesArg = arg->getValues();

        assert(AggOpCodeUtils::isPureBinaryReduction(opCode));

        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(AggOpCodeUtils::getBinaryOpCode(opCode));

        VT agg = AggOpCodeUtils::template getNeutral<VT>(opCode);
        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++)
                agg = func(agg, valuesArg[c], ctx);
            valuesArg += arg->getRowSkip();
        }

        return agg;
    }
};

// ----------------------------------------------------------------------------
// scalar <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct AggAll<CSRMatrix<VT>> {
    static VT aggArray(const VT * values, size_t numNonZeros, size_t numCells, EwBinaryScaFuncPtr<VT, VT, VT> func, bool isSparseSafe, VT neutral, DCTX(ctx)) {
        if(numNonZeros) {
            VT agg = values[0];
            for(size_t i = 1; i < numNonZeros; i++)
                agg = func(agg, values[i], ctx);

            if(!isSparseSafe && numNonZeros < numCells)
                agg = func(agg, 0, ctx);

            return agg;
        }
        else
            return func(neutral, 0, ctx);
    }

    static VT apply(AggOpCode opCode, const CSRMatrix<VT> * arg, DCTX(ctx)) {
        assert(AggOpCodeUtils::isPureBinaryReduction(opCode));

        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(AggOpCodeUtils::getBinaryOpCode(opCode));

        return aggArray(
                arg->getValues(0),
                arg->getNumNonZeros(),
                arg->getNumRows() * arg->getNumCols(),
                func,
                AggOpCodeUtils::isSparseSafe(opCode),
                AggOpCodeUtils::template getNeutral<VT>(opCode),
                ctx
        );
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_AGGALL_H
