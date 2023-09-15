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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H
#define SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H

#include <runtime/local/kernels/InPlaceUtils.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/UnaryOpCode.h>
#include <runtime/local/kernels/EwUnarySca.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct EwUnaryMat {
    static void apply(UnaryOpCode opCode, DTRes *& res, DTArg * arg, bool hasFutureUseArg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void ewUnaryMat(UnaryOpCode opCode, DTRes *& res, DTArg * arg, bool hasFutureUseArg, DCTX(ctx)) {
    EwUnaryMat<DTRes, DTArg>::apply(opCode, res, arg, hasFutureUseArg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct EwUnaryMat<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(UnaryOpCode opCode, DenseMatrix<VT> *& res, DenseMatrix<VT> * arg, bool hasFutureUseArg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(res == nullptr) {
            if(InPlaceUtils::isInPlaceable(arg, hasFutureUseArg)) {
                spdlog::debug("EwUnaryMat(Dense) - arg is in-placeable");
                res = arg;
                res->increaseRefCounter();
            }
            else {
                spdlog::debug("EwUnaryMat(Dense) - create new matrix for result");
                res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
            }
        }
        
        const VT * valuesArg = arg->getValues();
        VT * valuesRes = res->getValues();
        
        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);
        
        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++)
                valuesRes[c] = func(valuesArg[c], ctx);
            valuesArg += arg->getRowSkip();
            valuesRes += res->getRowSkip();
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H