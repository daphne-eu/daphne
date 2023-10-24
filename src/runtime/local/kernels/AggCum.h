/*
 * Copyright 2023 The DAPHNE Consortium
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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct AggCum {
    static void apply(AggOpCode opCode, DTRes *& res, const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void aggCum(AggOpCode opCode, DTRes *& res, const DTArg * arg, DCTX(ctx)) {
    AggCum<DTRes, DTArg>::apply(opCode, res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
struct AggCum<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(AggOpCode opCode, DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, DCTX(ctx)) {
        if(!AggOpCodeUtils::isPureBinaryReduction(opCode))
            throw std::runtime_error("the aggregation function used in aggCum must be a pure binary reduction");

        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);
        
        const VTArg * valuesArg = arg->getValues();
        VTRes * valuesResPrv = res->getValues();
        VTRes * valuesResCur = valuesResPrv;
        
        EwBinaryScaFuncPtr<VTRes, VTRes, VTArg> func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTArg>(
                AggOpCodeUtils::getBinaryOpCode(opCode)
        );

        // First row: copy from arg to res.
        for(size_t c = 0; c < numCols; c++)
            valuesResCur[c] = valuesArg[c];
        valuesArg += arg->getRowSkip();
        valuesResCur += res->getRowSkip();
        // Remaining rows: calculate from previous res row and current arg row.
        for(size_t r = 1; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++)
                valuesResCur[c] = func(valuesResPrv[c], valuesArg[c], ctx);
            valuesArg += arg->getRowSkip();
            valuesResPrv += res->getRowSkip();
            valuesResCur += res->getRowSkip();
        }
    }
};