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
#include <runtime/local/kernels/BinaryOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct OuterBinary {
    static void apply(BinaryOpCode opCode, DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void outerBinary(BinaryOpCode opCode, DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) {
    OuterBinary<DTRes, DTLhs, DTRhs>::apply(opCode, res, lhs, rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTLhs, typename VTRhs>
struct OuterBinary<DenseMatrix<VTRes>, DenseMatrix<VTLhs>, DenseMatrix<VTRhs>> {
    static void apply(BinaryOpCode opCode, DenseMatrix<VTRes> *& res, const DenseMatrix<VTLhs> * lhs, const DenseMatrix<VTRhs> * rhs, DCTX(ctx)) {
        if(lhs->getNumCols() != 1)
            throw std::runtime_error("outerBinary: lhs must be a column (mx1) matrix");
        if(rhs->getNumRows() != 1)
            throw std::runtime_error("outerBinary: rhs must be a row (1xn) matrix");

        const size_t numRowsLhs = lhs->getNumRows();
        const size_t numColsRhs = rhs->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRowsLhs, numColsRhs, false);
        
        const VTLhs * valuesLhs = lhs->getValues();
        const VTRhs * valuesRhs = rhs->getValues();
        VTRes * valuesRes = res->getValues();
        
        EwBinaryScaFuncPtr<VTRes, VTLhs, VTRhs> func = getEwBinaryScaFuncPtr<VTRes, VTLhs, VTRhs>(opCode);
        
        for(size_t r = 0; r < numRowsLhs; r++) {
            for(size_t c = 0; c < numColsRhs; c++)
                valuesRes[c] = func(valuesLhs[0], valuesRhs[c], ctx);
            valuesLhs += lhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }
    }
};