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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_CONDMATSCAMAT_H
#define SRC_RUNTIME_LOCAL_KERNELS_CONDMATSCAMAT_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>

#include <stdexcept>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTCond, class VTThen, class DTElse>
struct CondMatScaMat {
    static void apply(DTRes *& res, const DTCond * cond, VTThen thenVal, const DTElse * elseVal, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTCond, class VTThen, class DTElse>
void condMatScaMat(DTRes *& res, const DTCond * cond, VTThen thenVal, const DTElse * elseVal, DCTX(ctx)) {
    CondMatScaMat<DTRes, DTCond, VTThen, DTElse>::apply(res, cond, thenVal, elseVal, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, scalar, DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTVal, typename VTCond>
struct CondMatScaMat<DenseMatrix<VTVal>, DenseMatrix<VTCond>, VTVal, DenseMatrix<VTVal>> {
    static void apply(
        DenseMatrix<VTVal> *& res,
        const DenseMatrix<VTCond> * cond,
        VTVal thenVal,
        const DenseMatrix<VTVal> * elseVal,
        DCTX(ctx)
    ) {
        const size_t numRows = cond->getNumRows();
        const size_t numCols = cond->getNumCols();

        if(
            numRows != elseVal->getNumRows() ||
            numCols != elseVal->getNumCols()
        )
            throw std::runtime_error(
                    "CondMatScaMat: condition/else matrices must have the same shape"
            );

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTVal>>(numRows, numCols, false);

        VTVal * valuesRes = res->getValues();
        const VTCond * valuesCond = cond->getValues();
        const VTVal * valuesElse = elseVal->getValues();
        const size_t rowSkipRes = res->getRowSkip();
        const size_t rowSkipCond = cond->getRowSkip();
        const size_t rowSkipElse = elseVal->getRowSkip();

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++)
                valuesRes[c] = static_cast<bool>(valuesCond[c]) ? thenVal : valuesElse[c];
            valuesRes += rowSkipRes;
            valuesCond += rowSkipCond;
            valuesElse += rowSkipElse;
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix, scalar, Matrix
// ----------------------------------------------------------------------------

template<typename VTVal, typename VTCond>
struct CondMatMatSca<Matrix<VTVal>, Matrix<VTCond>, VTVal, Matrix<VTVal>> {
    static void apply(
        Matrix<VTVal> *& res,
        const Matrix<VTCond> * cond,
        VTVal thenVal,
        const Matrix<VTVal> * elseVal,
        DCTX(ctx)
    ) {
        const size_t numRows = cond->getNumRows();
        const size_t numCols = cond->getNumCols();

        if (numRows != elseVal->getNumRows() || numCols != elseVal->getNumCols()) {
            std::ostringstream errMsg;
            errMsg << "CondMatScaMat: condition/else matrices must have the same shape but have ("
                    << numRows << "," << numCols << ") and (" 
                    << elseVal->getNumRows() << "," << elseVal->getNumCols() << ")";
            throw std::runtime_error(errMsg.str());
        }

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTVal>>(numRows, numCols, false);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                res->append(r, c, static_cast<bool>(cond->get(r, c)) ? thenVal : elseVal->get(r, c));
        res->finishAppend();
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_CONDMATSCAMAT_H