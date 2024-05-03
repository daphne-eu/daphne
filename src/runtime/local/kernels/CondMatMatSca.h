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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_CONDMATMATSCA_H
#define SRC_RUNTIME_LOCAL_KERNELS_CONDMATMATSCA_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>

#include <sstream>
#include <stdexcept>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTCond, class DTThen, class VTElse>
struct CondMatMatSca {
    static void apply(DTRes *& res, const DTCond * cond, const DTThen * thenVal, VTElse elseVal, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTCond, class DTThen, class VTElse>
void condMatMatSca(DTRes *& res, const DTCond * cond, const DTThen * thenVal, VTElse elseVal, DCTX(ctx)) {
    CondMatMatSca<DTRes, DTCond, DTThen, VTElse>::apply(res, cond, thenVal, elseVal, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************


// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix, scalar
// ----------------------------------------------------------------------------

template<typename VTVal, typename VTCond>
struct CondMatMatSca<DenseMatrix<VTVal>, DenseMatrix<VTCond>, DenseMatrix<VTVal>, VTVal> {
    static void apply(
        DenseMatrix<VTVal> *& res,
        const DenseMatrix<VTCond> * cond,
        const DenseMatrix<VTVal> * thenVal,
        VTVal elseVal,
        DCTX(ctx)
    ) {
        const size_t numRows = cond->getNumRows();
        const size_t numCols = cond->getNumCols();

        if(
            numRows != thenVal->getNumRows() ||
            numCols != thenVal->getNumCols()
        )
            throw std::runtime_error(
                    "CondMatMatSca: condition/then matrices must have the same shape"
            );

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTVal>>(numRows, numCols, false);

        VTVal * valuesRes = res->getValues();
        const VTCond * valuesCond = cond->getValues();
        const VTVal * valuesThen = thenVal->getValues();
        const size_t rowSkipRes = res->getRowSkip();
        const size_t rowSkipCond = cond->getRowSkip();
        const size_t rowSkipThen = thenVal->getRowSkip();

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++)
                valuesRes[c] = static_cast<bool>(valuesCond[c]) ? valuesThen[c] : elseVal;
            valuesRes += rowSkipRes;
            valuesCond += rowSkipCond;
            valuesThen += rowSkipThen;
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix, Matrix, scalar
// ----------------------------------------------------------------------------

template<typename VTVal, typename VTCond>
struct CondMatMatSca<Matrix<VTVal>, Matrix<VTCond>, Matrix<VTVal>, VTVal> {
    static void apply(
        Matrix<VTVal> *& res,
        const Matrix<VTCond> * cond,
        const Matrix<VTVal> * thenVal,
        VTVal elseVal,
        DCTX(ctx)
    ) {
        const size_t numRows = cond->getNumRows();
        const size_t numCols = cond->getNumCols();

        if (numRows != thenVal->getNumRows() || numCols != thenVal->getNumCols()) {
            std::ostringstream errMsg;
            errMsg << "CondMatMatSca: condition/then matrices must have the same shape but have ("
                    << numRows << "," << numCols << ") and (" 
                    << thenVal->getNumRows() << "," << thenVal->getNumCols() << ")";
            throw std::runtime_error(errMsg.str());
        }

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTVal>>(numRows, numCols, false);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                res->append(r, c, static_cast<bool>(cond->get(r, c)) ? thenVal->get(r, c) : elseVal);
        res->finishAppend();
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_CONDMATMATSCA_H