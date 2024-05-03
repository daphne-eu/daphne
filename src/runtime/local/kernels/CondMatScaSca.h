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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_CONDMATSCASCA_H
#define SRC_RUNTIME_LOCAL_KERNELS_CONDMATSCASCA_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>

#include <stdexcept>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTCond, class VTThen, class VTElse>
struct CondMatScaSca {
    static void apply(DTRes *& res, const DTCond * cond, VTThen thenVal, VTElse elseVal, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTCond, class VTThen, class VTElse>
void condMatScaSca(DTRes *& res, const DTCond * cond, VTThen thenVal, VTElse elseVal, DCTX(ctx)) {
    CondMatScaSca<DTRes, DTCond, VTThen, VTElse>::apply(res, cond, thenVal, elseVal, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, scalar, scalar
// ----------------------------------------------------------------------------

template<typename VTVal, typename VTCond>
struct CondMatScaSca<DenseMatrix<VTVal>, DenseMatrix<VTCond>, VTVal, VTVal> {
    static void apply(
        DenseMatrix<VTVal> *& res,
        const DenseMatrix<VTCond> * cond,
        VTVal thenVal,
        VTVal elseVal,
        DCTX(ctx)
    ) {
        const size_t numRows = cond->getNumRows();
        const size_t numCols = cond->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTVal>>(numRows, numCols, false);

        VTVal * valuesRes = res->getValues();
        const VTCond * valuesCond = cond->getValues();
        const size_t rowSkipRes = res->getRowSkip();
        const size_t rowSkipCond = cond->getRowSkip();

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++)
                valuesRes[c] = static_cast<bool>(valuesCond[c]) ? thenVal : elseVal;
            valuesRes += rowSkipRes;
            valuesCond += rowSkipCond;
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix, scalar, scalar
// ----------------------------------------------------------------------------

template<typename VTVal, typename VTCond>
struct CondMatMatSca<Matrix<VTVal>, Matrix<VTCond>, VTVal, VTVal> {
    static void apply(
        Matrix<VTVal> *& res,
        const Matrix<VTCond> * cond,
        VTVal thenVal,
        VTVal elseVal,
        DCTX(ctx)
    ) {
        const size_t numRows = cond->getNumRows();
        const size_t numCols = cond->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTVal>>(numRows, numCols, false);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                res->append(r, c, static_cast<bool>(cond->get(r, c)) ? thenVal : elseVal);
        res->finishAppend();
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_CONDMATSCASCA_H