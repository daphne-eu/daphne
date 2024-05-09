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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_FILTERCOL_H
#define SRC_RUNTIME_LOCAL_KERNELS_FILTERCOL_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>

#include <stdexcept>

#include <cstddef>
#include <cstdint>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg, typename VTSel>
struct FilterCol {
    static void apply(DTRes *& res, const DTArg * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg, typename VTSel>
void filterCol(DTRes *& res, const DTArg * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) {
    FilterCol<DTRes, DTArg, VTSel>::apply(res, arg, sel, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT, typename VTSel>
struct FilterCol<DenseMatrix<VT>, DenseMatrix<VT>, VTSel> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();

        if(sel->getNumRows() != numColsArg)
            throw std::runtime_error("sel must have exactly one entry (row) for each column in arg");
        if(sel->getNumCols() != 1)
            throw std::runtime_error("sel must be a single-column matrix");

        size_t numColsRes = 0;
        for(size_t c = 0; c < numColsArg; c++)
            numColsRes += sel->get(c, 0);

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numColsRes, false);

        const VT * valuesArg = arg->getValues();
        VT * valuesRes = res->getValues();
        const size_t rowSkipArg = arg->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();
        for(size_t r = 0; r < numRows; r++) {
            for(size_t ca = 0, cr = 0; ca < numColsArg; ca++)
                if(sel->get(ca, 0))
                    valuesRes[cr++] = valuesArg[ca];
            valuesArg += rowSkipArg;
            valuesRes += rowSkipRes;
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template<typename VT, typename VTSel>
struct FilterCol<Matrix<VT>, Matrix<VT>, VTSel> {
    static void apply(Matrix<VT> *& res, const Matrix<VT> * arg, const Matrix<VTSel> * sel, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();

        if (sel->getNumRows() != numColsArg)
            throw std::runtime_error("sel must have exactly one entry (row) for each column in arg");
        if (sel->getNumCols() != 1)
            throw std::runtime_error("sel must be a single-column matrix");

        size_t numColsRes = 0;
        for (size_t c = 0; c < numColsArg; ++c)
            numColsRes += sel->get(c, 0);

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numColsRes, false);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r) {
            for (size_t cArg = 0, cRes = 0; cArg < numColsArg; ++cArg)
                if (sel->get(cArg, 0))
                    res->append(r, cRes++, arg->get(r, cArg));
        }
        res->finishAppend();
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_FILTERCOL_H