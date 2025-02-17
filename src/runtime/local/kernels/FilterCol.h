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

template <class DTRes, class DTArg, typename VTSel> struct FilterCol {
    static void apply(DTRes *&res, const DTArg *arg, const DenseMatrix<VTSel> *sel, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg, typename VTSel>
void filterCol(DTRes *&res, const DTArg *arg, const DenseMatrix<VTSel> *sel, DCTX(ctx)) {
    FilterCol<DTRes, DTArg, VTSel>::apply(res, arg, sel, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT, typename VTSel> struct FilterCol<DenseMatrix<VT>, DenseMatrix<VT>, VTSel> {
    static void apply(DenseMatrix<VT> *&res, const DenseMatrix<VT> *arg, const DenseMatrix<VTSel> *sel, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();

        if (sel->getNumRows() != numColsArg)
            throw std::runtime_error("sel must have exactly one entry (row) for each column in arg");
        if (sel->getNumCols() != 1)
            throw std::runtime_error("sel must be a single-column matrix");

        size_t numColsRes = 0;
        for (size_t c = 0; c < numColsArg; c++)
            numColsRes += sel->get(c, 0);

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numColsRes, false);

        const VT *valuesArg = arg->getValues();
        VT *valuesRes = res->getValues();
        const size_t rowSkipArg = arg->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();

        // Two alternative approaches for doing the main work.
        // Note that even though sel is essentially a bit vector, we represent it as a 64-bit integer at the moment, for
        // simplicity. As both the elements in sel used in approach 1 and the positions in approach 2 are currently
        // represented as 64-bit integers, approach 2 should always be faster than approach 1, because in approach 2 we
        // iterate over at most as many 64-bit values as in approach 1 while we can omit the check. Once we change to a
        // 1-bit representation for the values in sel, we should rethink the trade-off between the two approaches.
#if 0 // approach 1
      // For every row in arg, iterate over all elements in sel (one per column in arg), check if the respective
      // column should be part of the output and if so, copy the value to the output.
        for (size_t r = 0; r < numRows; r++) {
            for (size_t ca = 0, cr = 0; ca < numColsArg; ca++)
                if (sel->get(ca, 0))
                    valuesRes[cr++] = valuesArg[ca];
            valuesArg += rowSkipArg;
            valuesRes += rowSkipRes;
        }
#else // approach 2
      // Once in the beginning, create a vector of the positions of the columns we want to copy to the output.
      // Negligible effort, unless the number of rows in arg is very small.
        std::vector<size_t> positions;
        for (size_t c = 0; c < numColsArg; c++)
            if (sel->get(c, 0))
                positions.push_back(c);
        // For every row in arg, iterate over the array of those positions and copy the value in the respective column
        // to the output.
        for (size_t r = 0; r < numRows; r++) {
            for (size_t i = 0; i < positions.size(); i++)
                valuesRes[i] = valuesArg[positions[i]];
            valuesArg += rowSkipArg;
            valuesRes += rowSkipRes;
        }
#endif
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template <typename VT, typename VTSel> struct FilterCol<Matrix<VT>, Matrix<VT>, VTSel> {
    static void apply(Matrix<VT> *&res, const Matrix<VT> *arg, const Matrix<VTSel> *sel, DCTX(ctx)) {
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

#endif // SRC_RUNTIME_LOCAL_KERNELS_FILTERCOL_H