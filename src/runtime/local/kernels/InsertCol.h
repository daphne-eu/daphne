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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_INSERTCOL_H
#define SRC_RUNTIME_LOCAL_KERNELS_INSERTCOL_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>

#include <sstream>
#include <stdexcept>

#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg, class DTIns, typename VTSel> struct InsertCol {
    static void apply(DTArg *&res, const DTArg *arg, const DTIns *ins, const VTSel colLowerIncl,
                      const VTSel colUpperExcl, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg, class DTIns, typename VTSel>
void insertCol(DTArg *&res, const DTArg *arg, const DTIns *ins, const VTSel colLowerIncl, const VTSel colUpperExcl,
               DCTX(ctx)) {
    InsertCol<DTArg, DTIns, VTSel>::apply(res, arg, ins, colLowerIncl, colUpperExcl, ctx);
}

// ****************************************************************************
// Boundary validation
// ****************************************************************************

template <typename VTSel>
void validateArgsInsertCol(size_t colLowerIncl_Pos, VTSel colLowerIncl, size_t colUpperExcl_Pos, VTSel colUpperExcl,
                           size_t numRowsArg, size_t numColsArg, size_t numRowsIns, size_t numColsIns) {

    if (colUpperExcl_Pos < colLowerIncl_Pos || numColsArg < colUpperExcl_Pos ||
        (colLowerIncl_Pos == numColsArg && colLowerIncl_Pos != 0)) {
        std::ostringstream errMsg;
        errMsg << "invalid arguments '" << colLowerIncl << ", " << colUpperExcl
               << "' passed to InsertCol: it must hold 0 <= colLowerIncl <= "
                  "colUpperExcl <= #columns "
               << "and colLowerIncl < #columns (unless both are zero) where "
                  "#columns of arg is '"
               << numColsArg << "'";
        throw std::out_of_range(errMsg.str());
    }

    if (numColsIns != colUpperExcl_Pos - colLowerIncl_Pos) {
        std::ostringstream errMsg;
        errMsg << "invalid arguments '" << colLowerIncl << ", " << colUpperExcl
               << "' passed to InsertCol: the number of addressed columns in arg '"
               << colUpperExcl_Pos - colLowerIncl_Pos << "' and the number of columns in ins '" << numColsIns
               << "' must match";
        throw std::out_of_range(errMsg.str());
    }

    if (numRowsIns != numRowsArg) {
        std::ostringstream errMsg;
        errMsg << "invalid arguments passed to InsertCol: the number of rows "
                  "in arg '"
               << numRowsArg << "' and ins '" << numRowsIns << "' must match";
        throw std::out_of_range(errMsg.str());
    }
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTArg, typename VTSel> struct InsertCol<DenseMatrix<VTArg>, DenseMatrix<VTArg>, VTSel> {
    static void apply(DenseMatrix<VTArg> *&res, const DenseMatrix<VTArg> *arg, const DenseMatrix<VTArg> *ins,
                      VTSel colLowerIncl, VTSel colUpperExcl, DCTX(ctx)) {
        const size_t numRowsArg = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();
        const size_t numRowsIns = ins->getNumRows();
        const size_t numColsIns = ins->getNumCols();

        // Resolve negative indices here only
        size_t colLowerIncl_Pos = static_cast<size_t>(colLowerIncl);
        size_t colUpperExcl_Pos = static_cast<size_t>(colUpperExcl);
        if constexpr (std::is_signed<VTSel>::value) {
            if (colLowerIncl < 0)
                colLowerIncl_Pos =
                    static_cast<size_t>(static_cast<ptrdiff_t>(numColsArg) + static_cast<ptrdiff_t>(colLowerIncl));
            if (colUpperExcl < 0)
                colUpperExcl_Pos =
                    static_cast<size_t>(static_cast<ptrdiff_t>(numColsArg) + static_cast<ptrdiff_t>(colUpperExcl));
            if (colLowerIncl < 0 && colUpperExcl == 0)
                colUpperExcl_Pos = colLowerIncl_Pos + 1;
        }

        validateArgsInsertCol(colLowerIncl_Pos, colLowerIncl, colUpperExcl_Pos, colUpperExcl, numRowsArg, numColsArg,
                              numRowsIns, numColsIns);

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTArg>>(numRowsArg, numColsArg, false);

        VTArg *valuesRes = res->getValues();
        const VTArg *valuesArg = arg->getValues();
        const VTArg *valuesIns = ins->getValues();
        const size_t rowSkipRes = res->getRowSkip();
        const size_t rowSkipArg = arg->getRowSkip();
        const size_t rowSkipIns = ins->getRowSkip();

        // TODO Can be simplified/more efficient in certain cases.
        for (size_t r = 0; r < numRowsArg; r++) {
            std::copy(valuesArg, valuesArg + colLowerIncl_Pos, valuesRes);
            std::copy(valuesIns, valuesIns + numColsIns, valuesRes + colLowerIncl_Pos);
            std::copy(valuesArg + colUpperExcl_Pos, valuesArg + numColsArg, valuesRes + colUpperExcl_Pos);
            valuesRes += rowSkipRes;
            valuesArg += rowSkipArg;
            valuesIns += rowSkipIns;
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template <typename VTArg, typename VTSel> struct InsertCol<Matrix<VTArg>, Matrix<VTArg>, VTSel> {
    static void apply(Matrix<VTArg> *&res, const Matrix<VTArg> *arg, const Matrix<VTArg> *ins, VTSel colLowerIncl,
                      VTSel colUpperExcl, DCTX(ctx)) {
        const size_t numRowsArg = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();

        // Resolve negative indices here only
        size_t colLowerIncl_Pos = static_cast<size_t>(colLowerIncl);
        size_t colUpperExcl_Pos = static_cast<size_t>(colUpperExcl);
        if constexpr (std::is_signed<VTSel>::value) {
            if (colLowerIncl < 0)
                colLowerIncl_Pos =
                    static_cast<size_t>(static_cast<ptrdiff_t>(numColsArg) + static_cast<ptrdiff_t>(colLowerIncl));
            if (colUpperExcl < 0)
                colUpperExcl_Pos =
                    static_cast<size_t>(static_cast<ptrdiff_t>(numColsArg) + static_cast<ptrdiff_t>(colUpperExcl));
            if (colLowerIncl < 0 && colUpperExcl == 0)
                colUpperExcl_Pos = colLowerIncl_Pos + 1;
        }

        validateArgsInsertCol(colLowerIncl_Pos, colLowerIncl, colUpperExcl_Pos, colUpperExcl, numRowsArg, numColsArg,
                              ins->getNumRows(), ins->getNumCols());

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTArg>>(numRowsArg, numColsArg, false);

        res->prepareAppend();
        for (size_t r = 0; r < numRowsArg; ++r) {
            for (size_t c = 0; c < colLowerIncl_Pos; ++c)
                res->append(r, c, arg->get(r, c));
            for (size_t c = colLowerIncl_Pos; c < colUpperExcl_Pos; ++c)
                res->append(r, c, ins->get(r, c - colLowerIncl_Pos));
            for (size_t c = colUpperExcl_Pos; c < numColsArg; ++c)
                res->append(r, c, arg->get(r, c));
        }
        res->finishAppend();
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_INSERTROW_H
