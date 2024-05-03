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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SLICEROW_H
#define SRC_RUNTIME_LOCAL_KERNELS_SLICEROW_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <sstream>
#include <stdexcept>

#include <cstddef>
#include <cstdint>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg, typename VTSel>
struct SliceRow {
    static void apply(DTRes *& res, const DTArg * arg, const VTSel lowerIncl, const VTSel upperExcl, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg, typename VTSel>
void sliceRow(DTRes *& res, const DTArg * arg, const VTSel lowerIncl, const VTSel upperExcl, DCTX(ctx)) {
    SliceRow<DTRes, DTArg, VTSel>::apply(res, arg, lowerIncl, upperExcl, ctx);
}

// ****************************************************************************
// Boundary validation
// ****************************************************************************

template<typename VTSel>
void validateArgsSliceRow(VTSel lowerIncl, VTSel upperExcl, size_t numRowsArg) {
    if (lowerIncl < 0 || upperExcl < lowerIncl || numRowsArg < static_cast<size_t>(upperExcl)
        || (static_cast<size_t>(lowerIncl) == numRowsArg && lowerIncl != 0)) {
            std::ostringstream errMsg;
            errMsg << "invalid arguments '" << lowerIncl << ", " << upperExcl << "' passed to SliceRow: "
                    << "it must hold 0 <= lowerIncl <= upperExcl <= #rows "
                    << "and lowerIncl < #rows (unless both are zero) where #rows of arg is '" << numRowsArg << "'";
            throw std::out_of_range(errMsg.str());
        }
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTArg, typename VTSel>
struct SliceRow<DenseMatrix<VTArg>, DenseMatrix<VTArg>, VTSel> {
    static void apply(DenseMatrix<VTArg> *& res, const DenseMatrix<VTArg> * arg, const VTSel lowerIncl, const VTSel upperExcl, DCTX(ctx)) {
        const size_t numRowsArg = arg->getNumRows();
        validateArgsSliceRow(lowerIncl, upperExcl, numRowsArg);
        res = arg->sliceRow(lowerIncl, upperExcl);
    }        
};

// ----------------------------------------------------------------------------
// Frame <- Frame
// ----------------------------------------------------------------------------

template <typename VTSel>
struct SliceRow<Frame, Frame, VTSel> {
    static void apply(Frame *& res, const Frame * arg, const VTSel lowerIncl, const VTSel upperExcl, DCTX(ctx)) {
        const size_t numRowsArg = arg->getNumRows();
        validateArgsSliceRow(lowerIncl, upperExcl, numRowsArg);
        res = arg->sliceRow(lowerIncl, upperExcl);
    }        
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template<typename VTArg, typename VTSel>
struct SliceRow<Matrix<VTArg>, Matrix<VTArg>, VTSel> {
    static void apply(Matrix<VTArg> *& res, const Matrix<VTArg> * arg, const VTSel lowerIncl, const VTSel upperExcl, DCTX(ctx)) {
        const size_t numColsArg = arg->getNumCols();
        const size_t numRowsRes = static_cast<const size_t>(upperExcl - lowerIncl);
        validateArgsSliceRow(lowerIncl, upperExcl, arg->getNumRows());

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTArg>>(numRowsRes, numColsArg, false);

        res->prepareAppend();
        for (size_t r = 0; r < numRowsRes; ++r)
            for (size_t c = 0; c < numColsArg; ++c)
                res->append(r, c, arg->get(static_cast<const size_t>(lowerIncl) + r, c));
        res->finishAppend();
    }        
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SLICEROW_H