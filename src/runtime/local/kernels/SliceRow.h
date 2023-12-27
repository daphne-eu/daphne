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

// verifies 0 <= lowerIncl <= upperExcl <= numRowsArg
#define VALIDATE_ARGS(lowerIncl, upperExcl, DT) \
    const size_t numRowsArg = arg->getNumRows(); \
    if (lowerIncl < 0 || upperExcl < lowerIncl || numRowsArg < static_cast<const size_t>(upperExcl)) { \
            std::ostringstream errMsg; \
            errMsg << "invalid arguments '[[" << lowerIncl << "," << upperExcl << "], ...]' passed to SliceRow on " << DT << " with row boundaries '[0, " << numRowsArg << "]'"; \
            throw std::out_of_range(errMsg.str()); \
        }

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTArg, typename VTSel>
struct SliceRow<DenseMatrix<VTArg>, DenseMatrix<VTArg>, VTSel> {
    static void apply(DenseMatrix<VTArg> *& res, const DenseMatrix<VTArg> * arg, VTSel lowerIncl, VTSel upperExcl, DCTX(ctx)) {
        VALIDATE_ARGS(lowerIncl, upperExcl, "dense matrix");
        res = arg->sliceRow(static_cast<const size_t>(lowerIncl), static_cast<const size_t>(upperExcl));
    }        
};

// ----------------------------------------------------------------------------
// Frame <- Frame
// ----------------------------------------------------------------------------

template <typename VTSel>
struct SliceRow<Frame, Frame, VTSel> {
    static void apply(Frame *& res, const Frame * arg, VTSel lowerIncl, VTSel upperExcl, DCTX(ctx)) {
        VALIDATE_ARGS(lowerIncl, upperExcl, "frame");
        res = arg->sliceRow(static_cast<const size_t>(lowerIncl), static_cast<const size_t>(upperExcl));
    }        
};

#undef VALIDATE_ARGS

#endif //SRC_RUNTIME_LOCAL_KERNELS_SLICEROW_H