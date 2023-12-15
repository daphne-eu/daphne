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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SLICECOL_H
#define SRC_RUNTIME_LOCAL_KERNELS_SLICECOL_H

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
struct SliceCol {
    static void apply(DTRes *& res, const DTArg * arg, VTSel lowerIncl, VTSel upperExcl, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg, typename VTSel>
void sliceCol(DTRes *& res, const DTArg * arg, VTSel lowerIncl, VTSel upperExcl, DCTX(ctx)) {
    SliceCol<DTRes, DTArg, VTSel>::apply(res, arg, lowerIncl, upperExcl, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// verifies 0 <= lowerIncl <= upperExcl <= numColsArg
#define CHECK_BOUNDARY(lowerIncl, upperExcl, DT) \
    const size_t numColsArg = arg->getNumCols(); \
    if (lowerIncl < 0 || upperExcl < lowerIncl || numColsArg < static_cast<size_t>(upperExcl)) { \
            std::ostringstream errMsg; \
            errMsg << "invalid arguments '[..., [" << lowerIncl << "," << upperExcl << "]]' passed to SliceCol on " << DT << " with column boundaries '[0, " << numColsArg << "]'"; \
            throw std::out_of_range(errMsg.str()); \
        }

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTArg, typename VTSel>
struct SliceCol<DenseMatrix<VTArg>, DenseMatrix<VTArg>, VTSel> {
    static void apply(DenseMatrix<VTArg> *& res, const DenseMatrix<VTArg> * arg, VTSel lowerIncl, VTSel upperExcl, DCTX(ctx)) {
        CHECK_BOUNDARY(lowerIncl, upperExcl, "dense matrix");
        res = arg->sliceCol(static_cast<const size_t>(lowerIncl), static_cast<const size_t>(upperExcl));
    }        
};

// ----------------------------------------------------------------------------
// Frame <- Frame
// ----------------------------------------------------------------------------

template <typename VTSel>
struct SliceCol<Frame, Frame, VTSel> {
    static void apply(Frame *& res, const Frame * arg, VTSel lowerIncl, VTSel upperExcl, DCTX(ctx)) {
        CHECK_BOUNDARY(lowerIncl, upperExcl, "frame");
        res = arg->sliceCol(static_cast<const size_t>(lowerIncl), static_cast<const size_t>(upperExcl));
    }        
};

#undef CHECK_BOUNDARY
#endif //SRC_RUNTIME_LOCAL_KERNELS_SLICECOL_H
