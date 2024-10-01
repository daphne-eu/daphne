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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_MATRIXCONSTANT_H
#define SRC_RUNTIME_LOCAL_KERNELS_MATRIXCONSTANT_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template <class DTRes> struct MatrixConstant {
    static void apply(DTRes *&res, uint64_t matrixAddr, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes> void matrixConstant(DTRes *&res, uint64_t matrixAddr, DCTX(ctx)) {
    MatrixConstant<DTRes>::apply(res, matrixAddr, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct MatrixConstant<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, uint64_t matrixAddr, DCTX(ctx)) {
        // We create a copy of the DenseMatrix backing the matrix literal.
        // This is important since the matrix literal may be used inside a loop
        // with multiple iterations or inside a function with multiple
        // invocations. If we handed out the original DenseMatrix, it would be
        // freed by DAPHNE's garbage collection by the end of the
        // loop's/function's body.

        // TODO Currently, the original DenseMatrix objects created by the
        // parser are never freed, which is a memory leak. However, since matrix
        // literals should be used only for tiny matrices, the problem is not
        // significant. They will be freed automatically at the end of the
        // DAPHNE process. However, in long-running distributed workers these
        // matrix objects might pile up over time.

        DenseMatrix<VT> *orig = reinterpret_cast<DenseMatrix<VT> *>(matrixAddr);
        const size_t numRows = orig->getNumRows();
        const size_t numCols = orig->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        const VT *valuesOrig = orig->getValues();
        VT *valuesRes = res->getValues();

        memcpy(valuesRes, valuesOrig, numRows * numCols * sizeof(VT));
    }
};
#endif // SRC_RUNTIME_LOCAL_KERNELS_MATRIXCONSTANT_H
