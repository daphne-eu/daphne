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
#include <runtime/local/datastructures/DenseMatrix.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<class DTRes>
struct MatrixConstant {
    static void apply(DTRes *& res, uint64_t matrixAddr, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes>
void matrixConstant(DTRes *& res, uint64_t matrixAddr, DCTX(ctx)) { 
    MatrixConstant<DTRes>::apply(res, matrixAddr, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct MatrixConstant<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, uint64_t matrixAddr, DCTX(ctx)) {
        res = reinterpret_cast<DenseMatrix<VT>*>(matrixAddr);
    }
};
#endif //SRC_RUNTIME_LOCAL_KERNELS_MATRIXCONSTANT_H
