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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <cstddef>

namespace CUDA {
// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTLhs, class DTRhs> struct Solve {
    static void apply(DTRes *&res, const DTLhs *lhs, const DTRhs *rhs,
                      DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTLhs, class DTRhs>
void solve(DTRes *&res, const DTLhs *lhs, const DTRhs *rhs, DCTX(ctx)) {
    Solve<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, ctx);
}

template <typename VT>
struct Solve<DenseMatrix<VT>, DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, const DenseMatrix<VT> *lhs,
                      const DenseMatrix<VT> *rhs, DCTX(ctx));
};
} // namespace CUDA