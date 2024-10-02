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
#include <string>

namespace CUDA {
template <class DTRes, class DTLhs, class DTRhs> struct ColBind {
    static void apply(DTRes *&res, const DTLhs *lhs, const DTRhs *rhs, DCTX(ctx));
};

template <typename VTres, typename VTlhs, typename VTrhs>
struct ColBind<DenseMatrix<VTres>, DenseMatrix<VTlhs>, DenseMatrix<VTrhs>> {
    static void apply(DenseMatrix<VTres> *&res, const DenseMatrix<VTlhs> *lhs, const DenseMatrix<VTrhs> *rhs,
                      DCTX(ctx));
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template <class DTRes, class DTLhs, class DTRhs>
void colBind(DTRes *&res, const DTLhs *lhs, const DTRhs *rhs, DCTX(ctx)) {
    ColBind<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, ctx);
}
} // namespace CUDA