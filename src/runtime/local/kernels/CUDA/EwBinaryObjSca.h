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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/BinaryOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <cassert>
#include <cstddef>
#include <string>

#pragma once

namespace CUDA {
// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

    template<class DTRes, class DTLhs, typename VTRhs>
    struct EwBinaryObjSca {
        static void apply(BinaryOpCode opCode, DTRes *& res, DTLhs * lhs, VTRhs rhs, bool hasFutureUseLhs, DCTX(ctx)) = delete;
    };

// ****************************************************************************
// Convenience function
// ****************************************************************************

    template<class DTRes, class DTLhs, typename VTRhs>
    void ewBinaryObjSca(BinaryOpCode opCode, DTRes *& res, DTLhs * lhs, VTRhs rhs, bool hasFutureUseLhs, DCTX(ctx)) {
        EwBinaryObjSca<DTRes, DTLhs, VTRhs>::apply(opCode, res, lhs, rhs, hasFutureUseLhs, ctx);
    }

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, scalar
// ----------------------------------------------------------------------------

    template<typename VT>
    struct EwBinaryObjSca<DenseMatrix<VT>, DenseMatrix<VT>, VT> {
        static void apply(BinaryOpCode opCode, DenseMatrix<VT> *& res, DenseMatrix<VT> * lhs, VT rhs, bool hasFutureUseLhs, DCTX(ctx));
    };
}