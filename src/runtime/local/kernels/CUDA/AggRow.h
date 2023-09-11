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
#include <runtime/local/kernels/AggOpCode.h>

namespace CUDA {

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

    template<class DTRes, class DTArg>
    struct AggRow {
        static void apply(AggOpCode opCode, DenseMatrix<DTRes> *&res, const DenseMatrix<DTArg> *arg, DCTX(ctx));
    };

// ****************************************************************************
// Convenience function
// ****************************************************************************

    template<class DTRes, class DTArg>
    void aggRow(AggOpCode opCode, DTRes *&res, const DTArg *arg, DCTX(ctx)) {
        AggRow<DTRes, DTArg>::apply(opCode, res, arg, ctx);
    }

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

    template<typename VT>
    struct AggRow<DenseMatrix<VT>, DenseMatrix<VT>> {
        static void apply(AggOpCode opCode, DenseMatrix<VT> *&res, const DenseMatrix<VT> *arg, DCTX(ctx));
    };
}