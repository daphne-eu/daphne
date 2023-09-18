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
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/AggOpCode.h>

namespace CUDA {
// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

    template<typename VTRes, class DTArg>
    struct AggAll {
        static VTRes apply(AggOpCode opCode, const DTArg *arg, DCTX(dctx)) = delete;
    };

// ****************************************************************************
// Convenience function
// ****************************************************************************

    template<typename VTRes, class DTArg>
    VTRes aggAll(AggOpCode opCode, const DTArg *arg, DCTX(dctx)) {
        return AggAll<VTRes, DTArg>::apply(opCode, arg, dctx);
    }

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// scalar <- DenseMatrix
// ----------------------------------------------------------------------------

    template<typename VTRes, typename VTArg>
    struct AggAll<VTRes, DenseMatrix<VTArg>> {
        static VTRes apply(AggOpCode opCode, const DenseMatrix<VTArg> *arg, DCTX(dctx));
    };
}