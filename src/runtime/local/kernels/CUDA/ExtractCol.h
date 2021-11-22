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

#include <cassert>
#include <cstddef>
#include <string>

namespace CUDA {
    template<class DTRes, class DTArg, class DTSel>
    struct ExtractCol {
        static void apply(DTRes *&res, const DTArg *arg, const DTSel *sel, DCTX(ctx)) = delete;
    };

    template<class DTRes, class DTArg, class DTSel>
    struct ExtractCol<DenseMatrix<DTRes>, DenseMatrix<DTArg>, DenseMatrix<DTSel>> {
        static void
        apply(DenseMatrix<DTRes> *&res, const DenseMatrix<DTArg> *arg, const DenseMatrix<DTSel> *sel, DCTX(ctx));
    };


// ****************************************************************************
// Convenience function
// ****************************************************************************
    template<class DTRes, class DTArg, class DTSel>
    void extractCol(DTRes *&res, const DTArg *arg, const DTSel *sel, DCTX(ctx)) {
        ExtractCol<DTRes, DTArg, DTSel>::apply(res, arg, sel, ctx);
    }
}