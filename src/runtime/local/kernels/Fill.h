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

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename VTArg>
struct Fill {
    static void apply(DTRes *& res, VTArg arg, size_t numRows, size_t numCols, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, typename VTArg>
void fill(DTRes *& res, VTArg arg, size_t numRows, size_t numCols, DCTX(ctx)) {
    Fill<DTRes, VTArg>::apply(res, arg, numRows, numCols, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Fill<DenseMatrix<VT>, VT> {
    static void apply(DenseMatrix<VT> *& res, VT arg, size_t numRows, size_t numCols, DCTX(ctx)) {

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, arg == 0);

        if(arg != 0) {
            VT *valuesRes = res->getValues();
            for(auto i = 0ul; i < res->getNumItems(); ++i)
                valuesRes[i] = arg;
        }
    }
};
