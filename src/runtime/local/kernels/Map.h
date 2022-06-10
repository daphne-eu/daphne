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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAPOP_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAPOP_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <algorithm>

#include <cassert>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct Map {
    static void apply(DTRes *& res, const DTArg * arg , void* func, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void map(DTRes *& res, const DTArg * arg, void* func, DCTX(ctx)) {
    Map<DTRes, DTArg>::apply(res, arg, func, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
struct Map<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, void* func, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        // TODO_198 when is res != nullptr the case?
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);
        else if (res->getNumRows() != numRows || res->getNumCols() != numCols)
            throw std::runtime_error("res and arg must have the same shape");
        
        auto udf = reinterpret_cast<VTRes(*)(VTArg)>(func);

        const VTArg * valuesArg = arg->getValues();
        VTRes * valuesRes = res->getValues();

        // TODO_198 Regarding getRowSkip: is this correct?
        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++)
                valuesRes[c] = udf(valuesArg[c]);
            valuesArg += arg->getRowSkip();
            valuesRes += res->getRowSkip();
        }
    }
};
#endif //SRC_RUNTIME_LOCAL_KERNELS_MAPOP_H