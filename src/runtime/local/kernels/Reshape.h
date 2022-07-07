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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_RESHAPE_H
#define SRC_RUNTIME_LOCAL_KERNELS_RESHAPE_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <stdexcept>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct Reshape {
    static void apply(DTRes *& res, const DTArg * arg, size_t numRows, size_t numCols, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void reshape(DTRes *& res, const DTArg * arg, size_t numRows, size_t numCols, DCTX(ctx)) {
    Reshape<DTRes, DTArg>::apply(res, arg, numRows, numCols, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Reshape<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, size_t numRows, size_t numCols, DCTX(ctx)) {
        if(numRows * numCols != arg->getNumRows() * arg->getNumCols())
            throw std::runtime_error("reshape must retain the number of cells");

        if(arg->getRowSkip() == arg->getNumCols() && res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, arg->getValuesSharedPtr());
        else {
            if(res == nullptr)
                res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

            auto resVals = res->getValues();
            auto argVals = arg->getValues();
            size_t numArgRows = arg->getNumRows();
            size_t numArgCols = arg->getNumCols();
            for(size_t r = 0; r < numArgRows; r++) {
                memcpy(resVals, argVals, numArgCols * sizeof(VT));
                argVals += arg->getRowSkip();
                resVals += numArgCols;
            }
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_RESHAPE_H