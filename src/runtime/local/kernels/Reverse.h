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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_REVERSE_H
#define SRC_RUNTIME_LOCAL_KERNELS_REVERSE_H

#include "runtime/local/kernels/InPlaceUtils.h"
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <algorithm>
#include <type_traits>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename DTArg>
struct Reverse {
    static void apply(DTRes *& res, DTArg *arg, bool hasFutureUseArg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, typename DTArg>
void reverse(DTRes *& res, DTArg *arg, bool hasFutureUseArg, DCTX(ctx)) {
    Reverse<DTRes, DTArg>::apply(res, arg, hasFutureUseArg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Reverse<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, DenseMatrix<VT> *arg, bool hasFutureUseArg, DCTX(ctx)) {
        size_t numRows = arg->getNumRows();
        size_t numCols = arg->getNumCols();
        
        bool inPlaceUpdate = false;
        if(res == nullptr) {
            if(InPlaceUtils::isInPlaceable(arg, hasFutureUseArg)) {
                res = arg;
                res->increaseRefCounter();
                inPlaceUpdate = true;
            }
            else {
                res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
            }
        }
        
        VT *valuesArg = arg->getValues();
        VT * valuesRes = res->getValues();
        
        if(inPlaceUpdate) {
            if (arg->getRowSkip() == 1) {
                //std::reverse(valuesRes, valuesRes + numRows);
                std::reverse(valuesArg, valuesArg + numRows);
            }
            else {
                VT *valuesArgLastRow = valuesArg + ((numRows - 1) * arg->getRowSkip());
                VT *temp = new VT[numCols];

                for (size_t r = 0; r < numRows / 2; r++) {
                    // Swap rows
                    memcpy(temp, valuesArg, numCols * sizeof(VT));
                    memcpy(valuesArg, valuesArgLastRow, numCols * sizeof(VT));
                    memcpy(valuesArgLastRow, temp, numCols * sizeof(VT));

                    valuesArg += arg->getRowSkip();
                    valuesArgLastRow -= arg->getRowSkip();
                }

                delete[] temp;
            }
        }
        else {
            // This operation will often be applied to column (n x 1) matrices,
            // so this case could optionally be treated more efficiently.
            if (arg->getRowSkip() == 1){ // We need to check RowSkip in case of sub Matrix (see DenseMatrix.h)
                std::reverse_copy(valuesArg, valuesArg + numRows, valuesRes);
            }
            else {
                const VT *valuesArgLastRow = valuesArg + ((numRows - 1) * arg->getRowSkip());            
                for (size_t r = 0; r < numRows; r++) {                                
                    memcpy(valuesRes, valuesArgLastRow, numCols * sizeof(VT));
                    valuesRes += res->getRowSkip();
                    valuesArgLastRow -= arg->getRowSkip();
                }
            }
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_REVERSE_H