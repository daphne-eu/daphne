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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_INSERTCOL_H
#define SRC_RUNTIME_LOCAL_KERNELS_INSERTCOL_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <stdexcept>

#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTArg, class DTIns>
struct InsertCol {
    static void apply(
            DTArg *& res,
            const DTArg * arg, const DTIns * ins,
            size_t colLowerIncl, size_t colUpperExcl,
            DCTX(ctx)
    ) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTArg, class DTIns>
void insertCol(
        DTArg *& res,
        const DTArg * arg, const DTIns * ins,
        size_t colLowerIncl, size_t colUpperExcl,
        DCTX(ctx)
) {
    InsertCol<DTArg, DTIns>::apply(res, arg, ins, colLowerIncl, colUpperExcl, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct InsertCol<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(
            DenseMatrix<VT> *& res,
            const DenseMatrix<VT> * arg, const DenseMatrix<VT> * ins,
            size_t colLowerIncl, size_t colUpperExcl,
            DCTX(ctx)
    ) {
        const size_t numRowsArg = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();
        const size_t numRowsIns = ins->getNumRows();
        const size_t numColsIns = ins->getNumCols();
        
        if(numRowsIns != numRowsArg)
            throw std::runtime_error(
                    "insertCol: the number of rows in arg and ins must match"
            );
        if(numColsIns != colUpperExcl - colLowerIncl)
            throw std::runtime_error(
                    "insertCol: the number of addressed columns in arg and §"
                    "the number of columns in ins must match"
            );
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRowsArg, numColsArg, false);
        
        VT * valuesRes = res->getValues();
        const VT * valuesArg = arg->getValues();
        const VT * valuesIns = ins->getValues();
        const size_t rowSkipRes = res->getRowSkip();
        const size_t rowSkipArg = arg->getRowSkip();
        const size_t rowSkipIns = ins->getRowSkip();
        
        // TODO Can be simplified/more efficient in certain cases.
        for(size_t r = 0; r < numRowsArg; r++) {
            memcpy(valuesRes, valuesArg, colLowerIncl * sizeof(VT));
            memcpy(valuesRes + colLowerIncl, valuesIns, numColsIns * sizeof(VT));
            memcpy(valuesRes + colUpperExcl, valuesArg + colUpperExcl, (numColsArg - colUpperExcl) * sizeof(VT));
            valuesRes += rowSkipRes;
            valuesArg += rowSkipArg;
            valuesIns += rowSkipIns;
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_INSERTROW_H