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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_INSERTROW_H
#define SRC_RUNTIME_LOCAL_KERNELS_INSERTROW_H

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
struct InsertRow {
    static void apply(
        DTArg *& res,
        const DTArg * arg, const DTIns * ins,
        size_t rowLowerIncl, size_t rowUpperExcl,
        DCTX(ctx)
    ) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTArg, class DTIns>
void insertRow(
        DTArg *& res,
        const DTArg * arg, const DTIns * ins,
        size_t rowLowerIncl, size_t rowUpperExcl,
        DCTX(ctx)
) {
    InsertRow<DTArg, DTIns>::apply(res, arg, ins, rowLowerIncl, rowUpperExcl, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct InsertRow<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(
            DenseMatrix<VT> *& res,
            const DenseMatrix<VT> * arg, const DenseMatrix<VT> * ins,
            size_t rowLowerIncl, size_t rowUpperExcl,
            DCTX(ctx)
    ) {
        const size_t numRowsArg = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();
        const size_t numRowsIns = ins->getNumRows();
        const size_t numColsIns = ins->getNumCols();
        
        if(numRowsIns != rowUpperExcl - rowLowerIncl)
            throw std::runtime_error(
                    "insertRow: the number of addressed rows in arg and "
                    "the number of rows in ins must match"
            );
        if(numColsIns != numColsArg)
            throw std::runtime_error(
                    "insertRow: the number of columns in arg and ins must match"
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
        for(size_t r = 0; r < rowLowerIncl; r++) {
            memcpy(valuesRes, valuesArg, numColsArg * sizeof(VT));
            valuesRes += rowSkipRes;
            valuesArg += rowSkipArg;
        }
        for(size_t r = rowLowerIncl; r < rowUpperExcl; r++) {
            memcpy(valuesRes, valuesIns, numColsArg * sizeof(VT));
            valuesRes += rowSkipRes;
            valuesIns += rowSkipIns;
        }
        valuesArg += rowSkipArg * numRowsIns; // skip rows in arg
        for(size_t r = rowUpperExcl; r < numRowsArg; r++) {
            memcpy(valuesRes, valuesArg, numColsArg * sizeof(VT));
            valuesRes += rowSkipRes;
            valuesArg += rowSkipArg;
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_INSERTROW_H