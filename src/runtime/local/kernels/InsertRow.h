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

#include <sstream>
#include <stdexcept>

#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTArg, class DTIns, typename VTSel>
struct InsertRow {
    static void apply(
        DTArg *& res,
        const DTArg * arg, const DTIns * ins,
        VTSel rowLowerIncl, VTSel rowUpperExcl,
        DCTX(ctx)
    ) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTArg, class DTIns, typename VTSel>
void insertRow(
        DTArg *& res,
        const DTArg * arg, const DTIns * ins,
        VTSel rowLowerIncl, VTSel rowUpperExcl,
        DCTX(ctx)
) {
    InsertRow<DTArg, DTIns, VTSel>::apply(res, arg, ins, rowLowerIncl, rowUpperExcl, ctx);
}

// ****************************************************************************
// Boundary validation function
// ****************************************************************************

template<typename VTSel>
void validateInsertRowArgs(const size_t rowLowerIncl, VTSel rowLowerInclRaw, const size_t rowUpperExcl, VTSel rowUpperExclRaw,
                    const size_t numRowsArg, const size_t numColsArg, const size_t numRowsIns, const size_t numColsIns) {

    if (rowLowerInclRaw < 0 || rowUpperExclRaw < rowLowerInclRaw || numRowsArg < rowUpperExcl) {
        std::ostringstream errMsg;
        errMsg << "invalid arguments '" << rowLowerInclRaw << ", " << rowUpperExclRaw
                << "' passed to InsertRow: must be positive, rowLowerIncl must be smaller than rowUpperExcl "
                << "and both within rows of arg '" << numRowsArg << "'";
        throw std::out_of_range(errMsg.str());
    }

    if(numRowsIns != rowUpperExcl - rowLowerIncl){
        std::ostringstream errMsg;
        errMsg << "invalid arguments '" << rowLowerInclRaw << ", " << rowUpperExclRaw
                << "' passed to InsertRow: the number of addressed rows in arg '" << rowUpperExcl - rowLowerIncl
                << "' and the number of rows in ins '" << numRowsIns << "' must match";
        throw std::runtime_error(errMsg.str());
    }

    if(numColsIns != numColsArg) {
        std::ostringstream errMsg;
        errMsg << "invalid arguments passed to InsertRow: the number of columns in arg '" << numColsArg
                << "' and ins '" << numColsIns << "' must match";
        throw std::runtime_error(errMsg.str());
    }
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT, typename VTSel>
struct InsertRow<DenseMatrix<VT>, DenseMatrix<VT>, VTSel> {
    static void apply(
            DenseMatrix<VT> *& res,
            const DenseMatrix<VT> * arg, const DenseMatrix<VT> * ins,
            VTSel rowLowerInclRaw, VTSel rowUpperExclRaw,
            DCTX(ctx)
    ) {
        const size_t numRowsArg = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();
        const size_t numRowsIns = ins->getNumRows();
        const size_t numColsIns = ins->getNumCols();

        // VTSel enables better validation
        const size_t rowLowerIncl = static_cast<const size_t>(rowLowerInclRaw);
        const size_t rowUpperExcl = static_cast<const size_t>(rowUpperExclRaw);
        
        validateInsertRowArgs(rowLowerIncl, rowLowerInclRaw, rowUpperExcl, rowUpperExclRaw,
                    numRowsArg, numColsArg, numRowsIns, numColsIns);

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