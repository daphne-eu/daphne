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

#include <sstream>
#include <stdexcept>

#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTArg, class DTIns, typename VTSel>
struct InsertCol {
    static void apply(
            DTArg *& res,
            const DTArg * arg, const DTIns * ins,
            VTSel colLowerIncl, VTSel colUpperExcl,
            DCTX(ctx)
    ) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTArg, class DTIns, typename VTSel>
void insertCol(
        DTArg *& res,
        const DTArg * arg, const DTIns * ins,
        VTSel colLowerIncl, VTSel colUpperExcl,
        DCTX(ctx)
) {
    InsertCol<DTArg, DTIns, VTSel>::apply(res, arg, ins, colLowerIncl, colUpperExcl, ctx);
}

// ****************************************************************************
// Boundary validation function
// ****************************************************************************

template<typename VTSel>
void validateInsertColArgs(const size_t colLowerIncl, VTSel colLowerInclRaw, const size_t colUpperExcl, VTSel colUpperExclRaw,
                    const size_t numRowsArg, const size_t numColsArg, const size_t numRowsIns, const size_t numColsIns) {

    if (colLowerInclRaw < 0 || colUpperExclRaw < colLowerInclRaw || numColsArg < colUpperExcl) {
        std::ostringstream errMsg;
        errMsg << "invalid arguments '" << colLowerInclRaw << ", " << colUpperExclRaw
                << "' passed to InsertCol: must be positive, colLowerIncl must be smaller than colUpperExcl "
                << "and both within columns of arg '" << numColsArg << "'";
        throw std::out_of_range(errMsg.str());
    }

    if(numColsIns != colUpperExcl - colLowerIncl){
        std::ostringstream errMsg;
        errMsg << "InsertCol: the number of addressed rows in arg '" << colUpperExcl - colLowerIncl
                << "' and the number of rows in ins '" << numColsIns << "' must match";
        throw std::runtime_error(errMsg.str());
    }

    if(numRowsIns != numRowsArg) {
        std::ostringstream errMsg;
        errMsg << "insertRow: the number of columns in arg '" << numRowsArg << "' and ins '" << numRowsIns << "' must match";
        throw std::runtime_error(errMsg.str());
    }
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTArg, typename VTSel>
struct InsertCol<DenseMatrix<VTArg>, DenseMatrix<VTArg>, VTSel> {
    static void apply(
            DenseMatrix<VTArg> *& res,
            const DenseMatrix<VTArg> * arg, const DenseMatrix<VTArg> * ins,
            VTSel colLowerInclRaw, VTSel colUpperExclRaw,
            DCTX(ctx)
    ) {
        const size_t numRowsArg = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();
        const size_t numRowsIns = ins->getNumRows();
        const size_t numColsIns = ins->getNumCols();

        // VTSel enables better validation
        const size_t colLowerIncl = static_cast<const size_t>(colLowerInclRaw);
        const size_t colUpperExcl = static_cast<const size_t>(colUpperExclRaw);
        
        validateInsertColArgs(colLowerIncl, colLowerInclRaw, colUpperExcl, colUpperExclRaw,
                    numRowsArg, numColsArg, numRowsIns, numColsIns);

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTArg>>(numRowsArg, numColsArg, false);
        
        VTArg * valuesRes = res->getValues();
        const VTArg * valuesArg = arg->getValues();
        const VTArg * valuesIns = ins->getValues();
        const size_t rowSkipRes = res->getRowSkip();
        const size_t rowSkipArg = arg->getRowSkip();
        const size_t rowSkipIns = ins->getRowSkip();
        
        // TODO Can be simplified/more efficient in certain cases.
        for(size_t r = 0; r < numRowsArg; r++) {
            memcpy(valuesRes, valuesArg, colLowerIncl * sizeof(VTArg));
            memcpy(valuesRes + colLowerIncl, valuesIns, numColsIns * sizeof(VTArg));
            memcpy(valuesRes + colUpperExcl, valuesArg + colUpperExcl, (numColsArg - colUpperExcl) * sizeof(VTArg));
            valuesRes += rowSkipRes;
            valuesArg += rowSkipArg;
            valuesIns += rowSkipIns;
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_INSERTROW_H