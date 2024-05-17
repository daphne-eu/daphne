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
#include <runtime/local/datastructures/Matrix.h>

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
        const VTSel rowLowerIncl, const VTSel rowUpperExcl,
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
        const VTSel rowLowerIncl, const VTSel rowUpperExcl,
        DCTX(ctx)
) {
    InsertRow<DTArg, DTIns, VTSel>::apply(res, arg, ins, rowLowerIncl, rowUpperExcl, ctx);
}

// ****************************************************************************
// Boundary validation
// ****************************************************************************

template<typename VTSel>
void validateArgsInsertRow(size_t rowLowerIncl_Size, VTSel rowLowerIncl, size_t rowUpperExcl_Size, VTSel rowUpperExcl,
                    size_t numRowsArg, size_t numColsArg, size_t numRowsIns, size_t numColsIns) {
    
    if (rowUpperExcl_Size < rowLowerIncl_Size || numRowsArg < rowUpperExcl_Size
        || (rowLowerIncl_Size == numRowsArg && rowLowerIncl_Size != 0)) {
        std::ostringstream errMsg;
        errMsg << "invalid arguments '" << rowLowerIncl << ", " << rowUpperExcl
                << "' passed to InsertRow: it must hold 0 <= rowLowerIncl <= rowUpperExcl <= #rows "
                << "and rowLowerIncl < #rows (unless both are zero) where #rows of arg is '" << numRowsArg << "'";
        throw std::out_of_range(errMsg.str());
    }
    
    if(numRowsIns != rowUpperExcl_Size - rowLowerIncl_Size){
        std::ostringstream errMsg;
        errMsg << "invalid arguments '" << rowLowerIncl << ", " << rowUpperExcl
                << "' passed to InsertRow: the number of addressed rows in arg '" << rowUpperExcl_Size - rowLowerIncl_Size
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
            VTSel rowLowerIncl, VTSel rowUpperExcl,
            DCTX(ctx)
    ) {
        const size_t numRowsArg = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();
        const size_t numRowsIns = ins->getNumRows();
        const size_t numColsIns = ins->getNumCols();

        const size_t rowLowerIncl_Size = static_cast<const size_t>(rowLowerIncl);
        const size_t rowUpperExcl_Size = static_cast<const size_t>(rowUpperExcl);
        
        validateArgsInsertRow(rowLowerIncl_Size, rowLowerIncl, rowUpperExcl_Size, rowUpperExcl,
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
        for(size_t r = 0; r < rowLowerIncl_Size; r++) {
            memcpy(valuesRes, valuesArg, numColsArg * sizeof(VT));
            valuesRes += rowSkipRes;
            valuesArg += rowSkipArg;
        }
        for(size_t r = rowLowerIncl_Size; r < rowUpperExcl_Size; r++) {
            memcpy(valuesRes, valuesIns, numColsArg * sizeof(VT));
            valuesRes += rowSkipRes;
            valuesIns += rowSkipIns;
        }
        valuesArg += rowSkipArg * numRowsIns; // skip rows in arg
        for(size_t r = rowUpperExcl_Size; r < numRowsArg; r++) {
            memcpy(valuesRes, valuesArg, numColsArg * sizeof(VT));
            valuesRes += rowSkipRes;
            valuesArg += rowSkipArg;
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template<typename VT, typename VTSel>
struct InsertRow<Matrix<VT>, Matrix<VT>, VTSel> {
    static void apply(
            Matrix<VT> *& res,
            const Matrix<VT> * arg, const Matrix<VT> * ins,
            VTSel rowLowerIncl, VTSel rowUpperExcl,
            DCTX(ctx)
    ) {
        const size_t numRowsArg = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();

        const size_t rowLowerIncl_Size = static_cast<const size_t>(rowLowerIncl);
        const size_t rowUpperExcl_Size = static_cast<const size_t>(rowUpperExcl);

        validateArgsInsertRow(rowLowerIncl_Size, rowLowerIncl, rowUpperExcl_Size, rowUpperExcl,
                    numRowsArg, numColsArg, ins->getNumRows(), ins->getNumCols());

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRowsArg, numColsArg, false);

        // fill values above insertion, then between and lastly below
        res->prepareAppend();
        for (size_t r = 0; r < rowLowerIncl_Size; ++r)
            for (size_t c = 0; c < numColsArg; ++c)
                res->append(r, c, arg->get(r, c));

        for (size_t r = rowLowerIncl_Size; r < rowUpperExcl_Size; ++r)
            for (size_t c = 0; c < numColsArg; ++c)
                res->append(r, c, ins->get(r - rowLowerIncl_Size, c));
                
        for (size_t r = rowUpperExcl_Size; r < numRowsArg; ++r)
            for (size_t c = 0; c < numColsArg; ++c)
                res->append(r, c, arg->get(r, c));
        res->finishAppend();
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_INSERTROW_H
