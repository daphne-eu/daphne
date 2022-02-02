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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_ROWBIND_H
#define SRC_RUNTIME_LOCAL_KERNELS_ROWBIND_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>

#include <cassert>
#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTUp, class DTLow>
struct RowBind {
    static void apply(DTRes *& res, const DTUp * ups, const DTLow * lows, const ValueTypeCode * vtc, const string * lbl, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

// template<class DTRes, class DTUp, class DTLow>
// void rowBind(DTRes *& res, const DTUp * ups, const DTLow * lows, DCTX(ctx)) {
//     RowBind<DTRes, DTUp, DTLow>::apply(res, ups, lows, ctx);
// }

template<class DTRes, class DTUp, class DTLow>
void rowBind(DTRes *& res, const DTUp * ups, const DTLow * lows, const ValueTypeCode * vtc, const string * lbl, DCTX(ctx)) {
    RowBind<DTRes, DTUp, DTLow>::apply(res, ups, lows, vtc, lbl, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct RowBind<DenseMatrix<VT>, DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * ups, const DenseMatrix<VT> * lows, const ValueTypeCode * schema, const string * labels, DCTX(ctx)) {
        const size_t numCols = ups->getNumCols();
        assert((numCols == lows->getNumCols()) && "ups and lows must have the same number of columns");
        
        const size_t numRowsUps = ups->getNumRows();
        const size_t numRowsLows = lows->getNumRows();
        const size_t numColsUps = ups->getNumCols();
        const size_t numColsLows = lows->getNumCols();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRowsUps + numRowsLows, numCols, false);
        
        
        const VT * valuesUps = ups->getValues();
        const VT * valuesLows = lows->getValues();
        VT * valuesRes = res->getValues();
        
        memcpy(valuesRes             , valuesUps, numColsUps * numRowsUps * sizeof(VT));
        memcpy(valuesRes + numRowsUps * numColsUps, valuesLows, numColsLows * numRowsLows * sizeof(VT));
        
    }
};

// ----------------------------------------------------------------------------
// Frame <- Frame, Frame, schema, labels
// ----------------------------------------------------------------------------

template<>
struct RowBind<Frame, Frame, Frame> {
    static void apply(Frame *& res, const Frame * ups, const Frame * lows, const ValueTypeCode * schema, const string * labels, const  DCTX(ctx)) {
        res = DataObjectFactory::create<Frame>(ups->getNumRows()+lows->getNumRows(), ups->getNumCols(), schema, labels, 0);
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_ROWBIND_H