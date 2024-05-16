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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_FILTERROW_H
#define SRC_RUNTIME_LOCAL_KERNELS_FILTERROW_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <stdexcept>

#include <cstddef>
#include <cstdint>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg, typename VTSel>
struct FilterRow {
    static void apply(DTRes *& res, const DTArg * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg, typename VTSel>
void filterRow(DTRes *& res, const DTArg * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) {
    FilterRow<DTRes, DTArg, VTSel>::apply(res, arg, sel, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT, typename VTSel>
struct FilterRow<DenseMatrix<VT>, DenseMatrix<VT>, VTSel> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) {
        const size_t numRowsArg = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(sel->getNumRows() != numRowsArg)
            throw std::runtime_error("sel must have exactly one entry (row) for each row in arg");
        if(sel->getNumCols() != 1)
            throw std::runtime_error("sel must be a single-column matrix");

        size_t numRowsRes = 0;
        for(size_t r = 0; r < numRowsArg; r++)
            numRowsRes += sel->get(r, 0);

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRowsRes, numCols, false);

        const VT * valuesArg = arg->getValues();
        VT * valuesRes = res->getValues();
        const size_t rowSkipArg = arg->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();
        for(size_t r = 0; r < numRowsArg; r++) {
            if(sel->get(r, 0)) {
                memcpy(valuesRes, valuesArg, numCols * sizeof(VT));
                valuesRes += rowSkipRes;
            }
            valuesArg += rowSkipArg;
        }
    }
};

// ----------------------------------------------------------------------------
// Frame <- Frame
// ----------------------------------------------------------------------------

// 0 (row-wise) or 1 (column-wise)
#define FILTERROW_FRAME_MODE 0

template<typename VTSel>
struct FilterRow<Frame, Frame, VTSel> {
    static void apply(Frame *& res, const Frame * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        const ValueTypeCode * schema = arg->getSchema();
        
        if(sel->getNumRows() != numRows)
            throw std::runtime_error("sel must have exactly one entry (row) for each row in arg");
        if(sel->getNumCols() != 1)
            throw std::runtime_error("sel must be a single-column matrix");
        
#if FILTERROW_FRAME_MODE == 0
        // Add some padding due to stores in units of 8 bytes (see below). This
        // formula is a little pessimistic, though.
        const size_t numRowsAlloc = numRows + sizeof(uint64_t) / sizeof(uint8_t) - 1;
#elif FILTERROW_FRAME_MODE == 1
        const size_t numRowsAlloc = numRows;
#endif
        if(res == nullptr)
            res = DataObjectFactory::create<Frame>(
                    numRowsAlloc, numCols, schema, arg->getLabels(), false
            );
        
        const VTSel * valuesSel = sel->getValues();
        
#if FILTERROW_FRAME_MODE == 0
        // Some information on each column.
        size_t * const elementSizes = new size_t[numCols];
        const uint8_t ** argCols = new const uint8_t *[numCols];
        uint8_t ** resCols = new uint8_t *[numCols];
        // Initialize information on each column.
        for(size_t c = 0; c < numCols; c++) {
            elementSizes[c] = ValueTypeUtils::sizeOf(schema[c]);
            argCols[c] = reinterpret_cast<const uint8_t *>(arg->getColumnRaw(c));
            resCols[c] = reinterpret_cast<uint8_t *>(res->getColumnRaw(c));
        }
        // Actual filtering.
        for(size_t r = 0; r < numRows; r++) {
            if(valuesSel[r]) {
                for(size_t c = 0; c < numCols; c++) {
                    // We always copy in units of 8 bytes (uint64_t). If the
                    // actual element size is lower, the superfluous bytes will
                    // be overwritten by the next match. With this approach, we
                    // do not need to call memcpy for each element, nor
                    // interpret the types for a L/S of fitting size.
                    *reinterpret_cast<uint64_t *>(resCols[c]) = 
                            *reinterpret_cast<const uint64_t *>(argCols[c]);
                    resCols[c] += elementSizes[c];
                }
            }
            for(size_t c = 0; c < numCols; c++)
                argCols[c] += elementSizes[c];
        }
        auto resColsInit0 = reinterpret_cast<uint8_t *>(res->getColumnRaw(0));
        const size_t numRowsRes = (resCols[0] - resColsInit0) / elementSizes[0];
        res->shrinkNumRows(numRowsRes);
        // Free information on each column.
        delete[] elementSizes;
        delete[] argCols;
        delete[] resCols;
#elif FILTERROW_FRAME_MODE == 1
        // TODO Implement a columnar approach.
#endif
    }
};

#undef FILTERROW_FRAME_MODE

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template<typename VT, typename VTSel>
struct FilterRow<Matrix<VT>, Matrix<VT>, VTSel> {
    static void apply(Matrix<VT> *& res, const Matrix<VT> * arg, const Matrix<VTSel> * sel, DCTX(ctx)) {
        const size_t numRowsArg = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (sel->getNumRows() != numRowsArg)
            throw std::runtime_error("sel must have exactly one entry (row) for each row in arg");
        if (sel->getNumCols() != 1)
            throw std::runtime_error("sel must be a single-column matrix");

        size_t numRowsRes = 0;
        for (size_t r = 0; r < numRowsArg; ++r)
            numRowsRes += sel->get(r, 0);

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRowsRes, numCols, false);

        size_t resRow = 0;
        res->prepareAppend();
        for (size_t r = 0; r < numRowsArg; ++r) {
            if (sel->get(r, 0)) {
                for (size_t c = 0; c < numCols; ++c)
                    res->append(resRow, c, arg->get(r, c));
                ++resRow;
            }
        }
        res->finishAppend();
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_FILTERROW_H