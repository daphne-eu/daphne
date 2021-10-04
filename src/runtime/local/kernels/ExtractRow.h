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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_EXTRACTROW_H
#define SRC_RUNTIME_LOCAL_KERNELS_EXTRACTROW_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <stdexcept>

#include <cstddef>
#include <cstdint>
#include <cmath>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg, typename VTSel>
struct ExtractRow {
    static void apply(DTRes *& res, const DTArg * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg, typename VTSel>
void extractRow(DTRes *& res, const DTArg * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) {
    ExtractRow<DTRes, DTArg, VTSel>::apply(res, arg, sel, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Frame <- Frame
// ----------------------------------------------------------------------------

// 0 (row-wise) or 1 (column-wise)
#define EXTRACTROW_FRAME_MODE 0

template<typename VTSel>
struct ExtractRow<Frame, Frame, VTSel> {
    static void apply(Frame *& res, const Frame * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) {
        if(sel->getNumCols() != 1)
            throw std::runtime_error("sel must be a single-column matrix");
        
        const size_t numRowsSel = sel->getNumRows();
        const size_t numCols = arg->getNumCols();
        const ValueTypeCode * schema = arg->getSchema();
        
#if EXTRACTROW_FRAME_MODE == 0
        // Add some padding due to stores in units of 8 bytes (see below). This
        // formula is a little pessimistic, though.
        const size_t numRowsResAlloc = numRowsSel + sizeof(uint64_t) / sizeof(uint8_t) - 1;
#elif EXTRACTROW_FRAME_MODE == 1
        const size_t numRowsResAlloc = numRowsSel;
#endif
        if(res == nullptr)
            res = DataObjectFactory::create<Frame>(
                    numRowsResAlloc, numCols, schema, arg->getLabels(), false
            );
        
        const VTSel * valuesSel = sel->getValues();
        
#if EXTRACTROW_FRAME_MODE == 0
        // Some information on each column.
        const auto elementSizes = std::make_unique<size_t[]>(numCols);
        const auto argCols = std::make_unique<const uint8_t*[]>(numCols);
        auto resCols = std::make_unique<uint8_t*[]>(numCols);
        // Initialize information on each column.
        for(size_t c = 0; c < numCols; c++) {
            elementSizes[c] = ValueTypeUtils::sizeOf(schema[c]);
            argCols[c] = reinterpret_cast<const uint8_t *>(arg->getColumnRaw(c));
            resCols[c] = reinterpret_cast<uint8_t *>(res->getColumnRaw(c));
        }
        // Actual filtering.
        for(size_t r = 0; r < numRowsSel; r++) {
            const size_t pos = valuesSel[r];
            for(size_t c = 0; c < numCols; c++) {
                // We always copy in units of 8 bytes (uint64_t). If the
                // actual element size is lower, the superfluous bytes will
                // be overwritten by the next match. With this approach, we
                // do not need to call memcpy for each element, nor
                // interpret the types for a L/S of fitting size.
                // TODO Don't multiply by elementSize, but left-shift by
                // ld(elementSize).
                *reinterpret_cast<uint64_t *>(resCols[c]) = 
                        *reinterpret_cast<const uint64_t *>(
                                argCols[c] + pos * elementSizes[c]
                        );
                resCols[c] += elementSizes[c];
            }
        }
        res->shrinkNumRows(numRowsSel);

#elif EXTRACTROW_FRAME_MODE == 1
        // TODO Implement a columnar approach.
#endif
    }
};

#undef EXTRACTROW_FRAME_MODE

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT, typename VTSel>
struct ExtractRow<DenseMatrix<VT>, DenseMatrix<VT>, VTSel> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) {
        // input validation
        if(arg==nullptr){
            throw std::runtime_error("arg cannot be null");
        }
        if(sel== nullptr){
            throw std::runtime_error("sel cannot be null");
        }
        if(sel->getNumCols() != 1){
            throw std::runtime_error("sel must be a single-column matrix");
        }
        const size_t numRowsInSel = sel->getNumRows();
        const size_t numInputRows = arg->getNumRows();
        const size_t numInputCols = arg->getNumCols();   
        if(res ==nullptr){
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRowsInSel, numInputCols, false);
        }
        else if(res->getNumRows() != numRowsInSel || res->getNumCols() != numInputCols){
            // TODO what is the best strategy: throw a warning or just re-allocate?
            throw std::runtime_error("res is not null, but it has wrong numCols and numRows");
        }
        
        //Main Logic
        VT * allUpdatedValues = res->getValues();
        const VTSel * rowsInSel = sel->getValues();
        for(size_t r = 0; r < numRowsInSel; r++){
            const VTSel valSelectedRow = rowsInSel[r];  // only one column
            // TODO For performance reasons, we might skip such checks or make
            // them optional somehow, but it is okay for now.
            if(std::isnan(valSelectedRow) || valSelectedRow < 0 || static_cast<size_t>(valSelectedRow) > numInputRows-1){
                throw std::runtime_error("sel cannot have NaN nor negative nor value that is greater than numRows in arg");
            }  
            else
            {
                const VT * allValues = arg->getValues()+valSelectedRow*arg->getRowSkip();
                for(size_t c = 0; c < numInputCols; c++){
                    allUpdatedValues[c]=allValues[c];   
                }   
                allUpdatedValues += res->getRowSkip();                     
            }
        }
    }        
};
#endif //SRC_RUNTIME_LOCAL_KERNELS_EXTRACTROW_H
