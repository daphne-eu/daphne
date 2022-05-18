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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>

#include <stdexcept>

#include <cassert>
#include <cstddef>
#include <cstdint>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg, class DTSel>
struct ExtractCol {
    static void apply(DTRes *& res, const DTArg * arg, const DTSel * sel, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

// TODO Actually, the positions should be given as size_t to stay consistent
// with the rest of the code and DaphneIR (even though int64_t also makes
// sense), but currently, it would be too hard to get a matrix of size_t via
// DaphneDSL, since we do not have value type casts yet.
template<class DTRes, class DTArg, class DTSel>
void extractCol(DTRes *& res, const DTArg * arg, const DTSel * sel, DCTX(ctx)) {
    ExtractCol<DTRes, DTArg, DTSel>::apply(res, arg, sel, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix (positions)
// ----------------------------------------------------------------------------

template<typename VT>
struct ExtractCol<DenseMatrix<VT>, DenseMatrix<VT>, DenseMatrix<int64_t>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, const DenseMatrix<int64_t> * sel, DCTX(ctx)) {
        assert((sel->getNumCols() == 1) && "parameter colIdxs must be a column matrix");

        const size_t numColsRes = sel->getNumRows();
        const auto* colIdxs = reinterpret_cast<const size_t *>(sel->getValues());
        for(size_t i = 0; i < numColsRes; i++) {
            assert((colIdxs[i] < arg->getNumCols()) && "column index out of bounds");
        }
        
        const size_t numRows = arg->getNumRows();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numColsRes, false);

        const VT * valuesArg = arg->getValues();
        VT * valuesRes = res->getValues();
        
        const size_t rowSkipArg = arg->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();
        
        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numColsRes; c++)
                valuesRes[c] = valuesArg[colIdxs[c]];
            valuesArg += rowSkipArg;
            valuesRes += rowSkipRes;
        }
    }
};

// ----------------------------------------------------------------------------
// Frame <- Frame, String (column label)
// ----------------------------------------------------------------------------

template<>
struct ExtractCol<Frame, Frame, char> {
    static void apply(Frame *& res, const Frame * arg, const char * sel, DCTX(ctx)) {
        size_t colIdx = arg->getColumnIdx(sel);
        res = DataObjectFactory::create<Frame>(arg, 0, arg->getNumRows(), 1, &colIdx);
    }
};

template< typename VTSel >
struct ExtractCol<Frame, Frame, DenseMatrix<VTSel>> {
    static void apply(Frame *& res, const Frame * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) {
        if(sel->getNumCols() != 1)
            throw std::runtime_error("parameter colIdxs must be a column matrix");

        const size_t numColsRes = sel->getNumRows();
        const auto* colIdxs = reinterpret_cast<const size_t *>(sel->getValues());
        for(size_t i = 0; i < numColsRes; i++) {
            if(colIdxs[i] >= arg->getNumCols())
                throw std::runtime_error("column index out of bounds");
        }
        const size_t numRows = arg->getNumRows();

        res = DataObjectFactory::create<Frame>(arg, 0, numRows, numColsRes, colIdxs);
    }
};