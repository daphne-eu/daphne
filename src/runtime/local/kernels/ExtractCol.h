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

#include <sstream>
#include <stdexcept>

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
// Boundary validation
// ****************************************************************************

// index boundaries are verified later for performance
#define VALIDATE_ARGS(numColsSel) \
    if(numColsSel != 1) { \
        std::ostringstream errMsg; \
        errMsg << "invalid argument passed to ExtractCol: column selection must be given as column matrix but has '" \
            << numColsSel << "' columns instead of one"; \
        throw std::runtime_error(errMsg.str()); \
    }

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix (positions)
// ----------------------------------------------------------------------------

template<typename VTArg, typename VTSel>
struct ExtractCol<DenseMatrix<VTArg>, DenseMatrix<VTArg>, DenseMatrix<VTSel>> {
    static void apply(DenseMatrix<VTArg> *& res, const DenseMatrix<VTArg> * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) {
        VALIDATE_ARGS(sel->getNumCols());

        // left as VTSel to enable more boundary validation, converted to size_t later
        const VTSel * VTcolIdxs = sel->getValues();
        const size_t numColsRes = sel->getNumRows();
        const size_t numRows = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTArg>>(numRows, numColsRes, false);

        const VTArg * valuesArg = arg->getValues();
        VTArg * valuesRes = res->getValues();
        
        const size_t rowSkipArg = arg->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();
        
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numColsRes; c++) {
                const VTSel VTcolIdx = VTcolIdxs[c];
                const size_t colIdx = static_cast<const size_t>(VTcolIdx);
                if (VTcolIdx < 0 || numColsArg <= colIdx) {
                    std::ostringstream errMsg;
                    errMsg << "invalid argument '" << VTcolIdx << "' passed to ExtractCol: out of bounds "
                        "for dense matrix with column boundaries '[0, " << numColsArg << ")'";
                    throw std::out_of_range(errMsg.str());
                }

                valuesRes[c] = valuesArg[colIdx];
            }
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
        std::string delimiter = ".";
        const std::string colName = std::string(sel);
        const std::string frameName = colName.substr(0, colName.find(delimiter));
        const std::string colLabel = colName.substr(colName.find(delimiter) + delimiter.length(), colName.length());
        if (colLabel.compare("*") ==0) {
            const std::string * labels = arg->getLabels();
            const size_t numLabels = arg->getNumCols();
            std::vector<size_t> extractLabelIdxs;
            for (size_t i = 0; i < numLabels; i++) {
                std::string labelFrameName = labels[i].substr(0, labels[i].find(delimiter));
                if (labelFrameName.compare(frameName) == 0) {
                    extractLabelIdxs.push_back(arg->getColumnIdx(labels[i]));
                }
            }
            res = DataObjectFactory::create<Frame>(arg, 0, arg->getNumRows(), extractLabelIdxs.size(), extractLabelIdxs.data());
        } else {
            size_t colIdx = arg->getColumnIdx(sel);
            res = DataObjectFactory::create<Frame>(arg, 0, arg->getNumRows(), 1, &colIdx);
        }
        
    }
};

template< typename VTSel >
struct ExtractCol<Frame, Frame, DenseMatrix<VTSel>> {
    static void apply(Frame *& res, const Frame * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) {
        VALIDATE_ARGS(sel->getNumCols());

        // left as VTSel to enable more boundary validation, converted to size_t later
        const VTSel * VTvaluesSel = sel->getValues();
        const size_t* colIdxs = reinterpret_cast<const size_t*>(VTvaluesSel);
        const size_t numColsRes = sel->getNumRows();
        const size_t numRowsRes = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();
        for (size_t c = 0; c < numColsRes; c++) {
            const VTSel VTcolIdx = VTvaluesSel[c];
            if (VTcolIdx < 0 || numColsArg <= colIdxs[c]) {
                std::ostringstream errMsg;
                errMsg << "invalid argument '" << VTcolIdx << "' passed to ExtractCol: ouf of bounds "
                    "for frame with column boundaries '[0, " << numColsArg << ")'";
                throw std::out_of_range(errMsg.str());
            }
        }
        res = DataObjectFactory::create<Frame>(arg, 0, numRowsRes, numColsRes, colIdxs);
    }
};

#undef VALIDATE_ARGS