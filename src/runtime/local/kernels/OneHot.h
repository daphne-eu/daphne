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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_ONEHOT_H
#define SRC_RUNTIME_LOCAL_KERNELS_ONEHOT_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <stdexcept>

#include <cassert>
#include <cstddef>
#include <cstdint>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct OneHot {
    static void apply(DTRes *& res, const DTArg * arg, const DenseMatrix<int64_t> * info, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void oneHot(DTRes *& res, const DTArg * arg, const DenseMatrix<int64_t> * info, DCTX(ctx)) {
    OneHot<DTRes, DTArg>::apply(res, arg, info, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct OneHot<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, const DenseMatrix<int64_t> * info, DCTX(ctx)) {
        assert((info->getNumRows() == 1) && "parameter info must be a row matrix");
        
        const size_t numColsArg = arg->getNumCols();
        assert((numColsArg == info->getNumCols()) && "parameter info must provide information for each column of parameter arg");
        
        size_t numColsRes = 0;
        const int64_t * valuesInfo = info->getValues();
        for(size_t c = 0; c < numColsArg; c++) {
            const int64_t numDistinct = valuesInfo[c];
            if(numDistinct == -1)
                numColsRes++;
            else if(numDistinct > 0)
                numColsRes += numDistinct;
            else
                assert(false && "invalid info");
        }
        
        const size_t numRows = arg->getNumRows();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numColsRes, false);

        const VT * valuesArg = arg->getValues();
        VT * valuesRes = res->getValues();
        
        const size_t rowSkipArg = arg->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();
        
        for(size_t r = 0; r < numRows; r++) {
            size_t cRes = 0;
            for(size_t cArg = 0; cArg < numColsArg; cArg++) {
                const int64_t numDistinct = valuesInfo[cArg];
                if(numDistinct == -1)
                    // retain value from argument matrix
                    valuesRes[cRes++] = valuesArg[cArg];
                else {
                    // one-hot encode value from argument matrix
                    for(int64_t d = 0; d < numDistinct; d++)
                        valuesRes[cRes + d] = 0;
                    // throw error or log issue?
                    // if (static_cast<size_t>(valuesArg[cArg]) < numDistinct)
                    valuesRes[cRes + static_cast<size_t>(valuesArg[cArg])] = 1;
                    cRes += numDistinct;
                }
            }
            valuesArg += rowSkipArg;
            valuesRes += rowSkipRes;
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template<typename VT>
struct OneHot<Matrix<VT>, Matrix<VT>> {
    static void apply(Matrix<VT> *& res, const Matrix<VT> * arg, const Matrix<int64_t> * info, DCTX(ctx)) {
        const size_t numColsArg = arg->getNumCols();
        const size_t numRows = arg->getNumRows();
        
        if (info->getNumRows() != 1)
            throw std::runtime_error("OneHot: parameter info must be a row matrix");
        if (numColsArg != info->getNumCols())
            throw std::runtime_error("OneHot: parameter info must provide information for each column of parameter arg");
        
        size_t numColsRes = 0;
        for (size_t c=0; c < numColsArg; c++) {
            const int64_t numDistinct = info->get(0, c);
            if (numDistinct == -1)
                numColsRes++;
            else if (numDistinct > 0)
                numColsRes += numDistinct;
            else
                throw std::runtime_error("OneHot: parameter info values must be -1, 0, or positive signed integer");
        }
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numColsRes, false);

        res->prepareAppend();
        for(size_t r=0; r < numRows; ++r) {
            size_t cRes = 0;
            for(size_t cArg=0; cArg < numColsArg; ++cArg) {
                const int64_t numDistinct = info->get(0, cArg);
                if(numDistinct == -1)
                    // retain value from argument matrix
                    res->append(r, cRes++, arg->get(r, cArg));
                else {
                    // one-hot encode value from argument matrix
                    // skipped values are assumed 0, perform bounds check
                    size_t argVal = static_cast<size_t>(arg->get(r, cArg));
                    if (argVal < numDistinct)
                        res->append(r, cRes + argVal, 1);
                    else
                        throw std::out_of_range("OneHot: encoded values in arg must be smaller than respective vector length");
                    cRes += numDistinct;
                }
            }
        }
        res->finishAppend();
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_ONEHOT_H