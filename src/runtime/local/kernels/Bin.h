/*
 * Copyright 2024 The DAPHNE Consortium
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

#include <type_traits>

#include <cmath>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct Bin {
    static void apply(DTRes *& res, const DTArg * arg, int64_t numBins, typename DTArg::VT min, typename DTArg::VT max, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void bin(DTRes *& res, const DTArg * arg, int64_t numBins, typename DTArg::VT min, typename DTArg::VT max, DCTX(ctx)) {
    Bin<DTRes, DTArg>::apply(res, arg, numBins, min, max, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
struct Bin<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, int64_t numBins, VTArg min, VTArg max, DCTX(ctx)) {
        if(numBins <= 0)
            throw std::runtime_error("bin-kernel: numBins must be greater than zero");
        if(min > max)
            throw std::runtime_error("bin-kernel: min must not be greater than max");
        if(min == max && numBins > 1)
            throw std::runtime_error("bin-kernel: min equals max, so numBins must not be greater than 1");
        if(std::is_floating_point<VTArg>::value) {
            const VTArg inf = std::numeric_limits<VTArg>::infinity();
            if(std::isnan(min))
                throw std::runtime_error("bin-kernel: min must not be NaN");
            if(std::isnan(max))
                throw std::runtime_error("bin-kernel: max must not be NaN");
            if(min == inf)
                throw std::runtime_error("bin-kernel: min must not be infinity");
            if(max == inf)
                throw std::runtime_error("bin-kernel: max must not be infinity");
            if(min == -inf)
                throw std::runtime_error("bin-kernel: min must not be -infinity");
            if(max == -inf)
                throw std::runtime_error("bin-kernel: max must not be -infinity");
        }
        
        double binSize = static_cast<double>(max - min) / numBins;

        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);
        
        const VTArg * valuesArg = arg->getValues();
        VTRes * valuesRes = res->getValues();
        const size_t rowSkipArg = arg->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();

        if(min == max && numBins == 1)
            for(size_t r = 0; r < numRows; r++) {
                for(size_t c = 0; c < numCols; c++)
                    valuesRes[c] = 0;
                valuesRes += rowSkipRes;
            }
        else
            for(size_t r = 0; r < numRows; r++) {
                for(size_t c = 0; c < numCols; c++) {
                    VTArg v = valuesArg[c];
                    VTRes b;
                    if(v <= min) // important if VTArg is an unsigned integer type
                        b = 0;
                    else {
                        b = std::ceil(static_cast<double>(v - min) / binSize) - 1;
                        if(b < 0)
                            b = 0;
                        else if(b >= numBins)
                            b = numBins - 1;
                    }
                    valuesRes[c] = b;
                }
                valuesArg += rowSkipArg;
                valuesRes += rowSkipRes;
            }
    }
};