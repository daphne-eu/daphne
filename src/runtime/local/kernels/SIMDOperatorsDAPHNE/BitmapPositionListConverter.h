/*
 * Copyright 2023 The DAPHNE Consortium
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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORS_BITMAPPOSITIONLISTCONVERTER_H
#define SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORS_BITMAPPOSITIONLISTCONVERTER_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/AggAll.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <cassert>
#include <cstddef>
#include <unordered_set>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct BitmapPositionListConverter {
    static void apply(DTRes *& res, const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void bitmapPositionListConverter(DTRes *& res, const DTArg * arg, DCTX(ctx)) {
    BitmapPositionListConverter<DTRes, DTArg>::apply(res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
struct BitmapPositionListConverter<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = 1;
        const VTArg * valuesArg = arg->getValues();
        size_t length = 0;
        std::vector<size_t> positions;
        for(size_t r = 0; r < numRows; r++) {
            if(valuesArg[0] == 1) {
                length++;
                positions.push_back(r);
            }
            valuesArg += arg->getRowSkip();
        }
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(length, numCols, true);
        
        VTRes * valuesRes = res->getValues();

        for(size_t r = 0; r < positions.size(); r++) {
            *valuesRes = static_cast<VTRes>(positions[r]);
            valuesRes += res->getRowSkip();
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORS_BITMAPPOSITIONLISTCONVERTER_H