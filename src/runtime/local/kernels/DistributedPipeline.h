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
//#include <runtime/local/datastructures/DataObjectFactory.h>
//#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/distributed/coordinator/kernels/DistributedWrapper.h>
//#include <ir/daphneir/Daphne.h>


#include <iostream>

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

//template<class DTRes, class DTArg>
//struct DistributedPipeline {
//    static void apply(
//            DTRes *& res,
//            const char * irCode,
//            const DTArg ** inputs, size_t numInputs,
//            size_t * outRows, size_t numOutRows,
//            size_t *outCols, size_t numOutCols,
//            DCTX(ctx)
//    );
//};

// ****************************************************************************
// Convenience function
// ****************************************************************************

// One output.
template<class DTRes>
void distributedPipeline(
        DTRes ** outputs, size_t numOutputs,
        const Structure ** inputs, size_t numInputs,
        int64_t * outRows, int64_t * outCols,
        int64_t * splits, int64_t * combines,
        const char * irCode,
        DCTX(ctx)
) {

    auto wrapper = std::make_unique<DistributedWrapper<DTRes>>(ctx);
    // TODO *** -> **
    DTRes ***res = new DTRes **[numOutputs];
    for (size_t i = 0; i < numOutputs; i++)
        res[i] = outputs+i;
    wrapper->execute(irCode, res, inputs, numInputs, numOutputs, outRows, outCols,
            reinterpret_cast<VectorSplit *>(splits), reinterpret_cast<VectorCombine *>(combines));
}
