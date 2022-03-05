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

//#include <cassert>
//#include <cstddef>

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
        DTRes *& output0,
        const Structure ** inputs,
        size_t numInputs, size_t numOutputs,
        size_t * outRows, size_t * outCols,
        int64_t * splits, int64_t * combines,
        const char * irCode,
        DCTX(ctx)
) {
    assert(numOutputs == 1 && "FIXME: lowered to wrong kernel");

    auto wrapper = std::make_unique<DistributedWrapper<DTRes>>(ctx);

    DTRes **res[] = {&output0};
    wrapper->execute(irCode, res, inputs, numInputs, numOutputs, outRows, outCols,
            reinterpret_cast<VectorSplit *>(splits), reinterpret_cast<VectorCombine *>(combines));
}

// Two outputs.
template<class DTRes>
void distributedPipeline(
        DTRes *& output0,
        DTRes *& output1,
        const Structure ** inputs,
        size_t numInputs, size_t numOutputs,
        size_t * outRows, size_t * outCols,
        int64_t * splits, int64_t * combines,
        const char * irCode,
        DCTX(ctx)
) {
    assert(numOutputs == 2 && "FIXME: lowered to wrong kernel");

    auto wrapper = std::make_unique<DistributedWrapper<DTRes>>(ctx);

    DTRes*** res = new DTRes**[2];
    res[0] = &output0;
    res[1] = &output1;
    wrapper->execute(irCode, res, inputs, numInputs, numOutputs, outRows, outCols, 
            reinterpret_cast<VectorSplit *>(splits), reinterpret_cast<VectorCombine *>(combines));
}