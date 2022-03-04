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
//#include <runtime/local/vectorized/MTWrapper.h>
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
    std::cerr << "distributedPipeline-kernel received the following information" << std::endl;
            
    std::cerr << "\t" << numInputs << " inputs:" << std::endl;
    for(size_t i = 0; i < numInputs; i++)
        std::cerr << "\t\t" << i << ": split mode " << splits[i] << std::endl;
    
    std::cerr << "\t" << numOutputs << " outputs:" << std::endl;
    for(size_t i = 0; i < numOutputs; i++)
        std::cerr << "\t\t" << i << ": combine mode " << combines[0]
                << ", size " << outRows[0] << "x" << outCols[0] << std::endl;
    
    std::cerr << "\tDaphneIR code: " << std::endl << "```" << std::endl << irCode << "```" << std::endl;
    
    std::cerr << "\tNote that the program will most likely crash soon, "
        "since distributedPipeline does not set the outputs yet" << std::endl;
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
    std::cerr << "distributedPipeline-kernel received the following information" << std::endl;
            
    std::cerr << "\t" << numInputs << " inputs:" << std::endl;
    for(size_t i = 0; i < numInputs; i++)
        std::cerr << "\t\t" << i << ": split mode " << splits[i] << std::endl;
    
    std::cerr << "\t" << numOutputs << " outputs:" << std::endl;
    for(size_t i = 0; i < numOutputs; i++)
        std::cerr << "\t\t" << i << ": combine mode " << combines[0]
                << ", size " << outRows[0] << "x" << outCols[0] << std::endl;
    
    std::cerr << "\tDaphneIR code: " << std::endl << "\t```" << std::endl << irCode << "\t```" << std::endl;
    
    std::cerr << "\tNote that the program will most likely crash soon, "
        "since distributedPipeline does not set the outputs yet" << std::endl;
}