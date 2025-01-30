/*
 * Copyright 2025 The DAPHNE Consortium
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

#include "runtime/local/context/DaphneContext.h"
#include "runtime/local/datastructures/CSRMatrix.h"

template <typename T>
inline void convertMemRefToCSRMatrix(CSRMatrix<T> *&result, 
    size_t baseValuesPtr, size_t baseColIdxsPtr, size_t baseRowOffsetsPtr, 
    size_t maxNumRows, size_t numCols, size_t maxNumNonZeros, DCTX(ctx)) 
{
    auto no_op_deleter_1 = [](T *) {};
    auto no_op_deleter_2 = [](long unsigned int *) {};
    T *valuePtr = reinterpret_cast<T *>(baseValuesPtr);
    long unsigned int *colIdxsPtr = reinterpret_cast<long unsigned int *>(baseColIdxsPtr);
    long unsigned int *rowOffsetsPtr = reinterpret_cast<long unsigned int *>(baseRowOffsetsPtr);
    std::shared_ptr<T[]> ptrValues(valuePtr, no_op_deleter_1);
    std::shared_ptr<long unsigned int[]> ptrColIdxs(colIdxsPtr, no_op_deleter_2);
    std::shared_ptr<long unsigned int[]> ptrRowOffsets(rowOffsetsPtr, no_op_deleter_2);
    result = DataObjectFactory::create<CSRMatrix<T>>(maxNumRows, numCols, maxNumNonZeros, false);

    result->getValuesSharedPtr() = ptrValues;
    result->getColIdxsSharedPtr() = ptrColIdxs;
    result->getRowOffsetsSharedPtr() = ptrRowOffsets;

}
