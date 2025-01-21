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

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "runtime/local/context/DaphneContext.h"
#include "runtime/local/datastructures/CSRMatrix.h"

template <typename T>
inline StridedMemRefType<T, 1> convertCSRMatrixToValuesMemRef(const CSRMatrix<T> *input, DCTX(ctx)) {
    StridedMemRefType<T, 1> valuesMemRef{};

    valuesMemRef.basePtr = input->getValuesSharedPtr().get();
    valuesMemRef.data = valuesMemRef.basePtr;
    valuesMemRef.offset = 0;
    valuesMemRef.sizes[0] = input->getNumNonZeros(); // Is numRowsAllocated needed to account for views?
    valuesMemRef.strides[0] = 1;

    input->increaseRefCounter();

    return valuesMemRef;
}
