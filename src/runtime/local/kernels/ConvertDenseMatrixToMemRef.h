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

#pragma once

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "runtime/local/datastructures/DenseMatrix.h"

template <typename T>
inline StridedMemRefType<T, 2> convertDenseMatrixToMemRef(
    const DenseMatrix<T> *input, DCTX(ctx)) {
    StridedMemRefType<T, 2> memRef{};
    memRef.basePtr = input->getValuesSharedPtr().get();
    memRef.data = memRef.basePtr;
    memRef.offset = 0;
    memRef.sizes[0] = input->getNumRows();
    memRef.sizes[1] = input->getNumCols();

    // TODO(phil): needs to be calculated for non row-major memory layouts
    memRef.strides[0] = input->getNumCols();
    memRef.strides[1] = 1;
    input->increaseRefCounter();

    return memRef;
}
