/*
 * Copyright 2022 The DAPHNE Consortium
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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_GETDATAPOINTER_H
#define SRC_RUNTIME_LOCAL_KERNELS_GETDATAPOINTER_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/Frame.h>

#include <cstddef>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

// TODO MSC: This obviously will be templated once genKernelInst.py is fixed

// TODO MSC: may need to pass DM/StridedMemRefType as value
inline void getDenseMatrixFromMemRef(DenseMatrix<double> *&res,
                                     StridedMemRefType<double, 2> *memRef,
                                     DCTX(ctx)) {
    // DenseMatrix<double> *res = new DenseMatrix<double>(memRef->basePtr);

    // if(res == nullptr) {
#if 0
    std::cout << "MemRef -> DenseMatrix:\nMemRef{basePtr: " << memRef->basePtr
              << ", data: " << memRef->data << "}\n\n";

    for (size_t r = 0; r < 10; r++) {
        for (size_t c = 0; c < 10; c++) {
            // TODO MSC: Check for row/column major order on access
            std::cout << memRef->basePtr[10 * c + r];
            // printValue(os, get(r, c));
            if (c < 10 - 1) std::cout << ' ';
        }
        std::cout << std::endl;
    }
#endif
    res = DataObjectFactory::create<DenseMatrix<double>>(memRef->basePtr);
    // }
    // else
    //     throw std::runtime_error("DenseMatrix already exists for memref?\n");
}

// inline StridedMemRefType<float, 2> getMemRefDenseMatrix(
//     const DenseMatrix<float> *input, DCTX(ctx)) {
//
//     StridedMemRefType<float, 2> memRef{};
//     memRef.basePtr = input->getValuesSharedPtr().get();
//     memRef.data = memRef.basePtr;
//     memRef.offset = 0;
//     memRef.strides[0] = 0;
//     memRef.strides[1] = 0;
//     memRef.sizes[0] = input->getNumRows();
//     memRef.sizes[1] = input->getNumCols();
//
//     return memRef;
// }

inline void getMemRefDenseMatrix(StridedMemRefType<double, 2> *&result,
                                 const DenseMatrix<double> *input, DCTX(ctx)) {
    result = new StridedMemRefType<double, 2>();
    result->basePtr = input->getValuesSharedPtr().get();
    result->data = result->basePtr;
    result->offset = 0;
    result->strides[0] = 0;
    result->strides[1] = 0;
    result->sizes[0] = input->getNumRows();
    result->sizes[1] = input->getNumCols();
    input->increaseRefCounter();

#if 0
    std::cout << "DenseMatrix -> MemRef:\nMemRef{basePtr: " << result->basePtr
              << ", data: " << result->data << "}\n\n";
    for (size_t r = 0; r < 10; r++) {
        for (size_t c = 0; c < 10; c++) {
            // TODO MSC: Check for row/column major order on access
            std::cout << result->basePtr[10 * c + r];
            // printValue(os, get(r, c));
            if (c < 10 - 1) std::cout << ' ';
        }
        std::cout << std::endl;
    }
#endif
}

#endif  // SRC_RUNTIME_LOCAL_KERNELS_GETDATAPOINTER_H
