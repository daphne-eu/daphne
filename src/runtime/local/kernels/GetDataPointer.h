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

inline void convertMemRefToDenseMatrix(DenseMatrix<double> *&result,
                                       double *basePtr, size_t offset,
                                       size_t size0, size_t size1,
                                       size_t stride0, size_t stride1,
                                       DCTX(ctx)) {
    std::cout << "convertMemRefToDenseMatrix(result: " << result
              << ", memRef: " << basePtr << ", ctx: " << ctx << ")\n";

    std::shared_ptr<double[]> ptr(basePtr);
    result = DataObjectFactory::create<DenseMatrix<double>>(size0, size1, ptr);
    result->increaseRefCounter();
}

inline StridedMemRefType<double, 2> convertDenseMatrixToMemRef(
    const DenseMatrix<double> *input, DCTX(ctx)) {
    StridedMemRefType<double, 2> memRef{};
    memRef.basePtr = input->getValuesSharedPtr().get();
    memRef.data = memRef.basePtr;
    memRef.offset = 0;
    memRef.sizes[0] = input->getNumRows();
    memRef.sizes[1] = input->getNumCols();

    // TODO(phil): does not make a difference? maybe the mlir/llvm inferes
    // something since it's statically typed
    memRef.strides[0] = input->getNumCols();
    memRef.strides[1] = 1;  // TODO(phil): needs to be calculated for non
                            // row-major memory layouts
    input->increaseRefCounter();

    return memRef;
}

inline void convertDenseMatrixToMemRef(StridedMemRefType<float, 2> *&result,
                                       const DenseMatrix<float> *input,
                                       DCTX(ctx)) {
    std::cout << "convertDenseMatrixToMemRef(result: " << result
              << ", input: " << input << ", ctx: " << ctx << ")\n";
    result = new StridedMemRefType<float, 2>();
    result->basePtr = input->getValuesSharedPtr().get();
    result->data = result->basePtr;
    result->offset = 0;
    result->sizes[0] = input->getNumRows();
    result->sizes[1] = input->getNumCols();
    result->strides[0] = input->getNumCols();
    result->strides[1] = 0;
    input->increaseRefCounter();
}
#endif  // SRC_RUNTIME_LOCAL_KERNELS_GETDATAPOINTER_H
