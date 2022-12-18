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
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstddef>

// TODO: This obviously will be templated once genKernelInst.py is fixed

// TODO: may need to pass DM/StridedMemRefType as value
inline void getDenseMatrixFromMemRef(DenseMatrix<double> *&res, StridedMemRefType<double, 2>* memRef, DCTX(ctx))
{
    std::cout << "in getDenseMatrixFromMemRef call\n";
    // DenseMatrix<double> *res = new DenseMatrix<double>(memRef->basePtr);

    if(res == nullptr)
        res = DataObjectFactory::create<DenseMatrix<double>>(memRef->basePtr);
    else
        throw std::runtime_error("DenseMatrix already exists for memref?\n");
}

inline StridedMemRefType<float, 2> getMemRefDenseMatrix(
    const DenseMatrix<float> *input, DCTX(ctx)) {

    StridedMemRefType<float, 2> memRef{};
    memRef.basePtr = input->getValuesSharedPtr().get();
    memRef.data = memRef.basePtr;
    memRef.offset = 0;
    memRef.strides[0] = 1;
    memRef.sizes[0] = input->getNumRows();
    memRef.sizes[1] = input->getNumCols();

    return memRef;
}

inline StridedMemRefType<double, 2> getMemRefDenseMatrix(
    const DenseMatrix<double> *input, DCTX(ctx)) {

    StridedMemRefType<double, 2> memRef{};
    memRef.basePtr = input->getValuesSharedPtr().get();
    memRef.data = memRef.basePtr;
    memRef.offset = 0;
    memRef.strides[0] = 1;
    memRef.sizes[0] = input->getNumRows();
    memRef.sizes[1] = input->getNumCols();

    return memRef;
}

#endif  // SRC_RUNTIME_LOCAL_KERNELS_GETDATAPOINTER_H
