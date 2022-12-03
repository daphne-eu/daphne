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

#include <cstdint>
#include <memory>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Structure.h>
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstddef>
#include <vector>

// ****************************************************************************
// Convenience function
// ****************************************************************************

StridedMemRefType<float, 1> getDataPointer(const DenseMatrix<float> *input, DCTX(ctx)) {
// template<typename VT, int N>
// StridedMemRefType<VT, N> _getDataPointer__StridedMemRefType___DenseMatrix_double(const DenseMatrix<float> *input, DCTX(ctx)) {
// MemRefDescriptor
// uint64_t getDataPointer(const DenseMatrix<VT>* input, DCTX(ctx)) {
    StridedMemRefType<float, 1> memRef;
    memRef.basePtr = input->getValuesSharedPtr().get();
    memRef.data = input->getValuesSharedPtr().get();
    memRef.offset = 0;
    memRef.strides[0] = 1;
    memRef.sizes[0] = 3;

    return memRef;
    // return reinterpret_cast<uint64_t>(input->getValuesSharedPtr().get());
}

StridedMemRefType<double, 2> getMemRefDenseMatrix(
    const DenseMatrix<double> *input, DCTX(ctx)) {
    // template<typename VT, int N>
    // StridedMemRefType<VT, N>
    // _getDataPointer__StridedMemRefType___DenseMatrix_double(const
    // DenseMatrix<float> *input, DCTX(ctx)) {

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
