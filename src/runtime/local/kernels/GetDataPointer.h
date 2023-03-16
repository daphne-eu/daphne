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

#include "mlir/ExecutionEngine/CRunnerUtils.h"

template <typename T>
inline void convertMemRefToDenseMatrix(DenseMatrix<T> *&result, T *basePtr,
                                       size_t offset, size_t size0,
                                       size_t size1, size_t stride0,
                                       size_t stride1, DCTX(ctx)) {
    auto no_op_deleter = [](T*){};
    std::shared_ptr<T[]> ptr(basePtr, no_op_deleter);
    result = DataObjectFactory::create<DenseMatrix<T>>(size0, size1, ptr);
    result->increaseRefCounter();
}

template <typename T>
inline StridedMemRefType<T, 2> convertDenseMatrixToMemRef(
    const DenseMatrix<T> *input, DCTX(ctx)) {
    StridedMemRefType<T, 2> memRef{};
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

#endif  // SRC_RUNTIME_LOCAL_KERNELS_GETDATAPOINTER_H
/* since the python generator for kernels.cpp does not correctly work for these
 calls here's the backup for the manual ones (kernels.cpp not in git cause it's
 under build/)
 *
    void
 _getDenseMatrixFromMemRef__DenseMatrix_float__size_t__size_t__size_t__size_t__size_t__size_t(DenseMatrix<float>**res,
        float *basePtr, size_t offset, size_t size0, size_t size1,
        size_t stride0, size_t stride1, DaphneContext *ctx) {
        convertMemRefToDenseMatrix<float>(*res, basePtr, offset, size0, size1,
 stride0, stride1, ctx);
    }
    void
 _getDenseMatrixFromMemRef__DenseMatrix_double__size_t__size_t__size_t__size_t__size_t__size_t(DenseMatrix<double>**res,
        double *basePtr, size_t offset, size_t size0, size_t size1,
        size_t stride0, size_t stride1, DaphneContext *ctx) {
        convertMemRefToDenseMatrix<double>(*res, basePtr, offset, size0, size1,
 stride0, stride1, ctx);
    }

    void
 _getMemRefDenseMatrix__StridedMemRefType___DenseMatrix_float(StridedMemRefType<float,2>
 *res, const DenseMatrix<float>* input, DCTX(ctx)) { *res =
 convertDenseMatrixToMemRef<float>(input, ctx);
    }
    void
 _getMemRefDenseMatrix__StridedMemRefType___DenseMatrix_double(StridedMemRefType<double,2>
 *res, const DenseMatrix<double>* input, DCTX(ctx)) { *res =
 convertDenseMatrixToMemRef<double>(input, ctx);
    }
 *
 */
