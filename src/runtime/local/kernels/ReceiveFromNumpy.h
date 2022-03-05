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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_READ_H
#define SRC_RUNTIME_LOCAL_KERNELS_READ_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/FileMetaData.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<>
struct ReceiveFromNumpy {
    static void apply(int* arg, int size) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<>
void receiveFromNumpy(int* arg, int size, DCTX(ctx)) {
    ReceiveFromNumpy<DTRes>::apply(arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct ReceiveFromNumpy<DenseMatrix<VT>> {
    static void apply(int* arg, int size, DCTX(ctx)) {
        
    }
};


#endif //SRC_RUNTIME_LOCAL_KERNELS_READ_H