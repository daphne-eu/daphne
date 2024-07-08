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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_RANDTENSOR_H
#define SRC_RUNTIME_LOCAL_KERNELS_RANDTENSOR_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ContiguousTensor.h>

// TODO: 1. remove restriction to 3D, simply pass bounds via e.g. 2 ptrs
//       2. add variant for chunked tensor
//
// NOTE: Im not addressing the todos now because this is/was used in the initial dsl integration of tensors and was
//       simplified for that.

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename VTArg>
struct RandTensor {
    static void apply(DTRes *& res, size_t numX, size_t numY, size_t numZ, VTArg min, VTArg max, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, typename VTArg>
void randTensor3D(DTRes *& res, size_t numX, size_t numY, size_t numZ, VTArg min, VTArg max, DCTX(ctx)) {
    RandTensor<DTRes, VTArg>::apply(res, numX, numY, numZ, min, max, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<typename VT>
struct RandTensor<ContiguousTensor<VT>, VT> {
    static void apply(ContiguousTensor<VT> *& res, size_t numX, size_t numY, size_t numZ, VT min, VT max, DCTX(ctx)) {
        std::vector<size_t> shape = {numX, numY, numZ};
        res = DataObjectFactory::create<ContiguousTensor<VT>>(shape, InitCode::RAND);
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_RANDTENSOR_H
