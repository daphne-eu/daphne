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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_CASTSCAOBJ_H
#define SRC_RUNTIME_LOCAL_KERNELS_CASTSCAOBJ_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename VTArg>
struct CastScaObj {
    static void apply(DTRes *& res, const VTArg arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, typename VTArg>
void castScaObj(DTRes *& res, const VTArg arg, DCTX(ctx)) {
    CastScaObj<DTRes, VTArg>::apply(res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- Scalar
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
struct CastScaObj<DenseMatrix<VTRes>, VTArg> {
    static void apply(DenseMatrix<VTRes> *& res, const VTArg arg, DCTX(ctx)) {
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(1, 1, false);
        *res->getValues() = static_cast<VTRes>(arg);
    }
};

// ----------------------------------------------------------------------------
// Frame <- Scalar
// ----------------------------------------------------------------------------

template<typename VTArg>
struct CastScaObj<Frame, VTArg> {
    static void apply(Frame *& res, const VTArg arg, DCTX(ctx)) {
        auto col = DataObjectFactory::create<DenseMatrix<VTArg>>(1, 1, false);
        *col->getValues() = arg;
        std::vector<Structure *> cols = {col};
        res = DataObjectFactory::create<Frame>(cols, nullptr);
        
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_CASTSCAOBJ_H